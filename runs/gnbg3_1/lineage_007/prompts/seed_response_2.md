```python
import numpy as np


class CovarianceRankAdaptationOptimizer:
    """
    A population-based optimizer using rank-based covariance adaptation
    with archive-guided exploration and adaptive restarts.
    Different from CovarianceGuidedDE: uses eigendecomposition-free
    covariance updates, archive-based mutation, and ring topology.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(kwargs.get('np', 8 * dim), 300)
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        
        # Covariance adaptation state
        self.cov = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.step_size = 0.5 * 100 / np.sqrt(dim)
        self.mean = np.zeros(dim)
        
        # Adaptation tracking
        self.cov_rank_mu = np.zeros((dim, dim))
        self.learning_rate_cov = 2.0 / (dim + 2.0)
        self.learning_rate_step = 0.1
        self.attack_counter = 0
        self.last_improvement_gen = 0
        
        # Results tracking
        self.f_opt = np.inf
        self.x_opt = None
        self.gen_count = 0
        
        # Archive for elite solutions
        self.archive_size = min(5 * self.np, 500)
        self.archive = np.zeros((self.archive_size, dim))
        self.archive_fitness = np.full(self.archive_size, np.inf)
        self.archive_count = 0
        
        # Diversity and stagnation
        self.diversity_history = []
        self.stagnation_count = 0
        self.improvement_history = []
        
        # Ring migration
        self.migration_gap = kwargs.get('migration_gap', 7)
    
    def _initialize_population(self):
        """Sample initial population uniformly within bounds."""
        pop = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.np, self.dim)
        )
        return pop
    
    def _clip_to_bounds(self, pop):
        """Clip all candidates to search bounds."""
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])
    
    def _sample_gaussian_batch(self, n_samples):
        """Sample from current Gaussian distribution (mean, cov, step)."""
        L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(self.dim))
        z = np.random.randn(n_samples, self.dim)
        samples = self.mean + self.step_size * (z @ L.T)
        return samples
    
    def _mutate_with_archive(self, pop, fitness):
        """Generate mutations using archive-guided differential variation."""
        n = len(pop)
        trials = np.zeros((n, self.dim))
        
        # Use archive if available
        use_archive = self.archive_count > 2
        
        for i in range(n):
            if use_archive and np.random.rand() < 0.5:
                # Archive-based mutation: best + F * (r2 - r3)
                idx = np.random.choice(min(self.archive_count, len(self.archive)), 3, replace=False)
                base = self.archive[idx[0]]
                diff = self.archive[idx[1]] - self.archive[idx[2]]
                F = np.random.uniform(0.3, 1.2)
                trials[i] = base + F * diff + 0.05 * np.random.randn(self.dim)
            else:
                # Covariance-guided mutation
                trials[i] = self._sample_gaussian_batch(1)[0]
        
        return trials
    
    def _crossover_batch(self, targets, trials, cr):
        """Binomial crossover combining target and trial vectors."""
        mask = np.random.rand(len(targets), self.dim) < cr
        offspring = np.where(mask, trials, targets)
        return offspring
    
    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        """Select survivors via elitist comparison (minimization)."""
        improved = trial_fitness < fitness
        new_pop = np.where(improved[:, np.newaxis], trials, pop)
        new_fitness = np.where(improved, trial_fitness, fitness)
        return new_pop, new_fitness
    
    def _adapt_step_size(self, success_rate, fitness_variance):
        """Adapt step size using success rate and fitness variance."""
        target_success = 0.4
        lr = self.learning_rate_step + 0.15 * min(1.0, fitness_variance / (1.0 + abs(np.mean(self.improvement_history[-10:]) if self.improvement_history else 0.0)))
        
        if success_rate > target_success:
            self.step_size *= (1 + lr * (success_rate - target_success))
        else:
            self.step_size *= (1 - lr * (target_success - success_rate))
        
        self.step_size = np.clip(self.step_size, 1e-10, 100)
    
    def _adapt_covariance(self, z_offspring, z_mean):
        """Rank-one and rank-mu covariance update from CMA-ES."""
        alpha = 1.0 / (self.dim + 1.0)
        c1 = 2.0 / (self.dim ** 2 + 6.0)
        cmu = self.learning_rate_cov
        
        # Update evolution path pc
        self.pc = (1 - alpha) * self.pc + np.sqrt(alpha * (2 - alpha)) * z_mean
        
        # Rank-one covariance update
        rank_one = c1 * np.outer(self.pc, self.pc)
        
        # Rank-mu update using top individuals
        n_top = max(1, len(z_offspring) // 2)
        top_indices = np.argsort(np.linalg.norm(z_offspring, axis=1))[:n_top]
        z_top = z_offspring[top_indices]
        weights = np.ones(n_top) / n_top
        
        rank_mu = cmu * sum(w * np.outer(z, z) for z, w in zip(z_top, weights))
        
        # Apply update
        self.cov = (1 - c1 - cmu) * self.cov + rank_one + rank_mu
        self.cov = 0.5 * (self.cov + self.cov.T)
    
    def _adapt_covariance_variant_01(self, z_offspring):
        """Variant 01: Weighted rank-mu update using fitness improvement."""
        if len(z_offspring) == 0:
            return
        
        weights = np.ones(len(z_offspring)) / len(z_offspring)
        z_mean_weighted = np.sum(z_offspring * weights[:, np.newaxis], axis=0)
        
        cmu = 2.0 / (self.dim + 2.0)
        self.cov = (1 - cmu) * self.cov + cmu * np.outer(z_mean_weighted, z_mean_weighted)
    
    def _adapt_covariance_variant_08(self, z_offspring):
        """Variant 08: Eigenvalue perturbation for escaping local optima."""
        if len(z_offspring) == 0:
            return
        
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        epsilon = 0.05 + 0.1 * min(1.0, self.stagnation_count / 5.0)
        eigenvalues = eigenvalues + epsilon * np.random.randn(self.dim)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Move mean toward best known solution
        if self.x_opt is not None:
            self.mean = 0.85 * self.mean + 0.15 * self.x_opt
    
    def _adapt_covariance_variant_09(self, z_offspring, trial_fitness, fitness):
        """Variant 09: Fitness-weighted covariance adaptation."""
        if len(z_offspring) == 0:
            return
        
        improvements = fitness - trial_fitness
        positive_mask = improvements > 0
        positive_count = np.sum(positive_mask)
        
        if positive_count == 0:
            return
        
        z_positive = z_offspring[positive_mask]
        weights = improvements[positive_mask]
        weights = weights / (np.sum(weights) + 1e-10)
        
        z_mean_weighted = np.sum(z_positive * weights[:, np.newaxis], axis=0)
        cmu = 2.0 / (self.dim + 2.0)
        self.cov = (1 - cmu) * self.cov + cmu * np.outer(z_mean_weighted, z_mean_weighted)
    
    def _compute_diversity(self, pop):
        """Compute population diversity as mean distance to centroid."""
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances)
    
    def _update_archive(self, pop, fitness):
        """Update archive with best solutions found so far."""
        combined = np.vstack([self.archive[:self.archive_count], pop])
        combined_fitness = np.concatenate([self.archive_fitness[:self.archive_count], fitness])
        
        indices = np.argsort(combined_fitness)[:self.archive_size]
        self.archive = combined[indices]
        self.archive_fitness = combined_fitness[indices]
        self.archive_count = min(self.archive_count + len(pop), self.archive_size)
    
    def _migrate_ring(self, pop, fitness):
        """Ring topology migration for island-model diversity."""
        if self.gen_count % self.migration_gap != 0:
            return pop, fitness
        
        n_islands = max(1, self.np // 30)
        island_size = self.np // n_islands
        
        for i in range(n_islands):
            start = i * island_size
            end = start + island_size if i < n_islands - 1 else self.np
            
            next_start = (i + 1) * island_size
            next_end = next_start + island_size if i < n_islands - 2 else self.np
            
            if next_end > next_start and end > start:
                best_idx = start + np.argmin(fitness[start:end])
                worst_idx = next_start + np.argmax(fitness[next_start:next_end])
                pop[worst_idx] = pop[best_idx].copy()
                fitness[worst_idx] = fitness[best_idx]
        
        return pop, fitness
    
    def _restart_if_stagnant(self, pop, fitness):
        """Detect stagnation and perform adaptive restart."""
        diversity = self._compute_diversity(pop)
        self.diversity_history.append(diversity)
        
        if len(self.diversity_history) > 15:
            recent = self.diversity_history[-15:]
            relative_std = np.std(recent) / (np.mean(recent) + 1e-10)
            if relative_std < 0.05:
                self.stagnation_count += 1
            else:
                self.stagnation_count = max(0, self.stagnation_count - 1)
        
        should_restart = (
            self.stagnation_count > 2 or
            diversity < 0.05 or
            np.any(np.isnan(fitness)) or
            np.any(np.isinf(fitness))
        )
        
        if should_restart:
            best_idx = np.argmin(fitness)
            best, best_fit = pop[best_idx].copy(), fitness[best_idx]
            
            pop = self._initialize_population()
            pop[0] = best
            pop = self._clip_to_bounds(pop)
            
            self.cov = np.eye(self.dim)
            self.step_size = 0.5 * 100 / np.sqrt(self.dim)
            self.pc = np.zeros(self.dim)
            self.diversity_history = []
            self.stagnation_count = 0
            
            return pop, True
        
        return pop, False
    
    def _sample_trials_batch(self, pop, fitness, mode):
        """Sample batch of trial solutions using multiple strategies."""
        n_trials = self.np
        trials = np.zeros((n_trials, self.dim))
        
        for i in range(n_trials):
            if mode == 'exploit' or np.random.rand() < 0.7:
                trials[i] = self._sample_gaussian_batch(1)[0]
            else:
                idx = np.random.choice(min(5, len(pop)), 3, replace=False)
                base, a, b = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                diff = a - b
                trials[i] = base + self.step_size * diff + 0.1 * np.random.randn(self.dim)
        
        return trials
    
    def _build_trial_batch(self, pop, fitness):
        """Build complete batch of trial solutions with adaptive strategy."""
        if len(self.improvement_history) > 5:
            recent = [x[1] for x in self.improvement_history[-5:]]
            success_rate = np.mean(recent)
            mode = 'exploit' if success_rate > 0.3 else 'explore'
        else:
            mode = 'explore'
        
        trials = self._sample_trials_batch(pop, fitness, mode)
        trials = self._mutate_with_archive(pop, fitness)
        
        cr = 0.85
        for i in range(len(trials)):
            trials[i] = self._crossover_batch(pop[i:i+1], trials[i:i+1], cr)[0]
        
        return self._clip_to_bounds(trials)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating all batch operations."""
        pop = self._initialize_population()
        pop = self._clip_to_bounds(pop)
        
        fitness = func(pop)
        if len(fitness) < self.np:
            pop, fitness = pop[:len(fitness)], fitness[:len(fitness)]
        
        self.f_opt = np.min(fitness)
        self.x_opt = pop[np.argmin(fitness)].copy()
        
        self.mean = np.mean(pop, axis=0)
        self.cov = np.cov(pop.T) + 1e-8 * np.eye(self.dim)
        
        self.gen_count = 0
        self.improvement_history = []
        
        while not stopping_condition():
            self.gen_count += 1
            
            trials = self._build_trial_batch(pop, fitness)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trial_fitness) == 0:
                    break
            
            if stopping_condition():
                break
            
            improvements = trial_fitness < fitness
            n_improvements = np.sum(improvements)
            self.improvement_history.append((self.gen_count, n_improvements / len(fitness)))
            
            self._update_archive(trials, trial_fitness)
            
            pop, fitness = self._select_survivors_batch(pop, fitness, trials, trial_fitness)
            
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < self.f_opt:
                self.f_opt = fitness[gen_best_idx]
                self.x_opt = pop[gen_best_idx].copy()
                self.last_improvement_gen = self.gen_count
            
            z_offspring = (trials - self.mean) / (self.step_size + 1e-10)
            z_mean = np.mean(z_offspring, axis=0)
            
            fitness_var = np.var(fitness)
            self._adapt_step_size(n_improvements / len(fitness), fitness_var)
            
            if n_improvements > 0:
                self._adapt_covariance(z_offspring, z_mean)
            
            stale_generations = self.gen_count - self.last_improvement_gen
            if stale_generations > 15:
                self._adapt_covariance_variant_01(z_offspring)
            if stale_generations > 25:
                self._adapt_covariance_variant_08(z_offspring)
            if n_improvements > 0 and stale_generations < 15:
                self._adapt_covariance_variant_09(z_offspring, trial_fitness, fitness)
            
            pop, fitness = self._migrate_ring(pop, fitness)
            
            pop, restarted = self._restart_if_stagnant(pop, fitness)
            if restarted:
                fitness = func(pop)
                if len(fitness) < self.np:
                    pop, fitness = pop[:len(fitness)], fitness[:len(fitness)]
            
            self.mean = np.mean(pop, axis=0)
            
            if len(self.improvement_history) > 100:
                self.improvement_history = self.improvement_history[-100:]
        
        return self.f_opt, self.x_opt
```