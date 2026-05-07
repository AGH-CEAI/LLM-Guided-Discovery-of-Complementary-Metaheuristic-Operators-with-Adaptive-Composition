```python
import numpy as np


class CovarianceGuidedDifferentialEvolution:
    """
    Differential Evolution with covariance-guided mutation and adaptive operator selection.
    Uses multiple mutation strategies with rewards-based operator selection.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size 4*dim to 300
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Step size (F) and crossover rate (CR) with per-dimension adaptation
        self.f_base = 0.5
        self.cr_base = 0.5
        self.f_values = np.ones(self.np) * self.f_base
        self.cr_values = np.ones(self.np) * self.cr_base
        
        # Covariance matrix for guiding mutation
        self.cov_mean = np.zeros(dim)
        self.cov_matrix = np.eye(dim)
        self.cov_adapt_rate = 0.01
        
        # Archive for mutation (successful individuals from history)
        self.archive = np.zeros((0, dim))
        self.max_archive_size = self.np
        
        # Operator selection: 4 mutation strategies
        self.operator_names = ['current_to_pbest', 'rand', 'best_jitter', 'cov_guided']
        self.operator_rewards = np.zeros(4)
        self.operator_counts = np.ones(4)
        self.current_operator = 0
        self.operator_selection_rate = 0.1
        
        # Diversity tracking
        self.diversity_history = []
        self.stagnation_counter = 0
        self.max_stagnation = 50
        
        # Best solution tracking
        self.best_fitness_history = []
        
    def _initialize_population(self, func):
        """Initialize population with Latin Hypercube Sampling."""
        pop = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            pop[:, d] = np.random.uniform(self.lower_bound, self.upper_bound, self.np)
            pop[:, d] = np.sort(pop[:, d])
            idx = np.random.permutation(self.np)
            pop[:, d] = pop[idx, d]
        
        pop = np.clip(pop, self.lower_bound, self.upper_bound)
        fitness = func(pop)
        
        # Initialize covariance mean from population
        self.cov_mean = pop.mean(axis=0)
        
        return pop, fitness
    
    def _select_operator(self):
        """Thompson sampling for operator selection."""
        # UCB-based selection with rewards
        ucb_scores = (self.operator_rewards / np.maximum(self.operator_counts, 1) + 
                      0.5 * np.sqrt(np.log(self.operator_counts.sum() + 1) / 
                                   np.maximum(self.operator_counts, 1)))
        
        if np.random.random() < self.operator_selection_rate:
            self.current_operator = np.random.randint(0, 4)
        else:
            self.current_operator = np.argmax(ucb_scores)
        
        return self.current_operator
    
    def _mutate_current_to_pbest(self, population, fitness, trial_indices):
        """Current-to-pbest/1 mutation with archive."""
        np_op = len(trial_indices)
        p_best_ratio = 0.1
        n_best = max(1, int(p_best_ratio * self.np))
        pbest_indices = np.argsort(fitness)[:n_best]
        pbest = population[np.random.choice(pbest_indices)]
        
        # Build candidate pool with archive
        candidates = population
        if len(self.archive) > 0:
            candidates = np.vstack([candidates, self.archive])
        
        # Select 3 distinct indices different from target
        valid_mask = np.ones(len(candidates), dtype=bool)
        
        mutated = np.zeros((np_op, self.dim))
        for i, idx in enumerate(trial_indices):
            valid_mask[idx] = False
            
            # Select 3 distinct from pool
            available = np.where(valid_mask)[0]
            if len(available) < 3:
                sel = np.random.choice(len(candidates), 3, replace=False)
            else:
                sel = np.random.choice(available, 3, replace=False)
            
            # Current-to-pbest/1/bin
            f_scale = self.f_base * (1 + 0.5 * np.random.randn())
            f_scale = np.clip(f_scale, 0.1, 2.0)
            
            mutated[i] = (population[idx] + 
                         f_scale * (pbest - population[idx]) + 
                         f_scale * (candidates[sel[0]] - candidates[sel[1]]))
            
            valid_mask[idx] = True
        
        return mutated
    
    def _mutate_rand(self, population, trial_indices):
        """Classic rand/1 mutation."""
        np_op = len(trial_indices)
        candidates = np.vstack([population, self.archive]) if len(self.archive) > 0 else population
        n_cand = len(candidates)
        
        mutated = np.zeros((np_op, self.dim))
        for i in range(np_op):
            sel = np.random.choice(n_cand, 3, replace=False)
            f_scale = np.clip(self.f_base + 0.3 * np.random.randn(), 0.1, 2.0)
            mutated[i] = candidates[sel[0]] + f_scale * (candidates[sel[1]] - candidates[sel[2]])
        
        return mutated
    
    def _mutate_best_jitter(self, population, fitness, trial_indices):
        """Best/1 mutation with per-dimension jitter."""
        np_op = len(trial_indices))
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        candidates = np.vstack([population, self.archive]) if len(self.archive) > 0 else population
        n_cand = len(candidates)
        
        mutated = np.zeros((np_op, self.dim))
        for i in range(np_op):
            sel = np.random.choice(n_cand, 2, replace=False)
            jitter = np.random.uniform(0.7, 1.3, self.dim)
            f_scale = np.clip(self.f_base * jitter, 0.1, 2.0)
            mutated[i] = best + f_scale * (candidates[sel[0]] - candidates[sel[1]])
        
        return mutated
    
    def _mutate_cov_guided(self, population, fitness, trial_indices):
        """Covariance-guided mutation using eigendecomposition."""
        np_op = len(trial_indices)
        best_idx = np.argmin(fitness)
        base = population[best_idx]
        
        # Eigendecomposition of covariance matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            rotation = eigenvectors.T
        except np.linalg.LinAlgError:
            rotation = np.eye(self.dim)
            sqrt_eigenvalues = np.ones(self.dim)
        
        mutated = np.zeros((np_op, self.dim))
        for i in range(np_op):
            # Sample from covariance ellipsoid
            z = np.random.randn(self.dim)
            step = rotation.T @ (sqrt_eigenvalues * z)
            f_scale = np.clip(self.f_base * 0.8, 0.1, 1.5)
            mutated[i] = base + f_scale * step
        
        return mutated
    
    def _mutate_batch(self, population, fitness, trial_indices):
        """Batch mutation dispatcher."""
        operator = self._select_operator()
        
        if operator == 0:
            mutants = self._mutate_current_to_pbest(population, fitness, trial_indices)
        elif operator == 1:
            mutants = self._mutate_rand(population, trial_indices)
        elif operator == 2:
            mutants = self._mutate_best_jitter(population, fitness, trial_indices)
        else:
            mutants = self._mutate_cov_guided(population, fitness, trial_indices)
        
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        return mutants
    
    def _crossover_batch(self, population, mutants, trial_indices):
        """Binomial crossover with per-individual CR."""
        np_op = len(trial_indices)
        trials = population[trial_indices].copy()
        
        cr_values = np.clip(self.cr_values[trial_indices] + 0.1 * np.random.randn(np_op), 0.0, 1.0)
        
        for i in range(np_op):
            j_rand = np.random.randint(self.dim)
            mask = np.random.random(self.dim) < cr_values[i]
            mask[j_rand] = True
            trials[i, ~mask] = mutants[i, ~mask]
        
        trials = np.clip(trials, self.lower_bound, self.upper_bound)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with archive update."""
        # Handle NaN in trial_fitness
        valid_mask = ~np.isnan(trial_fitness)
        
        if not np.any(valid_mask):
            return population, fitness
        
        # Create new population array
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Find improvements
        improvements = valid_mask & (trial_fitness < fitness)
        
        # Update with improved individuals
        for i in np.where(improvements)[0]:
            # Archive the replaced individual
            if len(self.archive) < self.max_archive_size:
                self.archive = np.vstack([self.archive, population[i]])
            else:
                replace_idx = np.random.randint(len(self.archive))
                self.archive[replace_idx] = population[i]
            
            new_population[i] = trials[i]
            new_fitness[i] = trial_fitness[i]
        
        # Update operator rewards
        if np.any(improvements):
            self.operator_rewards[self.current_operator] += improvements.sum()
            self.operator_counts[self.current_operator] += 1
        
        return new_population, new_fitness
    
    def _adapt_step_size(self, fitness, trial_fitness):
        """Adapt F and CR based on success rate."""
        valid = ~np.isnan(trial_fitness)
        if np.sum(valid) == 0:
            return
        
        improvements = valid & (trial_fitness < fitness)
        success_rate = improvements.sum() / np.sum(valid)
        
        # Adapt F: decrease if success rate low, increase if high
        if success_rate > 0.3:
            self.f_base = min(1.0, self.f_base * 1.05)
        elif success_rate < 0.1:
            self.f_base = max(0.3, self.f_base * 0.95)
        
        # Adapt CR: increase diversity if stuck
        if success_rate < 0.05:
            self.cr_base = min(0.9, self.cr_base * 1.1)
    
    def _adapt_covariance_variant_01(self, population, fitness):
        """Adapt covariance matrix based on good solutions."""
        # Update covariance mean with momentum
        current_mean = population.mean(axis=0)
        self.cov_mean = 0.8 * self.cov_mean + 0.2 * current_mean
        
        # Update covariance matrix
        centered = population - self.cov_mean
        cov_update = (centered.T @ centered) / self.np
        
        # Blend with previous covariance
        self.cov_matrix = (1 - self.cov_adapt_rate) * self.cov_matrix + self.cov_adapt_rate * cov_update
        
        # Ensure positive definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def _adapt_covariance_variant_08(self, population, fitness):
        """Alternative covariance adaptation using best individuals."""
        n_elite = max(2, self.np // 5)
        elite_indices = np.argsort(fitness)[:n_elite]
        elite = population[elite_indices]
        
        # Covariance from elite subset
        elite_mean = elite.mean(axis=0)
        elite_centered = elite - elite_mean
        elite_cov = (elite_centered.T @ elite_centered) / n_elite
        
        # Blend with full population covariance
        self.cov_matrix = 0.7 * self.cov_matrix + 0.3 * elite_cov
    
    def _compute_diversity(self, population):
        """Compute population diversity (average pairwise distance)."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diffs = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt((diffs ** 2).sum(axis=2))
        
        # Average of upper triangle
        n = len(sample)
        mask = np.triu_indices(n, k=1)
        return distances[mask].mean()
    
    def _restart_if_stagnant(self, population, fitness, best_fitness):
        """Restart population if stagnation detected."""
        current_best = np.min(fitness)
        self.best_fitness_history.append(current_best)
        
        if len(self.best_fitness_history) > 20:
            recent_best = min(self.best_fitness_history[-20:])
            if recent_best >= best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        if self.stagnation_counter >= self.max_stagnation:
            # Random reinitialization of portion of population
            n_reset = self.np // 2
            reset_indices = np.random.choice(self.np, n_reset, replace=False)
            
            for d in range(self.dim):
                pop_col = population[reset_indices, d]
                pop_col[:] = np.random.uniform(self.lower_bound, self.upper_bound, n_reset)
            
            population[reset_indices] = np.clip(population[reset_indices], 
                                                 self.lower_bound, self.upper_bound)
            
            self.stagnation_counter = 0
            self.archive = np.zeros((0, self.dim))
            self.f_base = min(1.0, self.f_base * 1.5)
        
        return population
    
    def _update_operator_rewards(self, fitness, trial_fitness):
        """Normalize and decay operator rewards."""
        # Exponential decay
        self.operator_rewards *= 0.99
        self.operator_counts *= 0.99
        self.operator_counts = np.maximum(self.operator_counts, 0.1)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population, fitness = self._initialize_population(func)
        
        # Handle empty/budget-exhausted initialization
        if len(fitness) == 0 or np.all(np.isnan(fitness)):
            return np.inf, np.zeros(self.dim)
        
        # Truncate if needed
        if len(fitness) < self.np:
            population = population[:len(fitness)]
            fitness = fitness[:len(fitness)]
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_x = population[best_idx].copy()
        
        generation = 0
        
        while not stopping_condition():
            generation += 1
            
            # Generate trial indices
            trial_indices = np.arange(self.np)
            
            # Build trials batched
            mutants = self._mutate_batch(population, fitness, trial_indices)
            trials = self._crossover_batch(population, mutants, trial_indices)
            
            # Evaluate batch
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if len(trial_fitness) == 0:
                break
            
            # Check stopping after evaluation
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            
            # Adapt parameters
            self._adapt_step_size(fitness, trial_fitness)
            self._adapt_covariance_variant_01(population, fitness)
            self._adapt_covariance_variant_08(population, fitness)
            self._update_operator_rewards(fitness, trial_fitness)
            
            # Track best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_x = population[best_idx].copy()
            
            # Restart if stagnant
            population = self._restart_if_stagnant(population, fitness, best_fitness)
            
            # Re-evaluate restarted portion
            if self.stagnation_counter == 0 and generation > 1:
                pass  # Normal flow
            elif len(population) > 0:
                eval_mask = np.zeros(len(population), dtype=bool)
                # Fitness already updated from selection
            
            # Diversity-guided mutation rate adjustment
            diversity = self._compute_diversity(population)
            if diversity < 0.1 * self.dim:
                self.cr_base = min(0.95, self.cr_base * 1.2)
        
        return best_fitness, best_x
```