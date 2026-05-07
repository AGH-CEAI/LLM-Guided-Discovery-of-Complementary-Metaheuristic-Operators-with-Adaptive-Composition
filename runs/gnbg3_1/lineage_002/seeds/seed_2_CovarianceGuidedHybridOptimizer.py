import numpy as np


class CovarianceGuidedHybridOptimizer:
    """
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with differential mutation and weighted recombination.
    
    Key features:
    - Per-generation eigendecomposition for sampling (batch operation)
    - Rank-1 and rank-μ covariance updates
    - Cumulative step size adaptation (CSA)
    - Diversity-based restart mechanism
    - All evaluations are batched (no per-individual calls)
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = kwargs.get('np', min(10 * dim, 400))
        self.bounds = kwargs.get('bounds', (-100.0, 100.0))
        self.lower = self.bounds[0]
        self.upper = self.bounds[1]
        
        # CMA-ES inspired parameters
        self.lambda_ = self.np
        self.mu = max(1, self.np // 4)
        self.weights = self._compute_recombination_weights()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        
        # Step size and covariance
        self.sigma = 0.5 * (self.upper - self.lower) / 3.0
        self.sigma_min = 1e-10
        self.sigma_max = (self.upper - self.lower)
        
        # CSA parameters
        self.cs = (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.damps = 1.0 + max(0.0, np.sqrt(self.mu_eff) - 1.0)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        
        # Covariance adaptation
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / 
                       ((self.dim + 2.0) ** 2 + 2.0 * self.mu_eff))
        self.cov = np.eye(dim)
        self.cov_temp = np.zeros((dim, dim))
        
        # Diversity and stagnation
        self.diversity_threshold = kwargs.get('diversity_threshold', 1e-6)
        self.stagnation_window = kwargs.get('stagnation_window', 10 + int(dim / 10))
        self.min_improvement = kwargs.get('min_improvement', 1e-8)
        
        # State variables
        self.population = None
        self.fitness = None
        self.mean = None
        self.prev_mean = None
        self.best_x = None
        self.best_f = np.inf
        self.history_best = []
        self.generation = 0
        self.successful_steps = []
        
    def _compute_recombination_weights(self):
        """Compute negative and positive weights for weighted recombination."""
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = np.maximum(weights, 0.0)
        weights_sum = np.sum(weights)
        weights = weights / weights_sum
        weights[:self.mu // 2] *= -1.0
        return weights
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        self.fitness = np.full(self.np, np.inf)
        self.mean = self.population.mean(axis=0)
        self.prev_mean = self.mean.copy()
        self._clip_to_bounds(self.population)
        
    def _initialize_covariance(self):
        """Initialize covariance matrix to diagonal with appropriate variance."""
        self.cov = np.eye(self.dim) * (self.sigma ** 2)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
    def _compute_eigendecomposition(self):
        """Compute eigendecomposition of covariance matrix."""
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-20)
        self._eigvecs = eigvecs
        self._sqrt_eigvals = np.sqrt(eigvals)
        self._inv_sqrt_eigvals = 1.0 / self._sqrt_eigvals
        
    def _sample_trials_batch(self, num_trials):
        """Sample trial individuals using covariance matrix (batched)."""
        self._compute_eigendecomposition()
        z = np.random.randn(num_trials, self.dim)
        z = z @ np.diag(self._sqrt_eigvals) @ self._eigvecs.T
        trials = self.mean + self.sigma * z
        return trials
    
    def _mutate_batch(self, population, fitness):
        """Generate differential mutation vectors (batched)."""
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        for i in range(np_pop):
            indices = np.random.choice(np_pop, 4, replace=False)
            r1, r2, r3, r4 = indices[:4]
            f = np.random.uniform(0.5, 1.0)
            
            diff1 = population[r2] - population[r3]
            diff2 = population[r4] - population[i]
            trials[i] = population[r1] + f * diff1 + 0.5 * f * diff2
            
        return trials
    
    def _crossover_batch(self, parents, mutants):
        """Binomial crossover between parents and mutants (batched)."""
        cr = np.random.uniform(0.7, 0.95, size=(len(parents), 1))
        mask = np.random.rand(len(parents), self.dim) < cr
        mask[:, 0] = True
        offspring = np.where(mask, mutants, parents)
        return offspring
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select best individuals between parents and offspring (batched)."""
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)[:self.np]
        new_population = combined_pop[sorted_indices]
        new_fitness = combined_fit[sorted_indices]
        
        successful = trial_fitness < fitness
        self.successful_steps.append(np.sum(successful))
        
        return new_population, new_fitness, successful
    
    def _update_mean(self, population, fitness, successful):
        """Update mean vector using weighted recombination."""
        sorted_indices = np.argsort(fitness)[:self.mu]
        selected = population[sorted_indices]
        
        pos_weights = np.maximum(self.weights, 0)
        neg_weights = np.abs(np.minimum(self.weights, 0))
        
        pos_sum = np.sum(pos_weights)
        neg_sum = np.sum(neg_weights)
        
        if pos_sum > 0:
            pos_mean = np.sum(selected * pos_weights.reshape(-1, 1), axis=0) / pos_sum
        else:
            pos_mean = self.mean
            
        if neg_sum > 0:
            neg_mean = np.sum(selected * neg_weights.reshape(-1, 1), axis=0) / neg_sum
        else:
            neg_mean = self.mean
        
        self.prev_mean = self.mean.copy()
        self.mean = pos_mean - neg_mean if neg_sum > 0 else pos_mean
    
    def _adapt_step_size(self, successful):
        """Cumulative step size adaptation (CSA)."""
        if len(self.successful_steps) < 2:
            return
            
        recent_success_rate = np.mean(self.successful_steps[-20:]) / self.np
        expected_rate = 0.5 / self.np
        
        if recent_success_rate > expected_rate:
            self.ps += np.sqrt(self.cs * (2.0 - self.cs)) * self._inv_sqrt_eigvals @ self._eigvecs.T @ \
                       ((self.mean - self.prev_mean) / self.sigma)
        else:
            self.ps *= (1.0 - self.cs)
        
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / 0.3 - 1.0))
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
    
    def _adapt_covariance(self, successful):
        """Update covariance matrix using rank-1 and rank-μ updates."""
        if len(self.successful_steps) < 2:
            return
            
        delta_mean = (self.mean - self.prev_mean) / self.sigma
        
        # Rank-1 update
        self.pc = (1.0 - self.c1) * self.pc + np.sqrt(self.c1 * (2.0 - self.c1) * self.mu_eff) * delta_mean
        
        # Rank-μ update
        sorted_indices = np.argsort(self.fitness)[:self.mu]
        selected = self.population[sorted_indices]
        
        y = (selected - self.prev_mean) / self.sigma
        self.cov_temp = np.zeros((self.dim, self.dim))
        for i in range(min(self.mu, len(selected))):
            w = max(self.weights[i], 0)
            if w > 0:
                self.cov_temp += w * np.outer(y[i], y[i])
        
        # Apply updates
        self.cov = (1.0 - self.c1 - self.cmu) * self.cov + \
                   self.c1 * np.outer(self.pc, self.pc) + \
                   self.cmu * self.cov_temp
        
        # Ensure symmetry and positive definiteness
        self.cov = 0.5 * (self.cov + self.cov.T)
        
        eigvals = np.linalg.eigvalsh(self.cov)
        cond = np.max(eigvals) / np.min(eigvals) if np.min(eigvals) > 0 else 1e20
        
        if cond > 1e7 or np.min(eigvals) < 1e-10:
            self.cov = np.eye(self.dim) * (self.sigma ** 2)
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        np.clip(population, self.lower, self.upper, out=population)
        
    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        if len(population) < 2:
            return 0.0
        dist_matrix = np.linalg.norm(
            population[:, np.newaxis, :] - population[np.newaxis, :, :], axis=2
        )
        return np.mean(dist_matrix[np.triu_indices(len(population), k=1)])
    
    def _check_stagnation(self):
        """Check if optimization has stagnated."""
        if len(self.history_best) < self.stagnation_window:
            return False
        
        recent_best = min(self.history_best[-self.stagnation_window:])
        older_best = min(self.history_best[:self.stagnation_window])
        
        relative_improvement = (older_best - recent_best) / max(abs(older_best), 1e-10)
        
        return relative_improvement < self.min_improvement
    
    def _should_restart(self):
        """Determine if restart is needed."""
        diversity = self._compute_diversity(self.population)
        return self._check_stagnation() or diversity < self.diversity_threshold
    
    def _restart(self):
        """Restart optimization from current best."""
        self.sigma = max(self.sigma, 0.1 * (self.upper - self.lower))
        self.population = self._sample_trials_batch(self.np)
        self._clip_to_bounds(self.population)
        self.fitness = np.full(self.np, np.inf)
        self.mean = self.population.mean(axis=0)
        self.prev_mean = self.mean.copy()
        self._initialize_covariance()
        self.successful_steps = []
        self.history_best = [self.best_f]
    
    def _evaluate_batch(self, batch):
        """Clip and evaluate a batch of candidates."""
        self._clip_to_bounds(batch)
        return batch
    
    def __call__(self, func, stopping_condition):
        """Run the optimization."""
        self._initialize_population()
        self._initialize_covariance()
        self.best_f = np.inf
        self.best_x = self.mean.copy()
        self.generation = 0
        self.history_best = []
        self.successful_steps = []
        
        while not stopping_condition():
            # Sample trials using covariance matrix
            trials = self._sample_trials_batch(self.np)
            
            # Apply differential mutation
            mutants = self._mutate_batch(self.population, self.fitness)
            
            # Crossover
            offspring = self._crossover_batch(self.population, mutants)
            
            # Clip and evaluate
            offspring = self._evaluate_batch(offspring)
            trial_fitness = func(offspring)
            
            if len(trial_fitness) < len(offspring):
                valid_count = len(trial_fitness)
                offspring = offspring[:valid_count]
                trial_fitness = trial_fitness[:valid_count]
                if valid_count == 0:
                    break
                    
            if stopping_condition():
                break
            
            # Selection
            self.population, self.fitness, successful = self._select_survivors_batch(
                self.population, self.fitness, offspring, trial_fitness
            )
            
            # Update mean
            self._update_mean(self.population, self.fitness, successful)
            
            # Adaptation
            self._adapt_step_size(successful)
            self._adapt_covariance(successful)
            
            # Track best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_f:
                self.best_f = self.fitness[best_idx]
                self.best_x = self.population[best_idx].copy()
            
            self.history_best.append(self.best_f)
            self.generation += 1
            
            # Restart if needed
            if self._should_restart():
                self._restart()
        
        return self.best_f, self.best_x
