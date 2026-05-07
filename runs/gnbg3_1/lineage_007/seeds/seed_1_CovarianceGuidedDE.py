import numpy as np


class CovarianceGuidedDE:
    """
    Differential Evolution with covariance-guided mutation and adaptive archive.
    
    Key innovations:
    - Uses weighted covariance of top performers to guide mutation directions
    - Maintains archive of successful mutants for directional learning
    - Adaptive population topology via success-history based strategy selection
    - Per-individual adaptive F and Cr (inspired by JADE)
    """
    
    def __init__(self, dim, np_ratio=5, f_init=0.5, cr_init=0.5, p_archive=0.3):
        self.dim = dim
        self.np = np_ratio * dim  # Population size
        self.f_init = f_init
        self.cr_init = cr_init
        self.p_archive = p_archive
        
        # State variables
        self.population = None
        self.fitness = None
        self.f_values = None  # Per-individual F
        self.cr_values = None  # Per-individual Cr
        self.archive = None
        self.best_idx = None
        self.f_best = None
        self.x_best = None
        self.generation = 0
        self.n_func_evals = 0
        self.success_history_f = []
        self.success_history_cr = []
        self.stagnation_count = 0
        self.last_improvement_gen = 0
        
    def _initialize_population(self, func, stopping_condition):
        """Initialize population uniformly in [-100, 100]^dim."""
        self.population = np.random.uniform(-100, 100, size=(self.np, self.dim))
        self.fitness = func(self.population)
        self._check_budget_exhaustion(len(self.fitness))
        self.n_func_evals += len(self.fitness)
        
        # Initialize per-individual parameters
        self.f_values = np.clip(np.random.normal(self.f_init, 0.1, self.np), 0.1, 1.0)
        self.cr_values = np.clip(np.random.normal(self.cr_init, 0.1, self.np), 0.0, 1.0)
        
        # Initialize archive
        self.archive = np.empty((0, self.dim))
        
        # Track best
        self.best_idx = np.argmin(self.fitness)
        self.f_best = self.fitness[self.best_idx]
        self.x_best = self.population[self.best_idx].copy()
        self.last_improvement_gen = 0
        
    def _select_strategy_batch(self):
        """Select mutation strategy based on recent success (returns strategy indices)."""
        n_strategies = 5
        if len(self.success_history_f) < 3:
            return np.random.randint(0, n_strategies, self.np)
        
        # Success-history based selection (JADE-style)
        weights = np.array([len(self.success_history_f), 
                           len(self.success_history_cr) * 0.5,
                           max(1, self.generation - self.last_improvement_gen) * 0.1])
        weights = np.abs(weights) / np.sum(np.abs(weights))
        
        # Higher diversity in early generations
        if self.generation < 5:
            return np.random.randint(0, n_strategies, self.np)
        
        # Probabilistic selection based on historical performance
        strategies = np.zeros(self.np, dtype=int)
        for i in range(self.np):
            if np.random.random() < 0.7:  # Exploitation
                strategies[i] = np.random.choice(n_strategies, p=weights[:n_strategies] / weights[:n_strategies].sum())
            else:  # Exploration
                strategies[i] = np.random.randint(0, n_strategies)
        return strategies
    
    def _compute_weighted_statistics(self):
        """Compute weighted mean and covariance from top performers."""
        n_top = max(3, int(self.np * self.p_archive))
        sorted_idx = np.argsort(self.fitness)
        top_indices = sorted_idx[:n_top]
        
        # Weighted mean (inverse fitness weighting)
        weights = 1.0 / (self.fitness[top_indices] - self.fitness[top_indices].min() + 1e-8)
        weights /= weights.sum()
        
        top_pop = self.population[top_indices]
        weighted_mean = np.sum(top_pop * weights[:, np.newaxis], axis=0)
        
        # Weighted covariance
        centered = top_pop - weighted_mean
        weighted_cov = np.zeros((self.dim, self.dim))
        for i, w in enumerate(weights):
            weighted_cov += w * np.outer(centered[i], centered[i])
        
        # Regularize covariance
        weighted_cov += np.eye(self.dim) * 1e-6
        
        return weighted_mean, weighted_cov, top_indices
    
    def _mutate_batch(self, strategies):
        """Generate mutant vectors using multiple strategies."""
        mutants = np.empty((self.np, self.dim))
        weighted_mean, weighted_cov, top_idx = self._compute_weighted_statistics()
        
        # Eigendecomposition for covariance-guided mutation
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        except np.linalg.LinAlgError:
            sqrt_cov = np.eye(self.dim)
        
        for i in range(self.np):
            strategy = strategies[i]
            f = self.f_values[i]
            
            if strategy == 0:  # Covariance-guided current-to-pbest
                pbest_idx = np.random.choice(top_idx)
                candidates = np.setdiff1d(np.arange(self.np), [i, pbest_idx])
                if len(candidates) >= 2:
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                else:
                    r1, r2 = np.random.choice(self.np, 2, replace=True)
                    while r1 == i or r2 == i:
                        r1, r2 = np.random.choice(self.np, 2, replace=True)
                
                cov_offset = sqrt_cov @ np.random.randn(self.dim) * f * 0.5
                mutants[i] = (self.population[i] + f * (self.population[pbest_idx] - self.population[i]) +
                             f * (self.population[r1] - self.population[r2]) + cov_offset)
                
            elif strategy == 1:  # Archive-augmented
                pbest_idx = np.random.choice(top_idx)
                candidates = np.setdiff1d(np.arange(self.np), [i, pbest_idx])
                if len(candidates) >= 2:
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                else:
                    r1 = np.random.choice(self.np)
                    while r1 == i:
                        r1 = np.random.choice(self.np)
                
                # Include archive in donor selection
                if len(self.archive) > 0 and np.random.random() < 0.3:
                    archive_donor = self.archive[np.random.randint(len(self.archive))]
                    mutants[i] = (self.population[i] + f * (self.population[pbest_idx] - archive_donor) +
                                 f * (self.population[r1] - self.population[i]))
                else:
                    mutants[i] = (self.population[i] + f * (self.population[pbest_idx] - self.population[i]) +
                                 f * (self.population[r1] - self.population[i]))
                
            elif strategy == 2:  # Mean-guided with perturbation
                candidates = np.setdiff1d(np.arange(self.np), [i])
                if len(candidates) >= 2:
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                else:
                    r1, r2 = np.random.choice(self.np, 2, replace=True)
                    while r1 == i or r2 == i:
                        r1, r2 = np.random.choice(self.np, 2, replace=True)
                
                mean_offset = sqrt_cov @ np.random.randn(self.dim) * f * 0.3
                mutants[i] = (weighted_mean + f * (self.population[r1] - self.population[r2]) + mean_offset)
                
            elif strategy == 3:  # Ring topology with covariance
                n_neighbors = 3
                indices = np.random.choice(self.np, n_neighbors, replace=False)
                while i in indices:
                    indices = np.random.choice(self.np, n_neighbors, replace=False)
                
                ring_vector = np.mean(self.population[indices], axis=0)
                cov_offset = sqrt_cov @ np.random.randn(self.dim) * f * 0.4
                mutants[i] = self.population[i] + f * (ring_vector - self.population[i]) + cov_offset
                
            else:  # strategy == 4: Random target with covariance scaling
                candidates = np.setdiff1d(np.arange(self.np), [i])
                r1 = np.random.choice(candidates)
                cov_offset = sqrt_cov @ np.random.randn(self.dim) * f * 0.6
                mutants[i] = self.population[r1] + f * (self.population[i] - self.population[r1]) + cov_offset
        
        return mutants
    
    def _crossover_batch(self, mutants):
        """Perform binomial crossover with per-individual Cr."""
        trials = np.empty((self.np, self.dim))
        
        for i in range(self.np):
            cr = self.cr_values[i]
            j_rand = np.random.randint(self.dim)
            
            # Binomial crossover
            mask = np.random.random(self.dim) < cr
            mask[j_rand] = True  # Ensure at least one dimension comes from mutant
            
            trials[i] = np.where(mask, mutants[i], self.population[i])
        
        return trials
    
    def _clip_to_bounds(self, population):
        """Clip population to search bounds [-100, 100]."""
        return np.clip(population, -100, 100)
    
    def _select_survivors_batch(self, trials, trial_fitness):
        """Select survivors using greedy (1+1) selection per individual."""
        improved = trial_fitness < self.fitness
        
        # Update population
        self.population[improved] = trials[improved]
        self.fitness[improved] = trial_fitness[improved]
        
        # Add improved individuals to archive
        if np.any(improved):
            archive_new = self.population[improved]
            # Limit archive size to 2*NP
            if len(self.archive) + len(archive_new) > 2 * self.np:
                keep_idx = np.random.choice(len(self.archive) + len(archive_new), 
                                           2 * self.np, replace=False)
                combined = np.vstack([self.archive, archive_new])
                self.archive = combined[keep_idx]
            else:
                self.archive = np.vstack([self.archive, archive_new])
        
        # Track best solution
        current_best_idx = np.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.f_best:
            self.best_idx = current_best_idx
            self.f_best = self.fitness[self.best_idx]
            self.x_best = self.population[self.best_idx].copy()
            self.last_improvement_gen = self.generation
            self.stagnation_count = 0
            return True  # Improvement occurred
        else:
            self.stagnation_count += 1
            return False
    
    def _adapt_step_size(self, improved_mask, trial_fitness):
        """Adapt F values based on success history (JADE-style)."""
        if not np.any(improved_mask):
            # No improvements - increase exploration
            self.f_values = np.clip(self.f_values * 1.1, 0.1, 1.0)
            return
        
        # Record successful F values (weighted by improvement)
        successful_f = self.f_values[improved_mask]
        successful_cr = self.cr_values[improved_mask]
        
        self.success_history_f.extend(successful_f.tolist())
        self.success_history_cr.extend(successful_cr.tolist())
        
        # Keep history bounded
        max_history = 50
        if len(self.success_history_f) > max_history:
            self.success_history_f = self.success_history_f[-max_history:]
            self.success_history_cr = self.success_history_cr[-max_history:]
        
        # Update F: mean of successful values (with c parameter)
        c_f = 0.1
        if len(successful_f) > 0:
            f_new = np.mean(successful_f)
            f_mean = (1 - c_f) * np.mean(self.success_history_f) + c_f * f_new
            self.f_values = np.clip(np.random.normal(f_mean, 0.1, self.np), 0.1, 1.0)
        
        # Update Cr: mean of successful values
        c_cr = 0.1
        if len(successful_cr) > 0:
            cr_new = np.mean(successful_cr)
            cr_mean = (1 - c_cr) * np.mean(self.success_history_cr) + c_cr * cr_new
            self.cr_values = np.clip(np.random.normal(cr_mean, 0.1, self.np), 0.0, 1.0)
    
    def _compute_diversity(self):
        """Compute population diversity (average pairwise distance)."""
        if self.np < 2:
            return 1.0
        
        # Sample-based diversity estimation
        n_samples = min(50, self.np * (self.np - 1) // 2)
        total_dist = 0.0
        
        for _ in range(n_samples):
            i, j = np.random.choice(self.np, 2, replace=False)
            total_dist += np.linalg.norm(self.population[i] - self.population[j])
        
        avg_dist = total_dist / n_samples
        max_possible_dist = np.sqrt(self.dim) * 200  # Max distance in search space
        return avg_dist / max_possible_dist
    
    def _restart_if_stagnant(self, func, stopping_condition):
        """Restart population if stagnant or diversity is low."""
        diversity = self._compute_diversity()
        stagnation_threshold = max(10, self.np // 5)
        
        should_restart = (self.stagnation_count > stagnation_threshold or 
                         diversity < 0.01 or
                         self.generation - self.last_improvement_gen > stagnation_threshold * 2)
        
        if should_restart and self.generation > 5:
            # Keep best solution
            best = self.x_best.copy()
            best_f = self.f_best
            
            # Reinitialize with focus around best
            self.population = np.random.uniform(-100, 100, size=(self.np, self.dim))
            
            # Inject best solution and create variants
            self.population[0] = best
            for i in range(1, min(10, self.np)):
                self.population[i] = best + np.random.randn(self.dim) * 10
                self.population[i] = np.clip(self.population[i], -100, 100)
            
            # Evaluate new population
            self.fitness = func(self.population)
            if self._check_budget_exhaustion(len(self.fitness)):
                return True
            self.n_func_evals += len(self.fitness)
            
            # Reinitialize parameters
            self.f_values = np.clip(np.random.normal(0.5, 0.1, self.np), 0.1, 1.0)
            self.cr_values = np.clip(np.random.normal(0.5, 0.1, self.np), 0.0, 1.0)
            self.archive = np.empty((0, self.dim))
            self.stagnation_count = 0
            
            # Restore best if not in new population
            if self.fitness[0] > best_f:
                self.fitness[0] = best_f
                self.population[0] = best
                self.best_idx = 0
                self.f_best = best_f
                self.x_best = best.copy()
        
        return False
    
    def _check_budget_exhaustion(self, n_evaluations):
        """Check if function evaluations returned fewer than expected (budget exhausted)."""
        return False  # Caller handles this
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness values
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._initialize_population(func, stopping_condition)
        
        while not stopping_condition():
            # Restart check
            if self._restart_if_stagnant(func, stopping_condition):
                break
            
            # Generate trials
            strategies = self._select_strategy_batch()
            mutants = self._mutate_batch(strategies)
            trials = self._crossover_batch(mutants)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate
            trial_fitness = func(trials)
            
            # Check budget exhaustion
            if len(trial_fitness) < len(trials):
                # Truncate and break
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trial_fitness) == 0:
                    break
            
            self.n_func_evals += len(trial_fitness)
            
            if stopping_condition():
                break
            
            # Selection
            improved_mask = trial_fitness < self.fitness
            improved = self._select_survivors_batch(trials, trial_fitness)
            
            # Adaptation
            self._adapt_step_size(improved_mask, trial_fitness)
            
            self.generation += 1
        
        return self.f_best, self.x_best
