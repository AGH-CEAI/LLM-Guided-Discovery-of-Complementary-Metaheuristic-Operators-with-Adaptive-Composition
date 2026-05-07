import numpy as np


class CMAAdaptivenParticleSwarmOptimizer:
    """
    Hybrid optimizer combining Particle Swarm Optimization with CMA-ES style
    covariance adaptation. Uses multiple neighborhood topologies and an
    archive mechanism for diversity maintenance.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(200, max(4 * dim, 60))
        self.bounds = np.array([-100.0, 100.0])
        self.range = self.bounds[1] - self.bounds[0]
        
        # PSO parameters
        self.w_inertia = 0.729
        self.c1_personal = 1.494
        self.c2_social = 1.494
        self.chi = 0.729
        
        # CMA adaptation state
        self.cov_matrix = np.eye(dim)
        self.mean = np.zeros(dim)
        self.step_size = 0.3 * self.range
        self.cov_adapt_rate = 0.05
        self.p_cum = np.zeros(dim)
        
        # Archive for diversity
        self.archive_size = self.np // 2
        self.archive = None
        self.archive_fitness = None
        
        # Tracking
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = np.inf
        
        # Stagnation detection
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        self.improvement_history = []
        self.max_improvement_history = 100
        
        # Neighborhood (lbest topology - ring)
        self.neighborhood_size = 3
        self.neighborhood_indices = self._compute_neighborhood_indices()
        
        # Strategy adaptation
        self.strategy_weights = np.ones(3) / 3
        self.strategy_names = ['standard', 'covariance_guided', 'archive_influenced']
        
    def _compute_neighborhood_indices(self):
        """Compute ring neighborhood indices for each particle."""
        np_local = self.np
        n_size = self.neighborhood_size
        indices = np.zeros((np_local, 2 * n_size + 1), dtype=int)
        for i in range(np_local):
            neighbors = [(i + j) % np_local for j in range(-n_size, n_size + 1)]
            indices[i] = neighbors
        return indices
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop with batched operations."""
        self._initialize_population(func)
        self._initialize_archive()
        
        while not stopping_condition():
            self._update_personal_best_batch(func)
            self._update_neighborhood_best_batch()
            self._update_global_best_batch()
            
            self._update_covariance_adaptation()
            self._build_trial_batch(func)
            
            if stopping_condition():
                break
                
            self._select_survivors_batch(func)
            self._update_archive_batch()
            
            self._check_stagnation()
            if self._restart_if_needed(func, stopping_condition):
                continue
                
            self._adapt_parameters()
            
        return self.global_best_fitness, self.global_best.copy()
    
    def _initialize_population(self, func):
        """Initialize particle positions, velocities, and personal bests."""
        self.population = np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            -0.5 * self.range, 0.5 * self.range, size=(self.np, self.dim)
        )
        
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = self._handle_budget_exhaustion(self.fitness, self.population)
            
        self.personal_best = self.population.copy()
        self.personal_best_fitness = self.fitness.copy()
        
        self._update_global_best_from_population()
        self.mean = self.population.mean(axis=0)
        
    def _initialize_archive(self):
        """Initialize empty archive for diversity maintenance."""
        self.archive = []
        self.archive_fitness = np.array([])
        
    def _handle_budget_exhaustion(self, fitness, population):
        """Handle cases where func returns fewer values than expected."""
        actual_len = len(fitness)
        if actual_len < len(population):
            return np.pad(fitness, (0, len(population) - actual_len), 
                         constant_values=np.inf)
        return fitness
    
    def _update_personal_best_batch(self, func):
        """Update personal best positions using vectorized comparison."""
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = self._handle_budget_exhaustion(self.fitness, self.population)
            
        improved_mask = self.fitness < self.personal_best_fitness
        self.personal_best_fitness[improved_mask] = self.fitness[improved_mask]
        self.personal_best[improved_mask] = self.population[improved_mask]
        
    def _update_neighborhood_best_batch(self):
        """Update local best for each particle's neighborhood."""
        self.local_best = np.zeros((self.np, self.dim))
        self.local_best_fitness = np.full(self.np, np.inf)
        
        for i in range(self.np):
            neighbors = self.neighborhood_indices[i]
            neighbor_fitness = self.personal_best_fitness[neighbors]
            best_neighbor_idx = neighbors[np.argmin(neighbor_fitness)]
            self.local_best[i] = self.personal_best[best_neighbor_idx]
            self.local_best_fitness[i] = self.personal_best_fitness[best_neighbor_idx]
            
    def _update_global_best_from_population(self):
        """Update global best from current population."""
        current_best_idx = np.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.global_best_fitness:
            self.global_best = self.population[current_best_idx].copy()
            self.global_best_fitness = self.fitness[current_best_idx]
            
    def _update_global_best_batch(self):
        """Update global best from personal bests."""
        current_best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[current_best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[current_best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[current_best_idx]
            
    def _update_covariance_adaptation(self):
        """Update covariance matrix using successful direction vectors."""
        successful_mask = self.fitness < self.personal_best_fitness
        if successful_mask.sum() > 0:
            successful_directions = self.population[successful_mask] - self.mean
            successful_directions = successful_directions[~np.any(np.isnan(successful_directions), axis=1)]
            
            if len(successful_directions) > 0:
                cov_update = np.cov(successful_directions.T)
                if not np.any(np.isnan(cov_update)) and not np.any(np.isinf(cov_update)):
                    self.cov_matrix = (1 - self.cov_adapt_rate) * self.cov_matrix + \
                                     self.cov_adapt_rate * cov_update
                    
                    eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
                    eigenvalues = np.maximum(eigenvalues, 1e-10)
                    self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            self.p_cum = 0.9 * self.p_cum + 0.1 * successful_directions.mean(axis=0) if len(successful_directions) > 0 else 0.9 * self.p_cum
            
    def _sample_from_covariance(self, n_samples):
        """Sample directions from current covariance distribution."""
        try:
            L = np.linalg.cholesky(self.cov_matrix)
            samples = np.random.randn(n_samples, self.dim) @ L.T
            return samples * self.step_size
        except np.linalg.LinAlgError:
            return np.random.randn(n_samples, self.dim) * self.step_size
            
    def _build_trial_batch(self, func):
        """Build trial population using multiple strategies."""
        n_strategies = 3
        trials_per_strategy = self.np // n_strategies
        
        # Strategy 1: Standard PSO velocity update
        r1 = np.random.rand(self.np, self.dim)
        r2 = np.random.rand(self.np, self.dim)
        cognitive = self.c1_personal * r1 * (self.personal_best - self.population)
        social = self.c2_social * r2 * (self.local_best - self.population)
        standard_velocity = self.chi * (self.w_inertia * self.velocity + cognitive + social)
        standard_trials = self.population + standard_velocity
        
        # Strategy 2: Covariance-guided mutation
        cov_trials = np.zeros((trials_per_strategy, self.dim))
        for i in range(trials_per_strategy):
            idx = (i + self.stagnation_counter) % self.np
            direction = self._sample_from_covariance(1)
            cov_trials[i] = self.population[idx] + direction[0]
            
        # Strategy 3: Archive-influenced mutation
        archive_trials = np.zeros((trials_per_strategy, self.dim))
        if len(self.archive) > 0:
            for i in range(trials_per_strategy):
                parent_idx = i % self.np
                archive_idx = np.random.randint(0, min(len(self.archive), trials_per_strategy))
                diff = self.archive[archive_idx] - self.population[parent_idx]
                archive_trials[i] = self.population[parent_idx] + 0.5 * diff + \
                                   np.random.randn(self.dim) * self.step_size * 0.1
        else:
            archive_trials = standard_trials[:trials_per_strategy]
            
        # Combine strategies
        remaining = self.np - 3 * trials_per_strategy
        self.trials = np.vstack([standard_trials, cov_trials, archive_trials])
        if remaining > 0:
            extra_trials = self.population[:remaining] + np.random.randn(remaining, self.dim) * self.step_size
            self.trials = np.vstack([self.trials, extra_trials])
            
        # Clip to bounds
        self.trials = np.clip(self.trials, self.bounds[0], self.bounds[1])
        
        # Evaluate trials
        self.trial_fitness = func(self.trials)
        if len(self.trial_fitness) < len(self.trials):
            self.trial_fitness = self._handle_budget_exhaustion(self.trial_fitness, self.trials)
            
    def _select_survivors_batch(self, func):
        """Select survivors using greedy selection with diversity bonus."""
        n_replace = min(self.np, len(self.trials))
        
        # Calculate improvement
        improvement = self.fitness[:n_replace] - self.trial_fitness[:n_replace]
        
        # Update strategy weights based on improvements
        strategy_improvements = [
            improvement[:self.np // 3].mean(),
            improvement[self.np // 3:2 * self.np // 3].mean(),
            improvement[2 * self.np // 3:].mean()
        ]
        
        for i, imp in enumerate(strategy_improvements):
            if imp > 0:
                self.strategy_weights[i] += 0.1 * imp
            self.strategy_weights[i] *= 0.95
            
        self.strategy_weights = np.maximum(self.strategy_weights, 0.01)
        self.strategy_weights /= self.strategy_weights.sum()
        
        # Greedy selection with diversity consideration
        for i in range(n_replace):
            if self.trial_fitness[i] < self.fitness[i]:
                self.population[i] = self.trials[i].copy()
                self.fitness[i] = self.trial_fitness[i]
                
                if self.trial_fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.trials[i].copy()
                    self.personal_best_fitness[i] = self.trial_fitness[i]
                    
        # Update mean
        self.mean = self.population.mean(axis=0)
        
        # Update velocities of improved particles
        improved_mask = self.fitness < func(self.population) if len(self.trials) >= self.np else np.zeros(self.np, dtype=bool)
        
    def _update_archive_batch(self):
        """Update diversity archive with non-dominated solutions."""
        if len(self.archive) == 0:
            self.archive = self.population.copy()
            self.archive_fitness = self.fitness.copy()
            return
            
        all_solutions = np.vstack([self.archive, self.population])
        all_fitness = np.concatenate([self.archive_fitness, self.fitness])
        
        # Simple diversity-based archive maintenance
        if len(all_solutions) > self.archive_size:
            # Keep diverse solutions
            distances = np.zeros(len(all_solutions))
            for i in range(len(all_solutions)):
                dists = np.linalg.norm(all_solutions - all_solutions[i], axis=1)
                distances[i] = np.mean(np.partition(dists, min(5, len(dists)))[1:6])
                
            # Keep best fitness solutions and diverse solutions
            combined_score = 0.7 * (-all_fitness / all_fitness.min() if all_fitness.min() > 0 else -all_fitness) + \
                           0.3 * (distances / distances.max())
            keep_indices = np.argsort(combined_score)[-self.archive_size:]
        else:
            keep_indices = np.arange(len(all_solutions))
            
        self.archive = all_solutions[keep_indices]
        self.archive_fitness = all_fitness[keep_indices]
        
    def _check_stagnation(self):
        """Check if optimization has stagnated."""
        current_best = self.global_best_fitness
        self.improvement_history.append(current_best)
        
        if len(self.improvement_history) > self.max_improvement_history:
            self.improvement_history.pop(0)
            
        if len(self.improvement_history) >= 20:
            recent_improvement = self.improvement_history[-20] - current_best
            if recent_improvement < 1e-6 * abs(current_best + 1e-10):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
                
    def _restart_if_needed(self, func, stopping_condition):
        """Perform adaptive restart when stagnation is detected."""
        if self.stagnation_counter >= self.stagnation_threshold:
            if len(self.archive) > 0:
                # Restart from archive diversity
                n_from_archive = self.np // 3
                restart_population = self.archive[:n_from_archive].copy()
                
                remaining = self.np - n_from_archive
                if remaining > 0:
                    restart_population = np.vstack([
                        restart_population,
                        np.random.uniform(self.bounds[0], self.bounds[1], 
                                        size=(remaining, self.dim))
                    ])
            else:
                restart_population = np.random.uniform(
                    self.bounds[0], self.bounds[1], size=(self.np, self.dim)
                )
                
            self.population = restart_population
            self.velocity = np.random.uniform(
                -0.5 * self.range, 0.5 * self.range, size=(self.np, self.dim)
            )
            self.personal_best = self.population.copy()
            self.personal_best_fitness = np.full(self.np, np.inf)
            self.fitness = np.full(self.np, np.inf)
            self.stagnation_counter = 0
            self.step_size *= 0.8
            self.step_size = max(self.step_size, 0.01 * self.range)
            
            return True
        return False
        
    def _adapt_parameters(self):
        """Adapt PSO and adaptation parameters based on progress."""
        # Adapt inertia weight based on stagnation
        if self.stagnation_counter > 5:
            self.w_inertia = min(0.9, self.w_inertia + 0.01)
        else:
            self.w_inertia = max(0.3, self.w_inertia - 0.005)
            
        # Adapt step size based on success
        if len(self.improvement_history) >= 10:
            recent_change = abs(self.improvement_history[-1] - self.improvement_history[-10])
            if recent_change < 1e-8:
                self.step_size *= 1.05
            elif recent_change > 1.0:
                self.step_size *= 0.95
                
        self.step_size = np.clip(self.step_size, 0.01, 0.5 * self.range)
        
    def _compute_diversity(self):
        """Compute population diversity measure."""
        if len(self.population) < 2:
            return 0.0
        centroids = self.population.mean(axis=0)
        distances = np.linalg.norm(self.population - centroids, axis=1)
        return np.mean(distances)
