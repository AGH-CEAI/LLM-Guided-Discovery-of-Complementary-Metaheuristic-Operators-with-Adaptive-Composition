import numpy as np


class CovarianceGuidedAdaptiveDE:
    """
    Covariance-Guided Adaptive Differential Evolution.
    Combines CMA-ES-inspired covariance adaptation with multi-strategy DE.
    
    Key features:
    - Simplified covariance matrix adaptation guiding mutation directions
    - Multiple mutation strategies with adaptive selection weights
    - Dynamic step-size adaptation based on success history
    - Inertia-weighted position updates with temporal drift
    - Stagnation detection with covariance-based restart
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(4 * dim, 60), 300)
        self.bounds = (-100.0, 100.0)
        
        self.population = None
        self.fitness = None
        self.f_opt = None
        self.x_opt = None
        self.best_history = []
        
        # Covariance adaptation state
        self.cov_matrix = None
        self.cov_eigen_basis = None
        self.cov_step_size = None
        self.evolution_paths = None
        
        # Strategy adaptation state
        self.strategy_weights = None
        self.strategy_successes = None
        self.strategy_attempts = None
        
        # Dynamic parameters
        self.F_base = 0.5
        self.CR_base = 0.9
        self.inertia_weight = 0.7
        self.drift_rate = 0.01
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.improvement_history = []
        self.last_best_fitness = float('inf')
        
    def _initialize_population(self, func):
        lower, upper = self.bounds
        self.population = np.random.uniform(lower, upper, (self.NP, self.dim))
        self.population = np.clip(self.population, lower, upper)
        
        self.fitness = func(self.population)
        if len(self.fitness) < self.NP:
            self.fitness = np.resize(self.fitness, self.NP)
            self.fitness[np.isnan(self.fitness)] = 1e20
        
        best_idx = np.nanargmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        self.last_best_fitness = self.f_opt
        self.best_history = [self.f_opt]
        
        self._initialize_covariance()
        self._initialize_strategies()
        
    def _initialize_covariance(self):
        self.cov_matrix = np.eye(self.dim)
        self.cov_step_size = (self.bounds[1] - self.bounds[0]) * 0.05
        self.evolution_paths = np.zeros((2, self.dim))
        
    def _initialize_strategies(self):
        # Three mutation strategies: current-to-best, rand, covariance-guided
        self.strategy_weights = np.array([0.4, 0.3, 0.3])
        self.strategy_successes = np.zeros(3)
        self.strategy_attempts = np.zeros(3)
        
    def _update_covariance_adaptation(self, successful_steps, unsuccessful_steps):
        if len(successful_steps) < 2:
            return
            
        # Compute mean successful step
        mean_success = np.mean(successful_steps, axis=0)
        mean_success_normalized = mean_success / (np.linalg.norm(mean_success) + 1e-10)
        
        # Update evolution path for step size adaptation
        c_p = 1.0 / (self.dim ** 0.5)
        self.evolution_paths[0] = (1 - c_p) * self.evolution_paths[0] + np.sqrt(c_p * (2 - c_p)) * mean_success_normalized
        
        # Adapt step size based on path length
        path_length = np.linalg.norm(self.evolution_paths[0])
        expected_length = np.sqrt(self.dim) * (1 - 0.25 / self.dim)
        
        if path_length > 0:
            self.cov_step_size *= np.exp(0.1 * (path_length / expected_length - 1))
            self.cov_step_size = np.clip(self.cov_step_size, 0.01, 10.0)
        
        # Update covariance matrix using successful steps
        alpha_cov = 0.02
        if len(successful_steps) >= 2:
            centered = successful_steps - mean_success
            cov_estimate = np.outer(mean_success_normalized, mean_success_normalized)
            self.cov_matrix = (1 - alpha_cov) * self.cov_matrix + alpha_cov * cov_estimate
            
            # Ensure positive definiteness
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Update eigen basis for sampling
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            self.cov_eigen_basis = eigenvectors[:, idx]
        except np.linalg.LinAlgError:
            self.cov_eigen_basis = np.eye(self.dim)
            
    def _adapt_strategy_weights(self):
        total_attempts = np.sum(self.strategy_attempts) + 1e-10
        for i in range(3):
            if self.strategy_attempts[i] > 0:
                success_rate = self.strategy_successes[i] / self.strategy_attempts[i]
                # Reinforce successful strategies
                adjustment = 0.1 * (success_rate - 0.2)
                self.strategy_weights[i] = np.clip(
                    self.strategy_weights[i] + adjustment, 0.1, 0.6
                )
        
        # Normalize weights
        self.strategy_weights /= np.sum(self.strategy_weights)
        
    def _adapt_inertia_weight(self, current_fitness):
        improvement = self.last_best_fitness - current_fitness
        self.improvement_history.append(improvement)
        
        if len(self.improvement_history) > 10:
            self.improvement_history.pop(0)
        
        recent_improvement = np.mean(self.improvement_history[-5:])
        
        if recent_improvement > 1e-6:
            self.inertia_weight = np.clip(self.inertia_weight * 0.98, 0.3, 0.9)
        else:
            self.inertia_weight = np.clip(self.inertia_weight * 1.02, 0.3, 0.9)
            
    def _select_mutation_strategy_batch(self):
        strategies = np.random.choice(3, size=self.NP, p=self.strategy_weights)
        return strategies
        
    def _mutate_current_to_best_batch(self, indices, strategies):
        best_idx = np.argmin(self.fitness)
        trials = np.zeros((self.NP, self.dim))
        
        # Get indices for all mutations
        r1 = np.random.randint(0, self.NP, size=(self.NP,))
        r2 = np.random.randint(0, self.NP, size=(self.NP,))
        
        # Ensure r1, r2 are different and different from target
        mask = (r1 == r2) | (r1 == np.arange(self.NP)) | (r2 == np.arange(self.NP))
        while np.any(mask):
            r1[mask] = np.random.randint(0, self.NP, size=np.sum(mask))
            r2[mask] = np.random.randint(0, self.NP, size=np.sum(mask))
            mask = (r1 == r2) | (r1 == np.arange(self.NP)) | (r2 == np.arange(self.NP))
        
        # Current-to-best/1 mutation
        f_adaptive = self.F_base * (1 + 0.2 * np.random.randn(self.NP))
        f_adaptive = np.clip(f_adaptive, 0.3, 1.0)
        
        trials = self.population + f_adaptive[:, np.newaxis] * (self.population[best_idx] - self.population) + \
                 f_adaptive[:, np.newaxis] * (self.population[r1] - self.population[r2])
        
        return trials, indices
        
    def _mutate_rand_batch(self, indices, strategies):
        trials = np.zeros((self.NP, self.dim))
        
        r1 = np.random.randint(0, self.NP, size=(self.NP,))
        r2 = np.random.randint(0, self.NP, size=(self.NP,))
        r3 = np.random.randint(0, self.NP, size=(self.NP,))
        
        mask = (r1 == r2) | (r2 == r3) | (r1 == r3) | (r1 == np.arange(self.NP))
        while np.any(mask):
            r1[mask] = np.random.randint(0, self.NP, size=np.sum(mask))
            r2[mask] = np.random.randint(0, self.NP, size=np.sum(mask))
            r3[mask] = np.random.randint(0, self.NP, size=np.sum(mask))
            mask = (r1 == r2) | (r2 == r3) | (r1 == r3) | (r1 == np.arange(self.NP))
        
        f_adaptive = self.F_base * np.random.uniform(0.8, 1.2, size=self.NP)
        f_adaptive = np.clip(f_adaptive, 0.3, 1.0)
        
        trials = self.population[r1] + f_adaptive[:, np.newaxis] * (self.population[r2] - self.population[r3])
        
        return trials, indices
        
    def _mutate_covariance_guided_batch(self, indices, strategies):
        trials = np.zeros((self.NP, self.dim))
        
        # Sample from covariance distribution
        noise = np.random.randn(self.NP, self.dim)
        if self.cov_eigen_basis is not None:
            noise = noise @ self.cov_eigen_basis.T
        
        trials = self.population + self.cov_step_size * noise
        
        return trials, indices
        
    def _crossover_batch(self, population, trials, strategies):
        cr_adaptive = self.CR_base * np.random.uniform(0.8, 1.2, size=self.NP)
        cr_adaptive = np.clip(cr_adaptive, 0.1, 0.95)
        
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        
        cross_mask = np.random.rand(self.NP, self.dim) < cr_adaptive[:, np.newaxis]
        cross_mask[np.arange(self.NP), j_rand] = True
        
        offspring = np.where(cross_mask, trials, population)
        return offspring
        
    def _position_update_fitness_rank(self, population, fitness):
        # Rank-based position influence
        ranks = np.argsort(np.argsort(fitness))
        ranks = ranks / (self.NP - 1 + 1e-10)
        
        # Weighted average toward better individuals
        weights = 1.0 / (1.0 + ranks)
        weights = weights / np.sum(weights)
        
        mean_better = np.sum(population * weights[:, np.newaxis], axis=0)
        return mean_better
        
    def _position_update_temporal_drift(self, population, fitness):
        # Add small drift toward promising regions
        best_idx = np.argmin(fitness)
        best_pos = population[best_idx]
        
        drift_vector = 0.001 * (best_pos - np.mean(population, axis=0))
        return drift_vector
        
    def _select_survivors_batch(self, population, fitness, offspring, offspring_fitness, strategies):
        improved_mask = offspring_fitness < fitness
        
        new_population = np.where(improved_mask[:, np.newaxis], offspring, population)
        new_fitness = np.where(improved_mask, offspring_fitness, fitness)
        
        # Track strategy successes
        for s in range(3):
            mask = (strategies == s) & improved_mask
            self.strategy_successes[s] += np.sum(mask)
            self.strategy_attempts[s] += np.sum(strategies == s)
        
        # Track successful steps for covariance update
        successful_steps = offspring[improved_mask] - population[improved_mask]
        unsuccessful_steps = offspring[~improved_mask] - population[~improved_mask]
        
        return new_population, new_fitness, successful_steps, unsuccessful_steps
        
    def _restart_if_stagnant(self, func):
        current_best = np.min(self.fitness)
        
        if self.last_best_fitness - current_best < 1e-8:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        self.last_best_fitness = current_best
        
        if self.stagnation_counter >= 15:
            # Covariance-based restart: spread population along eigenvectors
            lower, upper = self.bounds
            
            new_population = np.tile(self.x_opt, (self.NP, 1))
            
            for i in range(self.NP):
                noise = np.random.randn(self.dim)
                if self.cov_eigen_basis is not None:
                    noise = noise @ self.cov_eigen_basis.T
                new_population[i] += self.cov_step_size * 3.0 * noise
            
            new_population = np.clip(new_population, lower, upper)
            self.population = new_population
            
            self.fitness = func(self.population)
            if len(self.fitness) < self.NP:
                self.fitness = np.resize(self.fitness, self.NP)
                self.fitness[np.isnan(self.fitness)] = 1e20
            
            self.stagnation_counter = 0
            self.cov_step_size *= 2.0
            self.cov_step_size = np.clip(self.cov_step_size, 0.01, 10.0)
            
            return True
        return False
        
    def _compute_diversity(self):
        mean_pos = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - mean_pos, axis=1)
        return np.mean(distances)
        
    def _update_best_solution(self, fitness):
        best_idx = np.nanargmin(fitness)
        if fitness[best_idx] < self.f_opt:
            self.f_opt = fitness[best_idx]
            self.x_opt = self.population[best_idx].copy()
            self.best_history.append(self.f_opt)
            return True
        return False
        
    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        
        while not stopping_condition():
            strategies = self._select_mutation_strategy_batch()
            
            # Generate trials for each strategy
            trials_s0, _ = self._mutate_current_to_best_batch(np.arange(self.NP), strategies)
            trials_s1, _ = self._mutate_rand_batch(np.arange(self.NP), strategies)
            trials_s2, _ = self._mutate_covariance_guided_batch(np.arange(self.NP), strategies)
            
            # Combine based on strategy selection
            trials = np.copy(self.population)
            for i in range(self.NP):
                if strategies[i] == 0:
                    trials[i] = trials_s0[i]
                elif strategies[i] == 1:
                    trials[i] = trials_s1[i]
                else:
                    trials[i] = trials_s2[i]
            
            # Apply rank-based position influence
            mean_better = self._position_update_fitness_rank(self.population, self.fitness)
            drift = self._position_update_temporal_drift(self.population, self.fitness)
            
            # Blend with mean position
            blend_factor = 0.1 * self.inertia_weight
            trials = (1 - blend_factor) * trials + blend_factor * mean_better + drift
            
            # Crossover
            trials = self._crossover_batch(self.population, trials, strategies)
            
            # Clip to bounds
            lower, upper = self.bounds
            trials = np.clip(trials, lower, upper)
            
            # Evaluate
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            n_received = len(trial_fitness)
            if n_received < self.NP:
                trial_fitness = np.resize(trial_fitness, self.NP)
                trial_fitness[np.isnan(trial_fitness)] = 1e20
                trials = trials[:n_received]
                trial_fitness = trial_fitness[:n_received]
                if stopping_condition():
                    return self.f_opt, self.x_opt
                if n_received == 0:
                    return self.f_opt, self.x_opt
                    
            if stopping_condition():
                return self.f_opt, self.x_opt
            
            # Selection
            self.population, self.fitness, successful_steps, unsuccessful_steps = \
                self._select_survivors_batch(self.population, self.fitness, trials, trial_fitness, strategies)
            
            # Update best
            self._update_best_solution(self.fitness)
            
            # Covariance adaptation
            if len(successful_steps) > 0:
                self._update_covariance_adaptation(successful_steps, unsuccessful_steps)
            
            # Strategy adaptation
            self._adapt_strategy_weights()
            
            # Inertia weight adaptation
            self._adapt_inertia_weight(self.f_opt)
            
            # Stagnation check with restart
            self._restart_if_stagnant(func)
            
        return self.f_opt, self.x_opt
