import numpy as np


class CovarianceGuidedSwarmOptimizer:
    """
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with particle swarm dynamics. Uses a multivariate normal distribution
    to guide search while maintaining PSO-like momentum.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        self.NP = min(8 * dim, 300)  # Population size
        
        # Evolution path and covariance adaptation parameters
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.cov = np.eye(dim)
        self.c1 = 2.0 / (dim ** 2)
        self.cmu = 2.0 / (dim * (dim + 5))
        self.cc = 2.0 / (dim + 7)
        self.cs = 1.0
        self.damps = 1.0
        
        # PSO parameters
        self.inertia = 0.729
        self.cognitive = 1.49445
        self.social = 1.49445
        
        # Adaptive parameters
        self.step_size = 0.5 * (self.upper - self.lower)
        self.target_success_rate = 0.2
        self.success_history = []
        self.penalty_factor = 1.0
        
        # Tracking
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = np.inf
        
    def __call__(self, func, stopping_condition):
        self._initialize_adaptation_params()
        population = self._initialize_population()
        self.personal_best = population.copy()
        fitness = self._evaluate_batch(func, population)
        self.personal_best_fitness = fitness.copy()
        self._update_global_best(population, fitness)
        gen = 0
        
        while not stopping_condition():
            gen += 1
            trials = self._generate_trials_batch(population)
            trial_fitness = self._evaluate_batch(func, trials)
            
            if len(trial_fitness) < len(trials):
                actual_count = len(trial_fitness)
                trials = trials[:actual_count]
                trial_fitness = trial_fitness[:actual_count]
                if stopping_condition():
                    break
            
            improvements = trial_fitness < fitness
            self._update_adaptation_success_rate(improvements)
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            self._update_personal_best(population, fitness)
            self._update_global_best(population, fitness)
            
            if self._compute_diversity(population) < 1e-6:
                population = self._restart_if_stagnant(population, fitness)
                fitness = self._evaluate_batch(func, population)
            
            self._adapt_covariance_batch(population, improvements)
            self._adapt_step_size(improvements)
            self._adapt_inertia_weight(gen, improvements)
            
            if gen % 50 == 0:
                self._repair_covariance()
        
        return self.global_best_fitness, self.global_best.copy()
    
    def _initialize_adaptation_params(self):
        """Initialize parameters that adapt during evolution."""
        self.success_history = []
        self.penalty_factor = 1.0
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        self.global_best_fitness = np.inf
    
    def _initialize_population(self):
        """Initialize population using covariance-guided sampling."""
        mean = np.random.uniform(self.lower, self.upper, self.dim)
        population = mean + np.random.randn(self.NP, self.dim) @ (
            np.linalg.cholesky(self.cov * self.step_size ** 2)
        ).T
        return self._clip_to_bounds(population)
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to the search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        return func(clipped)
    
    def _generate_trials_batch(self, population):
        """Generate trial population using multiple strategies."""
        n_trials = self.NP
        trials = np.empty((n_trials, self.dim))
        
        n_cov = n_trials // 3
        n_pso = n_trials - 2 * n_cov
        
        if n_cov > 0:
            trials[:n_cov] = self._sample_from_covariance(n_cov)
        
        if n_pso > 0:
            trials[n_cov:n_cov + n_pso] = self._pso_update_batch(
                population, n_pso
            )
        
        if n_cov > 0 and n_trials > n_cov + n_pso:
            trials[n_cov + n_pso:] = self._best1bin_mutation_batch(
                population, n_trials - n_cov - n_pso
            )
        
        return self._clip_to_bounds(trials)
    
    def _sample_from_covariance(self, n_samples):
        """Sample from current covariance distribution."""
        try:
            L = np.linalg.cholesky(self.cov * self.step_size ** 2)
        except np.linalg.LinAlgError:
            L = np.eye(self.dim) * self.step_size
        samples = self.global_best + np.random.randn(n_samples, self.dim) @ L.T
        return samples
    
    def _pso_update_batch(self, population, n_samples):
        """PSO-style position update with adaptive inertia."""
        velocities = self._compute_velocity_batch(population)
        new_positions = population[:n_samples] + velocities[:n_samples]
        return new_positions
    
    def _compute_velocity_batch(self, population):
        """Compute velocity for PSO component."""
        n = len(population)
        r1 = np.random.rand(n, self.dim)
        r2 = np.random.rand(n, self.dim)
        
        cognitive_component = (
            self.cognitive * r1 *
            (self.personal_best[:n] - population)
        )
        
        social_component = (
            self.social * r2 *
            (self.global_best - population)
        )
        
        velocities = (
            self.inertia * np.random.rand(n, 1) * np.random.randn(n, self.dim) * 0.1 +
            cognitive_component +
            social_component
        )
        return velocities
    
    def _best1bin_mutation_batch(self, population, n_samples):
        """DE/best/1/bin style mutation for diversity."""
        indices = np.random.permutation(len(population))[:3 * n_samples]
        mutants = population[indices].reshape(n_samples, 3, self.dim)
        r0, r1, r2 = mutants[:, 0], mutants[:, 1], mutants[:, 2]
        mutation = r0 + 0.5 * (r1 - r2)
        return mutation
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors based on fitness comparison."""
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        
        # Keep best NP individuals
        sorted_indices = np.argsort(combined_fit)[:self.NP]
        return combined_pop[sorted_indices], combined_fit[sorted_indices]
    
    def _update_personal_best(self, population, fitness):
        """Update personal best positions."""
        improved = fitness < self.personal_best_fitness
        self.personal_best[improved] = population[improved]
        self.personal_best_fitness[improved] = fitness[improved]
    
    def _update_global_best(self, population, fitness):
        """Update global best position."""
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.global_best_fitness:
            self.global_best = population[best_idx].copy()
            self.global_best_fitness = fitness[best_idx]
    
    def _update_adaptation_success_rate(self, improvements):
        """Track success rate for step size adaptation."""
        success_rate = np.mean(improvements)
        self.success_history.append(success_rate)
        if len(self.success_history) > 20:
            self.success_history.pop(0)
    
    def _adapt_covariance_batch(self, population, improvements):
        """CMA-ES style covariance adaptation."""
        mean = np.mean(population, axis=0)
        delta_mean = (mean - self.global_best) / self.step_size
        
        # Update evolution paths
        self.pc = (1 - self.cc) * self.pc + np.sqrt(
            self.cc * (2 - self.cc)
        ) * delta_mean
        
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs)
        ) * delta_mean
        
        # Rank-1 update
        rank1 = np.outer(self.pc, self.pc)
        
        # Rank-mu update based on successful mutations
        if np.sum(improvements) > 0:
            successful = population[improvements]
            diffs = (successful - self.global_best) / self.step_size
            rank_mu = np.mean([np.outer(d, d) for d in diffs], axis=0)
        else:
            rank_mu = np.zeros((self.dim, self.dim))
        
        # Apply covariance update
        self.cov = (
            (1 - self.c1 - self.cmu * np.mean(improvements)) * self.cov +
            self.c1 * rank1 +
            self.cmu * np.mean(improvements) * rank_mu
        )
    
    def _adapt_step_size(self, improvements):
        """Adapt step size based on success rate."""
        if len(self.success_history) >= 5:
            avg_success = np.mean(self.success_history)
            if avg_success > self.target_success_rate:
                self.step_size *= 1.0 + 0.2 * (avg_success - self.target_success_rate)
            else:
                self.step_size *= 1.0 - 0.2 * (self.target_success_rate - avg_success)
            
            self.step_size = np.clip(self.step_size, 0.01, 200.0)
    
    def _adapt_inertia_weight(self, generation, improvements):
        """Adapt inertia weight based on convergence."""
        if len(self.success_history) >= 10:
            recent_trend = np.mean(self.success_history[-5:]) - np.mean(
                self.success_history[-10:-5]
            )
            
            if recent_trend < -0.05:
                self.inertia = min(0.9, self.inertia * 1.05)
            elif recent_trend > 0.05:
                self.inertia = max(0.3, self.inertia * 0.95)
    
    def _compute_diversity(self, population):
        """Compute population diversity (average pairwise distance)."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_if_stagnant(self, population, fitness):
        """Restart population if diversity is too low."""
        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]
        
        new_pop = np.empty((self.NP, self.dim))
        new_pop[0] = best_pos
        
        for i in range(1, self.NP):
            new_pop[i] = best_pos + np.random.randn(self.dim) * self.step_size * 0.5
        
        self.personal_best = new_pop.copy()
        self.personal_best_fitness = np.full(self.NP, np.inf)
        
        # Reset evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        
        return self._clip_to_bounds(new_pop)
    
    def _repair_covariance(self):
        """Ensure covariance matrix remains positive definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
