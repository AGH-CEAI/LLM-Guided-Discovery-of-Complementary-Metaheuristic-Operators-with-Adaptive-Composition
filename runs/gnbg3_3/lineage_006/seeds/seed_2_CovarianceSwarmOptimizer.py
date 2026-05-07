import numpy as np


class CovarianceSwarmOptimizer:
    """CMA-ES inspired covariance adaptation combined with swarm-like personal best tracking.
    
    Key components:
    - Covariance matrix adaptation (CMA-ES style) for anisotropic exploration
    - Personal best memory per particle for swarm-like exploitation
    - Adaptive step size based on success ratio
    - Diversity-triggered restarts for multimodal robustness
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = int(8 * dim)
        self.lower = -100.0
        self.upper = 100.0
        
        self._init_cmaes_params()
        self._init_state()
    
    def _init_cmaes_params(self):
        self.sigma = 5.0
        self.coeff_cs = 0.3
        self.coeff_damps = 1.0 + self.coeff_cs
        self.coeff_cmu = 0.5 * (2.0 / (dim ** 2) + 0.5 + 1.0)
        self.coeff_cc = 4.0 / (dim + 4.0)
        self.coeff_hsig = 0.5 * ((4.0 + self.dim) / (self.dim + 4.0))
        
        self.weights = np.log((self.np + 1.0) / 2.0) - np.log(np.arange(1, self.np + 1))
        self.weights = np.maximum(self.weights, 0.0)
        self.weights /= np.sum(self.weights)
        self.mu_eff = (np.sum(self.weights) ** 2) / np.sum(self.weights ** 2)
        self.mu_cov = self.mu_eff
    
    def _init_state(self):
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        self.cov_path = np.zeros(self.dim)
        self.sigma_path = np.zeros(self.dim)
        self.personal_best_fitness = np.full(self.np, np.inf)
        self.personal_best_position = np.zeros((self.np, self.dim))
        
        self.best_fitness = np.inf
        self.best_position = np.zeros(self.dim)
        self.generation = 0
        self.succ_count = 0
        self.succ_rate = 0.0
        self.stagnation_counter = 0
        self.diversity = 0.0
        self.initial_diversity = 0.0
    
    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        self.initial_diversity = self.diversity
        
        while not stopping_condition():
            self.generation += 1
            
            trial_pop, trial_indices = self._generate_trial_batch()
            trial_fit = func(trial_pop)
            
            if len(trial_fit) < len(trial_pop):
                trial_fit = self._handle_budget_exhaustion(trial_pop, trial_fit)
                if len(trial_fit) == 0:
                    break
            
            self._update_personal_best_batch(trial_pop, trial_fit, trial_indices)
            self._update_population_batch(trial_pop, trial_fit, trial_indices)
            
            self._adapt_step_size()
            self._adapt_covariance_matrix()
            self._compute_diversity()
            
            self._adapt_inertia_weight()
            
            if self._check_stagnation():
                self._restart_from_best(func)
            
            if stopping_condition():
                break
        
        return self.best_fitness, self.best_position
    
    def _initialize_population(self, func):
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        self.fitness = func(self.population)
        
        if len(self.fitness) < self.np:
            self.fitness = self._handle_budget_exhaustion(self.population, self.fitness)
            if len(self.fitness) == 0:
                self.best_fitness = np.inf
                return
        
        self._update_best_from_population()
        self._init_personal_best()
        self._compute_diversity()
    
    def _init_personal_best(self):
        self.personal_best_fitness = self.fitness.copy()
        self.personal_best_position = self.population.copy()
    
    def _generate_trial_batch(self):
        num_trials = self.np
        trial_pop = np.zeros((num_trials, self.dim))
        trial_indices = np.arange(num_trials)
        
        for i in range(num_trials):
            z = np.random.randn(self.dim)
            trial_pop[i] = self.mean + self.sigma * (self.cov @ z)
        
        trial_pop = np.clip(trial_pop, self.lower, self.upper)
        return trial_pop, trial_indices
    
    def _handle_budget_exhaustion(self, trial_pop, trial_fit):
        valid_len = len(trial_fit)
        if valid_len < len(trial_pop):
            trial_pop = trial_pop[:valid_len]
        return trial_fit
    
    def _update_personal_best_batch(self, trial_pop, trial_fit, trial_indices):
        improved_mask = trial_fit < self.personal_best_fitness[trial_indices]
        improved_indices = trial_indices[improved_mask]
        
        if len(improved_indices) > 0:
            self.personal_best_fitness[improved_indices] = trial_fit[improved_mask]
            self.personal_best_position[improved_indices] = trial_pop[improved_mask]
    
    def _update_population_batch(self, trial_pop, trial_fit, trial_indices):
        better_mask = trial_fit < self.fitness[trial_indices]
        better_indices = trial_indices[better_mask]
        
        self.population[better_indices] = trial_pop[better_mask]
        self.fitness[better_indices] = trial_fit[better_mask]
        
        self._update_best_from_population()
    
    def _update_best_from_population(self):
        valid_mask = np.isfinite(self.fitness)
        if not np.any(valid_mask):
            return
        
        min_idx = np.argmin(self.fitness[valid_mask])
        min_fit = self.fitness[valid_mask][min_idx]
        
        if min_fit < self.best_fitness:
            all_indices = np.where(valid_mask)[0]
            self.best_fitness = min_fit
            self.best_position = self.population[all_indices[min_idx]].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def _adapt_step_size(self):
        norm_sigma_path = np.linalg.norm(self.sigma_path)
        expected_norm = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))
        
        cs_delta = self.coeff_cs / (norm_sigma_path + 1e-20)
        self.succ_rate = 0.0
        
        if self.generation > 1:
            self.succ_rate = self.succ_count / self.np
            cs_adjusted = self.coeff_cs * (self.succ_rate - 0.5) / np.sqrt(self.mu_eff)
            self.sigma *= np.exp(np.clip(cs_adjusted, -5.0, 5.0))
        
        self.sigma = np.clip(self.sigma, 1e-10, 1e3)
        self.succ_count = 0
    
    def _adapt_covariance_matrix(self):
        z_norm = np.random.randn(self.dim)
        self.sigma_path = (
            (1.0 - self.coeff_cc) * self.sigma_path +
            np.sqrt(self.coeff_cc * (2.0 - self.coeff_cc) * self.mu_eff) *
            ((self.cov @ self.mean) - self.cov @ self._old_mean) / self.sigma
        )
        
        hsig = 1.0 if np.linalg.norm(self.sigma_path) < (
            1.5 + 1.0 / (self.dim + 0.5)) * expected_norm else 0.0
        
        rank_one = self.coeff_cmu * self.mu_cov * np.outer(self.sigma_path, self.sigma_path)
        
        centered_pop = self.population - self.mean
        weighted_cov = np.sum(
            self.weights[:, np.newaxis, np.newaxis] *
            centered_pop[:, :, np.newaxis] * centered_pop[:, np.newaxis, :],
            axis=0
        )
        rank_mu = self.coeff_cmu * (1.0 / (self.sigma ** 2)) * weighted_cov
        
        self.cov = (
            (1.0 - self.coeff_cmu * (1.0 - self.coeff_cmu) * self.succ_rate) * self.cov +
            rank_one * hsig + rank_mu
        )
        
        self._ensure_covariance_positive_definite()
        self._old_mean = self.mean.copy()
    
    def _ensure_covariance_positive_definite(self):
        self.cov = 0.5 * (self.cov + self.cov.T)
        min_eig = np.min(np.linalg.eigvalsh(self.cov))
        if min_eig < 1e-10:
            self.cov += (1e-9 - min_eig) * np.eye(self.dim)
    
    def _compute_diversity(self):
        centered = self.population - self.mean
        mean_dist = np.mean(np.sqrt(np.sum(centered ** 2, axis=1)))
        self.diversity = mean_dist
    
    def _adapt_inertia_weight(self):
        base_rate = 0.4 + 0.5 * (1.0 - self.succ_rate)
        noise = np.random.uniform(-0.1, 0.1)
        self.inertia = np.clip(base_rate + noise, 0.1, 0.9)
    
    def _check_stagnation(self):
        diversity_threshold = self.initial_diversity * 0.01
        stagnation_threshold = max(50, 5 * self.dim)
        
        return (self.diversity < diversity_threshold) or (self.stagnation_counter > stagnation_threshold)
    
    def _restart_from_best(self, func):
        self.mean = self.best_position.copy() + np.random.uniform(-0.5 * self.sigma, 0.5 * self.sigma, self.dim)
        self.mean = np.clip(self.mean, self.lower, self.upper)
        
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        self.population[0] = self.best_position.copy()
        
        self.cov = np.eye(self.dim)
        self.sigma = max(5.0, self.sigma * 0.5)
        self.sigma_path = np.zeros(self.dim)
        self.stagnation_counter = 0
        
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = self._handle_budget_exhaustion(self.population, self.fitness)
        
        self._init_personal_best()
        self._update_best_from_population()
        self._compute_diversity()
