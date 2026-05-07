import numpy as np


class DiagonalCovarianceMemeticOptimizer:
    """
    A population-based metaheuristic combining CMA-ES-style diagonal covariance adaptation
    with a memetic local search component. Uses (mu/mu_I, lambda) selection with
    cumulative path adaptation for step-size control.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self._dim = dim
        self._pop_size = 5 * dim
        self._lower_bound = np.full(dim, -100.0)
        self._upper_bound = np.full(dim, 100.0)
        
        self._step_size = 0.3 * (self._upper_bound - self._lower_bound)[0] / np.sqrt(dim)
        self._cov_diag = np.ones(dim) * ((self._upper_bound - self._lower_bound)[0] / np.sqrt(dim)) ** 2
        self._evolution_path = np.zeros(dim)
        
        self._p_succ = min(25, self._pop_size)
        self._c_succ = 1.0 / (dim * 2)
        self._c_path = 1.0 / (dim * 4)
        self._c_cov = 0.02
        self._c_mean = 0.3 / dim
        self._memetic_rate = 0.05
        
        self._population = None
        self._fitness = None
        self._mean = None
        self._success_history = []
        self._fitness_history = []
        self._stagnation_counter = 0
        self._best_fitness = np.inf
        self._best_x = None
    
    def _initialize_population(self):
        """Generate initial population within bounds."""
        pop = np.random.uniform(self._lower_bound, self._upper_bound, (self._pop_size, self._dim))
        self._mean = np.mean(pop, axis=0)
        self._success_history = []
        self._fitness_history = []
        self._stagnation_counter = 0
        return pop
    
    def _mutate_batch(self):
        """Generate trial population using diagonal covariance-adapted sampling."""
        std = np.sqrt(self._cov_diag)
        z = np.random.randn(self._pop_size, self._dim)
        noise = z * std
        
        best_contribution = 0.4 * (self._best_x - self._mean)
        mean_shift = 0.3 * (self._mean - np.mean(self._population, axis=0))
        
        trials = self._mean + self._step_size * noise + best_contribution + mean_shift
        return trials
    
    def _apply_memetic_refinement(self, pop, fit):
        """Apply local search to top fraction of population (batched)."""
        n_local = max(1, int(self._pop_size * self._memetic_rate))
        indices = np.argpartition(fit, n_local)[:n_local]
        
        local_pop = pop[indices].copy()
        step = 0.1 * self._step_size * np.random.randn(n_local, self._dim)
        local_trials = np.clip(local_pop + step, self._lower_bound, self._upper_bound)
        
        return indices, local_trials
    
    def _select_batch(self, pop, fit, trials, trial_fit):
        """Greedy selection: accept improving trials."""
        improved = trial_fit < fit
        new_pop = np.where(improved[:, np.newaxis], trials, pop)
        new_fit = np.where(improved, trial_fit, fit)
        return new_pop, new_fit, improved
    
    def _update_success_history(self, improved):
        """Maintain sliding window of success rates."""
        self._success_history.append(np.mean(improved))
        if len(self._success_history) > self._p_succ:
            self._success_history.pop(0)
    
    def _update_step_size_and_path(self, improved, trials):
        """Cumulative path adaptation for step-size control."""
        if len(self._success_history) < self._p_succ // 2:
            return
        
        succ_rate = np.mean(self._success_history)
        p_target = 0.15
        indicator = 1.0 if succ_rate > p_target else 0.0
        
        self._evolution_path = (1 - self._c_path) * self._evolution_path + np.sqrt(
            self._c_path * (2 - self._c_path)
        ) * np.mean(trials[improved] - self._mean, axis=0) / self._step_size
        
        self._step_size *= np.exp(self._c_succ * (indicator - 0.5) + 0.2 * np.sum(self._evolution_path ** 2) / self._dim - 0.2)
        self._step_size = np.clip(self._step_size, 1e-10, 0.5 * (self._upper_bound - self._lower_bound)[0])
    
    def _update_covariance_diag(self, improved, trials):
        """Rank-1 covariance update based on successful mutations."""
        if np.sum(improved) < 1:
            return
        
        mean_delta = np.mean(trials[improved] - self._mean, axis=0)
        rank1_update = mean_delta ** 2 / (np.sum(mean_delta ** 2) + 1e-10)
        
        self._cov_diag = (1 - self._c_cov) * self._cov_diag + self._c_cov * rank1_update * np.sum(mean_delta ** 2)
        self._cov_diag = np.clip(self._cov_diag, 1e-8, 1e6)
    
    def _update_mean(self):
        """Shift mean toward better-performing individuals."""
        weights = np.exp(-self._fitness / (np.mean(self._fitness) + 1e-10))
        weights /= np.sum(weights)
        self._mean = (1 - self._c_mean) * self._mean + self._c_mean * np.sum(
            self._population * weights[:, np.newaxis], axis=0
        )
    
    def _compute_diversity(self, pop):
        """Calculate average pairwise Euclidean distance."""
        if len(pop) < 2:
            return 0.0
        dists = np.linalg.norm(pop[:, np.newaxis] - pop[np.newaxis, :], axis=2)
        return np.mean(dists[np.triu_indices_from(dists, k=1)])
    
    def _check_restart_criteria(self):
        """Detect stagnation via fitness plateau or diversity collapse."""
        if len(self._fitness_history) < 50:
            return False
        
        recent_window = self._fitness_history[-25:]
        improvement = self._fitness_history[-50] - min(recent_window)
        
        if improvement > -1e-6:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        
        diversity_ok = self._compute_diversity(self._population) > 1e-5
        return self._stagnation_counter >= 8 or not diversity_ok
    
    def _perform_restart(self):
        """Reinitialize while preserving best discovered solution."""
        best_idx = np.argmin(self._fitness)
        best_x, best_f = self._population[best_idx].copy(), self._fitness[best_idx]
        
        self._population = self._initialize_population()
        self._fitness = self._fitness_func(self._population)
        self._population[0] = best_x
        self._fitness[0] = best_f
        
        self._mean = np.mean(self._population, axis=0)
        self._evolution_path *= 0.1
        self._cov_diag = np.ones(self._dim) * ((self._upper_bound - self._lower_bound)[0] / np.sqrt(self._dim)) ** 2
        self._stagnation_counter = 0
    
    def _track_best_solution(self):
        """Update best fitness and solution if improvement found."""
        best_idx = np.argmin(self._fitness)
        if self._fitness[best_idx] < self._best_fitness:
            self._best_fitness = self._fitness[best_idx]
            self._best_x = self._population[best_idx].copy()
    
    def __call__(self, func, stopping_condition):
        """Execute optimization loop."""
        self._fitness_func = func
        self._population = self._initialize_population()
        self._fitness = func(self._population)
        self._track_best_solution()
        self._fitness_history.append(self._best_fitness)
        
        while not stopping_condition():
            trials = self._mutate_batch()
            trials = np.clip(trials, self._lower_bound, self._upper_bound)
            
            trial_fit = func(trials)
            if len(trial_fit) < len(trials):
                break
            
            if stopping_condition():
                break
            
            self._population, self._fitness, improved = self._select_batch(
                self._population, self._fitness, trials, trial_fit
            )
            
            self._update_success_history(improved)
            self._update_step_size_and_path(improved, trials)
            self._update_covariance_diag(improved, trials)
            self._update_mean()
            
            indices, local_trials = self._apply_memetic_refinement(self._population, self._fitness)
            local_fit = func(local_trials)
            if len(local_fit) == len(local_trials):
                improved_local = local_fit < self._fitness[indices]
                self._population[indices[improved_local]] = local_trials[improved_local]
                self._fitness[indices[improved_local]] = local_fit[improved_local]
            
            self._track_best_solution()
            self._fitness_history.append(self._best_fitness)
            
            if len(self._fitness_history) > 100:
                self._fitness_history.pop(0)
            
            if self._check_restart_criteria():
                self._perform_restart()
        
        return self._best_fitness, self._best_x
