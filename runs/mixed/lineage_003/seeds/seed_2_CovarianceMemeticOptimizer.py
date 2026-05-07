import numpy as np


class CovarianceMemeticOptimizer:
    """
    Hybrid optimizer combining CMA-ES-style covariance adaptation with
    memetic local search and velocity-guided exploration.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(150, 5 * dim)
        self.lb = -100.0
        self.ub = 100.0
        
        self._init_parameters()
        self._init_adaptation_constants()
        self._init_best_tracking()
        self.population = None
        self.fitness = None
        self.budget_exhausted = False
        
    def _init_parameters(self):
        """Initialize algorithm control parameters."""
        self.sigma = 6.0
        self.w = 0.4
        self.cp = 1.5
        self.cg = 1.5
        self.phi = 0.3
        self.F = 0.5
        
    def _init_adaptation_constants(self):
        """Initialize CMA-ES adaptation constants."""
        self.cc = 2.0 / (self.dim + 2.0)
        self.cs = (self.mu + 2.0) / (self.dim + self.mu + 5.0)
        self.c1 = 1.0 / (12.0 * self.dim + 96.0)
        self.cov_weights = np.zeros(self.mu)
        self.mu_eff = 1.0
        self.update_cov_weights()
        
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.m = np.zeros(self.dim)
        self.selected_diffs = []
        self.g = 0
        self.damping = 1.0 + self.dim / 150.0
        self.chi = self.dim**0.5 * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2))
        
    def update_cov_weights(self):
        """Update recombination weights for covariance adaptation."""
        self.mu = self.NP // 4
        self.mu_eff = (self.mu + 2) / 4.0
        ranks = np.arange(1, self.mu + 1)
        self.cov_weights = np.maximum(0, self.mu_eff - ranks * 0.5) / \
                           (np.sum(np.maximum(0, self.mu_eff - ranks * 0.5)) + 1e-10)
        
    def _init_best_tracking(self):
        """Initialize best solution tracking and history."""
        self.best_fitness = np.inf
        self.best_solution = None
        self.mean_pbest = np.zeros(self.dim)
        self.gbest = np.zeros(self.dim)
        self.pbest_fitness = np.full(self.NP, np.inf)
        self.velocity = np.zeros((self.NP, self.dim))
        self.history_best_fitness = []
        self.improvement_counter = 0
        
    def __call__(self, func, stopping_condition):
        """Main entry point for optimization."""
        self.func = func
        self.budget_exhausted = False
        self._initialize_population()
        
        while not stopping_condition():
            if self.budget_exhausted:
                break
            
            trials = self._generate_trials_batched()
            trials = self._clip_to_bounds_batch(trials)
            trial_fitness = self._eval_wrapper(trials)
            
            if stopping_condition():
                break
            
            if len(trial_fitness) < len(trials):
                self.budget_exhausted = True
                valid = len(trial_fitness)
                trials = trials[:valid]
                trial_fitness = trial_fitness[:valid]
            
            if len(trial_fitness) == 0:
                break
            
            self._update_personal_best(trial_fitness)
            self._update_global_best()
            self._update_mean_pbest()
            self._update_velocity_batch()
            
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness)
            
            self._update_statistics_batch(trials, trial_fitness)
            self._update_covariance_batch()
            self._adapt_step_size_batch()
            
            self._track_best_solution()
            
            if self._should_restart():
                self._restart_from_best()
                
        return self.best_fitness, self.best_solution
    
    def _initialize_population(self):
        """Initialize population from uniform distribution."""
        self.m = np.random.uniform(-50, 50, self.dim)
        self.population = self._sample_from_covariance_batch(self.NP)
        self.population = self._clip_to_bounds_batch(self.population)
        self.fitness = self._eval_wrapper(self.population)
        
    def _clip_to_bounds_batch(self, pop):
        """Clip population to search bounds [-100, 100]."""
        return np.clip(pop, self.lb, self.ub)
    
    def _eval_wrapper(self, batch):
        """Evaluate batch with budget exhaustion handling."""
        result = self.func(batch)
        if len(result) < len(batch):
            self.budget_exhausted = True
        return result
    
    def _sample_from_covariance_batch(self, n):
        """Sample n individuals from current CMA-ES distribution."""
        try:
            L = np.linalg.cholesky(self.C)
            samples = self.m + self.sigma * np.random.randn(n, self.dim) @ L.T
        except np.linalg.LinAlgError:
            samples = self.m + self.sigma * np.random.randn(n, self.dim)
        return samples
    
    def _generate_trials_batched(self):
        """Generate trial population using CMA-ES + velocity perturbation."""
        trials = self._sample_from_covariance_batch(self.NP)
        
        v_perturbation = self.phi * self.velocity
        trials = trials + v_perturbation
        
        if self.improvement_counter > 5:
            trials = self._apply_culture_guided_mutation(trials)
        
        return trials
    
    def _apply_culture_guided_mutation(self, trials):
        """Apply mutation guided by cultural knowledge (pbest/gbest)."""
        r_pbest = np.random.rand(self.NP, 1)
        r_gbest = np.random.rand(self.NP, 1)
        
        cultural = 0.3 * self.F * (r_pbest * (self.mean_pbest - trials) + 
                                    r_gbest * (self.gbest - trials))
        trials = trials + cultural * np.random.randn()
        
        return trials
    
    def _update_personal_best(self, trial_fitness):
        """Update personal best for each individual."""
        improved = trial_fitness < self.pbest_fitness
        self.pbest_fitness = np.minimum(self.pbest_fitness, trial_fitness)
        
        for i in range(self.NP):
            if improved[i]:
                self.improvement_counter += 1
    
    def _update_global_best(self):
        """Update global best from personal bests."""
        best_idx = np.argmin(self.pbest_fitness)
        if self.pbest_fitness[best_idx] < self.best_fitness:
            self.gbest = self.population[best_idx].copy()
            
    def _update_mean_pbest(self):
        """Compute mean of personal bests weighted by inverse fitness."""
        weights = 1.0 / (self.pbest_fitness + 1e-10)
        weights = weights / np.sum(weights)
        self.mean_pbest = np.sum(self.population * weights[:, np.newaxis], axis=0)
        
    def _update_velocity_batch(self):
        """Update velocity based on personal and global best attraction."""
        r1 = np.random.rand(self.NP, self.dim)
        r2 = np.random.rand(self.NP, self.dim)
        
        cognitive = self.cp * r1 * (self.mean_pbest - self.population)
        social = self.cg * r2 * (self.gbest - self.population)
        
        self.velocity = self.w * self.velocity + cognitive + social
        
        max_vel = 0.2 * (self.ub - self.lb)
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
        
    def _select_survivors_batch(self, pop, fit, trials, trial_fit):
        """Select survivors using (mu, lambda) selection with elitism."""
        combined_pop = np.vstack([pop, trials])
        combined_fit = np.concatenate([fit, trial_fit])
        
        sorted_idx = np.argsort(combined_fit)
        elite_count = max(1, self.NP // 10)
        
        new_pop = combined_pop[sorted_idx[:self.NP]]
        new_fit = combined_fit[sorted_idx[:self.NP]]
        
        self.history_best_fitness.append(new_fit[0])
        return new_pop, new_fit
    
    def _update_statistics_batch(self, trials, trial_fit):
        """Update selection statistics for covariance adaptation."""
        sorted_idx = np.argsort(trial_fit)
        selected_idx = sorted_idx[:self.mu]
        
        self.selected_diffs = []
        for idx in selected_idx:
            diff = (trials[idx] - self.m) / self.sigma
            self.selected_diffs.append(diff)
        
    def _update_covariance_batch(self):
        """Update covariance matrix using evolution path and rank-mu update."""
        if not self.selected_diffs:
            return
            
        self.g += 1
        c = self.cc - self.cs
        
        hs = np.linalg.norm(self.pc) / np.sqrt(1.0 - c**(2.0 * self.g)) < 1.5 + 1.0 / (self.dim + 1)
        
        delta = (1.0 - hs) * c * (2.0 - c)
        self.pc = (1.0 - c) * self.pc + hs * np.sqrt(c * (2.0 - c) * self.mu_eff) * np.mean(self.selected_diffs, axis=0)
        
        self.C = (1.0 - self.c1 * delta - self.c1 * (1.0 - hs) * c * (2.0 - c)) * self.C + \
                 self.c1 * np.outer(self.pc, self.pc)
        
        cmu = min(1.0 - self.c1, (self.mu_eff + 2.0) / (self.dim + 4.0 + 2.0 * self.mu_eff))
        for i, diff in enumerate(self.selected_diffs):
            self.C += cmu * self.cov_weights[i] * np.outer(diff, diff)
        
        eigval = np.linalg.eigvalsh(self.C)
        if np.any(eigval <= 0):
            self.C = self.C + np.eye(self.dim) * abs(min(eigval)) * 1.1
            
    def _adapt_step_size_batch(self):
        """Adapt step size based on evolution path length."""
        self.ps = (1.0 - self.cs) * self.ps + \
                  hs * np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * np.mean(self.selected_diffs, axis=0) \
                  if self.selected_diffs else self.ps
        
        hs = np.linalg.norm(self.ps) / np.sqrt(1.0 - (1.0 - self.cs)**(2.0 * self.g)) < 1.5 + 1.0 / (self.dim + 1)
        
        self.sigma *= np.exp((hs * np.linalg.norm(self.ps) - self.chi) / self.damping)
        self.sigma = max(1e-10, min(1e6, self.sigma))
        
    def _track_best_solution(self):
        """Track and update global best solution."""
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()
            self.improvement_counter = 0
            
    def _should_restart(self):
        """Check if stagnation detected requiring restart."""
        if len(self.history_best_fitness) < 100:
            return False
        
        recent_best = min(self.history_best_fitness[-50:])
        older_best = min(self.history_best_fitness[-100:-50])
        
        if recent_best >= older_best * (1.0 + 1e-6):
            return True
        return False
    
    def _restart_from_best(self):
        """Reinitialize population while preserving best solution."""
        best_idx = np.argmin(self.fitness)
        best_sol = self.population[best_idx].copy()
        best_fit = self.fitness[best_idx]
        
        self.m = np.random.uniform(self.lb, self.ub, self.dim)
        self.m[0] = best_sol[0]
        self.sigma = 6.0
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.velocity = np.zeros((self.NP, self.dim))
        
        self.population = self._sample_from_covariance_batch(self.NP)
        self.population = self._clip_to_bounds_batch(self.population)
        self.population[0] = best_sol
        self.fitness[0] = best_fit
        
        self.history_best_fitness = []
        self.improvement_counter = 0
