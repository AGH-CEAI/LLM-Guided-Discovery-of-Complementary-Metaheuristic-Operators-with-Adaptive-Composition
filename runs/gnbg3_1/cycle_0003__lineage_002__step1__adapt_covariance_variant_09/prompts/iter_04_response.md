**Idea: Eigenvalue-Driven Restarts with Forced Exploration**

The worst unsolved tasks (16, 17, 18, 20, 23, 12, 19, 10) are stuck at errors ~1e+1 to 1e+2, meaning the optimizer is trapped far from the global optimum. The current variant_09 uses moderate ccov scaling (up to 1.5x) which is insufficient to escape these deep local optima. This variant uses **aggressive restart triggers** when stagnation is severe: it perturbs the best solution and reinitializes the population with forced exploration using eigenvalue manipulation and adaptive noise injection proportional to stagnation depth.

```python
def _adapt_covariance_variant_09(self):
    """Eigenvalue-driven restarts with forced exploration for severely stuck tasks."""
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.improvement_ema = 1.0
    
    improvement = max(1e-10, self.prev_f_opt - self.f_opt)
    self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
    self.prev_f_opt = self.f_opt
    
    stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
    
    # Track stagnation depth for aggressive restarts
    if not hasattr(self, 'stagnation_depth'):
        self.stagnation_depth = 0
    if stagnation:
        self.stagnation_depth += 1
    else:
        self.stagnation_depth = max(0, self.stagnation_depth - 2)
    
    # Force restart when severely stuck
    if self.stagnation_depth >= 20 or (stagnation and diversity < 0.05):
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]
        
        # Adaptive noise scaling
        noise_scale = min(10.0, 1.0 + 0.5 * self.stagnation_depth)
        
        # Eigenvalue-based forced exploration
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            log_eigvals = np.log(eigvals + 1e-10)
            target_log = np.mean(log_eigvals) + 2.0 * np.std(log_eigvals)
            eigvals_scaled = np.exp(np.clip(log_eigvals, target_log - 2.0, target_log + 2.0))
            self.C = eigvecs @ np.diag(eigvals_scaled) @ eigvecs.T
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim) * (self.ub[0] - self.lb[0]) ** 2 * 0.01
        
        # Perturb elite and reinitialize
        pert_elite = elite + np.random.randn(self.dim) * noise_scale * self.sigma
        self.population = np.random.uniform(
            self.lb, self.ub, (self.NP, self.dim)
        )
        self.population[0] = self._clip_to_bounds(pert_elite)
        
        # Increase step size for exploration
        self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.5)
        
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        self.fitness = self.func(self.population)
        self.stagnation_depth = 0
        self.stagnation_counter = 0
        
        best_new = np.argmin(self.fitness)
        if self.fitness[best_new] < self.f_opt:
            self.f_opt = float(np.asarray(self.fitness[best_new]).flatten()[0])
            self.x_opt = self.population[best_new].copy()
        return
    
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    ccov_adaptive = base_ccov * (2.0 if self.stagnation_depth >= 5 else 1.0) * (2.0 if diversity < 0.3 else 1.0)
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    rank_mu_scale = 3.0 if (stagnation or diversity < 0.3) else 1.0
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              ccov_adaptive * rank_mu_scale * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```