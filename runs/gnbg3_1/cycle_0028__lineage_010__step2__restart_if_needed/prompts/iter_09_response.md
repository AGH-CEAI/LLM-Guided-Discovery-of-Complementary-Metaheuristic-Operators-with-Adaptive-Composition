Looking at the data, the worst unsolved tasks (17, 16, 10, 20, 18, 23, 12, 19) are stuck at errors of 10-75, indicating severe trapping in local optima. The current restart mechanism fully reinitializes the population, which may be too disruptive. For these high-error tasks, the optimizer needs more targeted escape mechanisms.

**Idea: Adaptive Hyperbolic Restart**
Instead of full reinitialization, this approach perturbs around the elite using adaptive step sizing based on how far the current error is from target. It keeps the best solution's region while introducing controlled exploration scaled to problem difficulty.

```python
def _restart_if_needed(self):
    """Adaptive restart with hyperbolic step boosting for high-error tasks."""
    diversity = self._compute_diversity()
    
    if not (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
        return
    
    best_idx = np.argmin(self.fitness)
    elite = self.population[best_idx].copy()
    elite_fit = self.fitness[best_idx]
    
    # Compute error magnitude for adaptive perturbation sizing
    error_mag = max(abs(elite_fit), 1.0)
    # Hyperbolic scaling: large errors get larger perturbations
    # log scale bridges 1e-8 (target) to 1e+2 (worst tasks)
    log_error = np.log10(error_mag + 1e-15)
    target_log = -8.0
    decades_from_target = max(log_error - target_log, 0.0)
    
    # Adaptive perturbation radius: scales with problem difficulty
    bound_range = max(self.ub[0] - self.lb[0], 1e-6)
    # Perturbation grows exponentially with decades from target
    # 0 decades → 0.1% of range; 10 decades → 30% of range
    perturb_factor = np.tanh(decades_from_target / 5.0) * 0.3
    perturb_radius = perturb_factor * bound_range
    perturb_radius = np.clip(perturb_radius, 1e-6, bound_range * 0.5)
    
    # Generate new population centered on elite with adaptive Gaussian perturbation
    new_pop = np.zeros((self.NP, self.dim))
    for i in range(self.NP):
        # Gaussian perturbation around elite, scaled by problem difficulty
        if i == 0:
            new_pop[i] = elite.copy()
        else:
            # Use isotropic Gaussian with adaptive radius
            new_pop[i] = elite + np.random.randn(self.dim) * perturb_radius
    
    # Clip to bounds
    new_pop = self._clip_to_bounds(new_pop)
    
    # Update state: keep new population, recompute covariance
    self.population = new_pop
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    # Recompute covariance from new population (learns new structure)
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.C = self._ensure_positive_definite(self.C)
    
    # Adapt sigma based on perturbation radius
    self.sigma = np.clip(perturb_radius / 3.0, 1e-10, 10.0)
    
    # Reset evolution paths for fresh covariance adaptation
    self.pc = np.zeros(self.dim)
    self.ps = np.zeros(self.dim)
    self.L = None
    
    # Evaluate new population
    self.fitness = self.func(self.population)
    
    # Update best
    new_best_idx = np.argmin(self.fitness)
    new_best_fit = float(np.asarray(self.fitness[new_best_idx]).flatten()[0])
    if new_best_fit < elite_fit:
        self.f_opt = new_best_fit
        self.x_opt = self.population[new_best_idx].copy()
    else:
        self.f_opt = float(np.asarray(elite_fit).flatten()[0])
        self.x_opt = elite.copy()
    
    self.f_opt_prev = self.f_opt
    self.stagnation_counter = 0
```