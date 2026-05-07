**Idea: Multi-Elite Progressive Restart with Targeted Diversification**
A fundamentally different restart strategy that preserves multiple top performers and uses directed reinitialization based on failure mode (stagnation vs. diversity loss), rather than single-elite full reset.
```python
def _restart_if_needed(self):
    """Progressive multi-elite restart with targeted diversification."""
    diversity = self._compute_diversity()
    
    needs_restart = (self.stagnation_counter > self.max_stagnation or 
                     diversity < self.min_diversity or 
                     np.any(np.isnan(self.C)))
    
    if not needs_restart:
        return
    
    # Preserve top 15% performers (fundamentally different: multi-elite vs single)
    num_elites = max(1, int(0.15 * self.NP))
    elite_indices = np.argsort(self.fitness)[:num_elites]
    elites = self.population[elite_indices].copy()
    elite_fitness = self.fitness[elite_indices].copy()
    
    # Compute elite statistics for directed diversification
    elite_centroid = np.mean(elites, axis=0)
    elite_spread = np.std(elites, axis=0) + 1e-10
    
    # Determine spread factor based on failure mode
    if diversity < self.min_diversity:
        spread_factor = 2.5  # More aggressive for low diversity
    elif self.stagnation_counter > self.max_stagnation * 1.5:
        spread_factor = 2.0  # Moderate for severe stagnation
    else:
        spread_factor = 1.5  # Baseline
    
    # Reinitialize non-elite population with directed sampling
    for i in range(num_elites, self.NP):
        # Blend: 60% around elite centroid, 40% global exploration
        if np.random.random() < 0.6:
            candidate = elite_centroid + spread_factor * elite_spread * np.random.randn(self.dim)
        else:
            candidate = np.random.uniform(self.lb, self.ub)
        self.population[i] = self._clip_to_bounds(candidate)
    
    # Add boundary-focused samples for multimodal tasks (15% of population)
    n_boundary = max(2, int(0.15 * self.NP))
    for i in range(n_boundary):
        idx = self.NP - 1 - i
        u = np.random.random(self.dim)
        boundary = np.where(u < 0.5, np.power(u * 2, 3), 1.0 - np.power((1.0 - u) * 2, 3))
        self.population[idx] = self.lb + boundary * (self.ub - self.lb)
    
    # Reinsert elites at top positions
    for i, (elite, fit) in enumerate(zip(elites, elite_fitness)):
        self.population[i] = elite
        self.fitness[i] = fit
    
    # Reset covariance to reflect new population structure
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
    self.C = self._ensure_positive_definite(self.C)
    
    # Reset step-size and evolution path to avoid stale momentum
    self.sigma = max(self.sigma, 0.1 * (self.ub[0] - self.lb[0]) / 3.0)
    self.ps = np.zeros(self.dim)
    self.pc = np.zeros(self.dim)
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    # Clear adaptive operator state to allow fresh exploration
    self._reset_operator_state()
    self.stagnation_counter = 0
```