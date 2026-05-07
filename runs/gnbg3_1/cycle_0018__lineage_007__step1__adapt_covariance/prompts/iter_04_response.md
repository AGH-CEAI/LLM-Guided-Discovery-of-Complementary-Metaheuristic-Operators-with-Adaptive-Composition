**Idea: Forced Restart with Elite Injection**

The worst unsolved tasks (16, 17, 23, 20, 18, 19) have errors of 10-70, indicating the population is trapped in a basin far from global optimum. Existing variants only perturb the covariance matrix within the current distribution — they cannot escape if the search has collapsed to a wrong region. This approach **forcibly restarts the search** when trapped, injecting the current best solution as an elite anchor while redispersing the population across the search space.

```python
def _adapt_covariance(self):
    """Forced restart with elite injection for escaping catastrophic local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Detect catastrophic trapping: extreme condition number OR near-zero variance
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)
    
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    pop_spread = np.mean(np.std(self.population, axis=0))
    bound_range = self.ub[0] - self.lb[0]
    rel_spread = pop_spread / (bound_range + 1e-10)
    
    is_catastrophically_trapped = (cond > 1e6) or (rel_var < 1e-5) or (rel_spread < 1e-4)
    
    # Also trigger if error is catastrophically large (stuck far from optimum)
    is_catastrophically_far = self.f_opt > 1e3
    
    if is_catastrophically_trapped or is_catastrophically_far:
        # Perform a forced restart with elite injection
        elite_pos = self.x_opt.copy()
        elite_fit = self.f_opt
        
        # Reinitialize population with LHS
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Inject elite at center of search space with small perturbation
        center = 0.5 * (self.ub + self.lb)
        perturbation_scale = 0.1 * (self.ub - self.lb)
        elite_pos_perturbed = elite_pos + np.random.randn(self.dim) * perturbation_scale
        elite_pos_perturbed = self._clip_to_bounds(elite_pos_perturbed)
        
        # Place elite near center of new population
        self.population[0] = elite_pos_perturbed
        self.fitness = self.func(self.population)
        
        # Update best if elite is better than random init
        new_best_idx = np.argmin(self.fitness)
        new_best_fit = float(np.asarray(self.fitness[new_best_idx]).flatten()[0])
        if new_best_fit < self.f_opt:
            self.f_opt = new_best_fit
            self.x_opt = self.population[new_best_idx].copy()
        
        # Reset CMA-ES state
        self.C = np.eye(self.dim) * ((bound_range / 6.0) ** 2)
        self.sigma = bound_range / 6.0
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.stagnation_counter = 0
        
        # Keep operator selection state but reset internal tracking
        self.generation = 0
```