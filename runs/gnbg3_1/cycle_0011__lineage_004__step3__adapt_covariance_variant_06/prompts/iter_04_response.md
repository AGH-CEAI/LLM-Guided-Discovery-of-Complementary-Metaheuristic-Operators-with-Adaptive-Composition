Looking at the unsolved tasks, I see errors stuck at 1e+1 to 6e+1 level, which indicates the optimizer is fundamentally trapped in local optima or suffering from covariance collapse. The current variant_06 only scales learning rates during convergence but doesn't actually escape these traps.

**Idea: Forced Restart with Elite Preservation**
Periodically force a complete reinitialization with Latin Hypercube sampling while preserving the best solutions found. This is fundamentally different from scaling learning rates because it completely resets exploration when trapped.

```python
def _adapt_covariance_variant_06(self):
    """Forced restart with elite preservation for escaping deep local optima."""
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
    
    # Force restart logic: detect extreme stagnation or ill-conditioning
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)
    
    stagnation_threshold = max(20, self.dim * 2)
    should_restart = (self.stagnation_counter > stagnation_threshold) or (cond > 1e6) or (np.min(eigvals) < 1e-12)
    
    if should_restart:
        # Track best found so far
        best_idx = np.argmin(self.fitness)
        best_pos = self.population[best_idx].copy()
        best_fit = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        
        # Reinitialize with fresh LHS population
        new_pop = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            new_pop[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = new_pop
        self.fitness = self.func(self.population)
        
        # Insert elite at random position
        elite_slot = np.random.randint(0, self.NP)
        self.population[elite_slot] = best_pos
        self.fitness[elite_slot] = best_fit
        
        # Re-evaluate to ensure consistency
        self.fitness = self.func(self.population)
        
        # Reset CMA-ES state
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = 0.5 * (self.ub[0] - self.lb[0])
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        # Update best from new population
        best_new_idx = np.argmin(self.fitness)
        new_best_fit = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
        if new_best_fit < best_fit:
            self.f_opt = new_best_fit
            self.x_opt = self.population[best_new_idx].copy()
        else:
            self.f_opt = best_fit
            self.x_opt = best_pos.copy()
        
        self.stagnation_counter = 0
```