**Idea: Progressive Restart with Normalized Intensity**
A fundamentally different strategy that detects stagnation via fitness variance and triggers *progressive restarts* with intensity normalized by generational progress. Unlike previous variants that try to perturb the covariance in place, this approach completely resets key parameters (step size, pc, C) while preserving the best solution, injecting large controlled perturbations that scale with how long the algorithm has been stuck.

```python
def _adapt_covariance_original(self):
    """Progressive restart with normalized intensity for escaping severe stagnation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Standard evolution path update (unchanged)
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Detect stagnation via fitness variance
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    threshold = max(fit_scale * 1e-6, 1e-8)
    is_stagnant = fit_var < threshold
    
    if is_stagnant:
        # Normalize restart intensity by generational progress to prevent excessive restarts
        gen_norm = max(self.generation, 1)
        restart_intensity = min(1.0 + np.log1p(gen_norm) / gen_norm, 5.0)
        
        # Preserve best solution
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]
        
        # Reinitialize population around best solution
        spread = restart_intensity * 0.5 * (self.ub[0] - self.lb[0])
        samples = np.random.randn(self.NP, self.dim)
        for d in range(self.dim):
            bins = np.linspace(elite[d] - spread, elite[d] + spread, self.NP + 1)
            bins = np.clip(bins, self.lb[d], self.ub[d])
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = self._clip_to_bounds(samples)
        self.fitness = self.func(self.population)
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Reset and scale step size
        self.sigma = min(self.sigma * (1.0 + restart_intensity * 0.5), 0.3 * (self.ub[0] - self.lb[0]))
        
        # Reset and perturb covariance matrix
        eigvals = np.linalg.eigvalsh(self.C)
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.C += restart_intensity * 0.2 * np.random.randn(self.dim, self.dim)
        self.C = 0.5 * (self.C + self.C.T)
        self.C = self._ensure_positive_definite(self.C)
        
        # Reset evolution paths
        self.pc = np.zeros(self.dim)
        
        # Scale up learning rates for faster adaptation after restart
        self.cc = min(self.cc * (1.0 + restart_intensity * 0.3), 0.3)
        self.ccov = min(self.ccov * (1.0 + restart_intensity * 0.5), 0.5)
        
        # Update best if elite is better
        if elite_fit < self.f_opt:
            self.f_opt = elite_fit
            self.x_opt = elite.copy()
```