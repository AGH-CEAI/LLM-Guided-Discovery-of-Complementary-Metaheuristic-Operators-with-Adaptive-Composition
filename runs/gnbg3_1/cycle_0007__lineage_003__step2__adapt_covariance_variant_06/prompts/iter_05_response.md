Looking at the benchmark results, I see a clear pattern: **all 24 tasks are unsolved**, with the worst tasks (16, 17, 20, 23, 12) stuck at errors of 10-100. This suggests the algorithm is converging to local optima and losing diversity. The existing variants use adaptive learning rates and history tracking, but none have a truly effective **diversity-recovery mechanism**.

The key insight is that Tasks 10-23 (all with errors > 10) represent functions where CMA-ES gets trapped in local optima, and the only effective counter is a **targeted restart** that preserves the best solution while re-injecting diversity.

**Idea: Diversity-Triggered Anisotropic Restart**

This variant detects "stuck" states via fitness variance AND eigenvalue spread, then performs an anisotropic restart centered on the best solution with elite preservation. Unlike variant_01 (which restarts on extreme condition numbers) or variant_06 (which only scales learning rates), this approach actively regenerates population diversity in the direction of the best solution.

```python
def _adapt_covariance_variant_06(self):
    """Diversity-triggered anisotropic restart with elite preservation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    
    is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
    
    if is_converging:
        scale = 10.0
    else:
        scale = 1.0 + 5.0 * min(rel_var, 0.1)
    
    ccov_scaled = min(self.ccov * scale, 0.5)
    cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
    
    self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - ccov_scaled) * self.C + 
              ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
    
    if is_converging:
        self.C += 0.05 * np.eye(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Anisotropic restart if still stuck after adaptation
    fit_var_after = np.var(self.fitness)
    rel_var_after = fit_var_after / (fit_scale ** 2 + 1e-10)
    eigvals_after = np.linalg.eigvalsh(self.C)
    eig_spread_after = np.min(eigvals_after) / (np.max(eigvals_after) + 1e-10)
    
    needs_restart = (rel_var_after < 0.005) and (eig_spread_after < 1e-5)
    
    if needs_restart:
        self.pc = np.zeros(self.dim)
        self.C = np.eye(self.dim) * (self.sigma ** 2)
        
        sorted_idx = np.argsort(self.fitness)
        elites = [self.population[i].copy() for i in sorted_idx[:min(5, self.NP)]]
        
        spread = 0.3 * self.sigma
        new_pop = self.x_opt + np.random.randn(self.NP, self.dim) * spread
        new_pop = self._clip_to_bounds(new_pop)
        
        n_elites = min(5, self.NP)
        new_pop[:n_elites] = elites[:n_elites]
        
        self.population = new_pop
        self.fitness = self.func(self.population)
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        
        self.stagnation_counter = 0
```