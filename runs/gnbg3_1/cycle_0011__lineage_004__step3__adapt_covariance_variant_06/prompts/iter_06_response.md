**Idea: Eigenvalue-Ratio Restart Escaping**
Uses eigendecomposition to detect ill-conditioned covariance (which causes the ~1e+1 stuck errors on tasks 16, 17, 11) and triggers aggressive restart with perturbations along smallest eigendirections to escape local basins.

```python
def _adapt_covariance_variant_06(self):
    """Eigenvalue-ratio restart escaping for ill-conditioned traps."""
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-15)
    cond = np.max(eigvals) / np.min(eigvals)
    
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    restart_threshold = max(1e4, self.dim * 100)
    should_restart = (cond > restart_threshold) or (rel_var < 1e-6 and cond > self.dim * 10)
    
    if should_restart:
        # Perturb along smallest eigendirections (most compressed = most trapped)
        min_eig = np.min(eigvals)
        for i in range(min(5, self.dim)):
            strength = min_eig * 5.0 * (i + 1)
            self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])
        
        # Global expansion for fresh exploration
        self.C += 0.3 * np.mean(eigvals) * np.eye(self.dim)
        
        # Reset evolution path and increase step size
        self.pc *= 0.0
        self.sigma = min(self.sigma * 3.0, (self.ub[0] - self.lb[0]) * 0.1)
        self.stagnation_counter = 0
        
        # Ensure well-conditioned restart
        eigvals_new = np.linalg.eigvalsh(self.C)
        eigvals_new = np.maximum(eigvals_new, 1e-10)
        if np.max(eigvals_new) / np.min(eigvals_new) > 1e6:
            self.C = np.eye(self.dim) * np.mean(eigvals_new)
        return
    
    # Adaptive rates based on fitness variance
    rate_scale = 1.0 + 10.0 * min(rel_var, 0.1)
    cc_adapt = np.clip(self.cc * rate_scale, 0.001, 0.3)
    ccov_adapt = np.clip(self.ccov * rate_scale, 1e-10, 0.5)
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * rank_one +
              (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```