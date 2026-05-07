**Idea: Diversity-Sensitive Covariance Adaptation**

A fundamentally different approach that detects convergence via population fitness variance and eigenvalue spread, then aggressively scales learning rates and injects perturbation to escape local optima on the hardest multi-modal tasks (17, 16, 6).

```python
def _adapt_covariance(self):
    """
    Adapt covariance matrix with diversity-sensitive learning rate scaling.
    Detects convergence via fitness variance and eigenvalue spread, then
    aggressively escapes local optima on multi-modal tasks.
    """
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    # Diversity detection: fitness variance normalized by scale
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Eigenvalue spread as second convergence signal
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    
    # Convergence detected when both signals indicate stagnation
    is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
    
    # Diversity-sensitive learning rate scaling
    if is_converging:
        # Up to 10x increase for escaped local optima
        scale = 10.0
    else:
        # Moderate boost proportional to diversity
        scale = 1.0 + 5.0 * min(rel_var, 0.1)
    
    ccov_scaled = min(self.ccov * scale, 0.5)
    cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
    
    # Recompute pc with scaled learning rate
    self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update with adaptive weights
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - ccov_scaled) * self.C + 
              ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
    
    # Inject exploration noise when converging
    if is_converging:
        self.C += 0.05 * np.eye(self.dim)
    
    # Ensure positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-10:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
```