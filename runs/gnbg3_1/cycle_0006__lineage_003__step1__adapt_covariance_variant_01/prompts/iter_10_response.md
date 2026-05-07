Looking at the error magnitudes, Tasks 16/17 stuck at ~10^1-10^2 indicate severe stagnation/local optima trapping. The key insight is that existing variants use fixed/deterministic adaptation rates that fail to respond to the specific failure mode: **premature convergence with ill-conditioned covariance matrices**.

I need a fundamentally different approach that explicitly **detects and escapes convergence traps** using condition number monitoring and adaptive learning rate modulation.

**Idea: Condition-Number Triggered Restart with Adaptive Learning**
```python
def _adapt_covariance_variant_01(self):
    """Condition-number triggered restart with adaptive learning rate scaling."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute condition number as diversity proxy
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
    
    # Adaptive learning rate based on condition number
    if cond > 1e6:
        ccov_scale = 0.1
        cc_scale = 0.1
    elif cond > 1e4:
        ccov_scale = 0.3
        cc_scale = 0.3
    elif cond > 1e2:
        ccov_scale = 0.5
        cc_scale = 0.5
    else:
        ccov_scale = 1.0
        cc_scale = 1.0
    
    # Fitness variance for additional adaptation signal
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    if rel_var < 1e-4:
        ccov_scale *= 0.5
        cc_scale *= 0.5
    
    # Trigger covariance restart on extreme condition number
    if cond > 1e8 or eig_spread < 1e-8:
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.L = None
    
    # Adaptive learning rates
    ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
    ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
    
    # Evolution path with adaptive rate
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine updates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
              ccov_1 * rank_one + 
              ccov_mu * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```