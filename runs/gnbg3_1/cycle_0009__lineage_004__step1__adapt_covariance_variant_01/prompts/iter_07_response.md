**Idea: Isotropic Reset with Aggressive Exploration Scaling**

The worst unsolved tasks (16, 17, 20, 18, 23, 19, 11, 12) have errors stuck at 10-80, meaning the algorithm is trapped in local optima with a collapsed covariance matrix. The current variant_01 only triggers restart on extreme condition numbers (>1e8), which is too conservative. This variant implements an **isotropic reset with aggressive scaling** that explicitly detects stagnation and resets the covariance to a sphere, forcing re-exploration when progress stalls.

```python
def _adapt_covariance_variant_01(self):
    """Isotropic reset with aggressive exploration scaling for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Detect stagnation: no improvement in recent generations
    is_stagnant = self.stagnation_counter > self.max_stagnation // 2
    
    # Also trigger reset if covariance is too elongated
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-10)
    
    # Fitness-based trigger: if best fitness hasn't improved and condition is high
    fit_var = np.var(self.fitness)
    rel_var = fit_var / (max(abs(self.f_opt), 1.0) ** 2 + 1e-10)
    is_converged = rel_var < 1e-3 and cond > 1e3
    
    # Force isotropic reset on stagnation or extreme conditioning
    if is_stagnant or is_converged or cond > 1e7:
        # Reset to isotropic covariance with larger spread
        C_mean = np.mean(eigvals)
        self.C = np.eye(self.dim) * max(C_mean, self.sigma ** 2)
        self.pc = np.zeros(self.dim)
        self.L = None
        
        # Increase step size to encourage exploration
        if self.sigma < 0.5 * (self.ub[0] - self.lb[0]) / 3.0:
            self.sigma *= 2.0
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    # Adaptive learning rates based on condition number
    if cond > 1e6:
        ccov_scale = 0.2
        cc_scale = 0.2
    elif cond > 1e4:
        ccov_scale = 0.4
        cc_scale = 0.4
    elif cond > 1e2:
        ccov_scale = 0.7
        cc_scale = 0.7
    else:
        ccov_scale = 1.0
        cc_scale = 1.0
    
    # Additional scaling for stagnant/poorly-conditioned cases
    if is_stagnant or is_converged:
        ccov_scale *= 2.0
        cc_scale *= 1.5
    
    ccov_scale = min(ccov_scale, 2.0)
    cc_scale = min(cc_scale, 1.0)
    
    # Adaptive learning rates
    ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
    ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
    
    # Evolution path update
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