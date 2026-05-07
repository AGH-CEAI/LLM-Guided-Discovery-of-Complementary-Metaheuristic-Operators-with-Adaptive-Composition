**Idea: Adaptive Condition Number Steering**

A fundamentally different approach that actively steers the covariance matrix's condition number toward a target range by dynamically rescaling eigenvalues, preventing collapse on ill-conditioned problems. Unlike prior variants that react to stagnation, this proactively maintains exploration capacity.

```python
def _adapt_covariance_variant_06(self):
    """Adaptive condition number steering for robust exploration."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Compute current condition number
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-15)
    cond_curr = np.max(eigvals) / np.min(eigvals)
    
    # Target condition range for good exploration
    cond_target = min(self.dim, 1e4)
    cond_max = 1e6
    
    # Determine scaling factor to steer condition number
    if cond_curr > cond_max:
        # Severe collapse - aggressive rescaling
        scale_factor = np.sqrt(cond_curr / cond_target)
        eigvals = eigvals * scale_factor
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.pc *= 0.5  # Dampen evolution path
    elif cond_curr > cond_target:
        # Moderate steering toward target
        scale_factor = np.sqrt(cond_curr / cond_target)
        eigvals = eigvals * scale_factor
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    else:
        # Healthy condition - add slight axis-aligned perturbation for robustness
        min_eig = np.min(eigvals)
        perturbation = 0.01 * min_eig * np.ones_like(eigvals)
        eigvals = eigvals + perturbation
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Detect premature convergence via fitness stagnation
    rel_var = np.var(self.fitness) / (max(abs(self.f_opt), 1.0) ** 2 + 1e-10)
    is_premature = rel_var < 1e-3 and cond_curr > 1e3
    
    # Adaptive learning rate based on condition health
    if is_premature:
        ccov_boost = 3.0
        cc_boost = 2.0
    else:
        ccov_boost = 1.0
        cc_boost = 1.0
    
    ccov_scaled = np.clip(self.ccov * ccov_boost, 1e-8, 0.5)
    cc_scaled = np.clip(self.cc * cc_boost, 0.001, 0.3)
    
    # Re-compute pc with scaled rate if premature
    if is_premature:
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Apply covariance update
    self.C = ((1.0 - ccov_scaled) * self.C + 
              ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```