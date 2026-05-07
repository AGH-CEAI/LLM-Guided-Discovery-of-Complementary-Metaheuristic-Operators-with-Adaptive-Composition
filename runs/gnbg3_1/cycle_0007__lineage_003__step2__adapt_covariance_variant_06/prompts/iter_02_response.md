**Idea: Forced Diversity Injection with Eigenvalue Floor**

The worst unsolved tasks (16, 17, 20, 23, 10, 12, 18, 19) have errors of 10-100+, indicating catastrophic failure where the covariance matrix collapses or exploration dies completely. Existing variants only *scale* learning rates but never force diversity when the covariance becomes degenerate. This approach actively prevents collapse by enforcing eigenvalue floors and injecting fresh variance when the condition number exceeds critical thresholds.

```python
def _adapt_covariance_variant_06(self):
    """Forced diversity injection with eigenvalue floor to prevent catastrophic collapse."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Compute condition number and spread
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-15)
    eig_spread = eig_min / (eig_max + 1e-15)
    
    # Fitness variance for stagnation detection
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Detect collapse: extreme condition or vanishing spread
    is_collapsed = (cond > 1e6) or (eig_spread < 1e-6) or (rel_var < 1e-6)
    
    # Determine base learning rates
    if is_collapsed:
        # Force diversity: use small ccov, inject fresh variance
        ccov_eff = self.ccov * 0.2
        cc_eff = self.cc * 0.3
        
        # Enforce eigenvalue floor to prevent re-collapse
        target_min = 0.01 * np.mean(eigvals)
        if eig_min < target_min:
            self.C += (target_min - eig_min) * np.eye(self.dim)
        
        # Add isotropic perturbation for exploration burst
        noise_scale = 0.1 * np.mean(eigvals)
        self.C += noise_scale * np.eye(self.dim)
    else:
        # Normal operation with moderate damping based on condition
        damp_factor = 1.0 + min(cond / 1e4, 2.0)
        ccov_eff = self.ccov / damp_factor
        cc_eff = self.cc / max(damp_factor * 0.5, 0.5)
    
    # Recompute pc with adapted rate if collapsed
    if is_collapsed:
        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
    
    # Standard rank-mu update
    self.C = ((1.0 - ccov_eff) * self.C + 
              ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))
    
    # Final safety: enforce minimum eigenvalue after update
    final_eigvals = np.linalg.eigvalsh(self.C)
    final_min = np.min(final_eigvals)
    if final_min < 1e-12:
        self.C += (1e-10 - final_min) * np.eye(self.dim)
    
    self.C = self._ensure_positive_definite(self.C)
```