Looking at the error magnitudes, the worst tasks (16-22) have errors of 10^1 to 10^2 — indicating the optimizer is getting catastrophically stuck in local optima far from global solutions. The current variant_08 only triggers exploration on stagnation counter, which appears insufficient. I need a fundamentally different approach that detects premature convergence via fitness variance AND matrix conditioning, then applies aggressive multi-direction perturbation.

**Idea: Fitness Variance-Triggered Multi-Eigendirection Exploration**

This variant detects convergence using BOTH low fitness variance AND high covariance condition number simultaneously, then applies stronger perturbation along the weakest (not strongest) eigendirections to escape local optima. It also scales learning rates dynamically based on population diversity.

```python
def _adapt_covariance_variant_08(self):
    """Fitness variance + condition-triggered multi-direction exploration for catastrophic local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Detect convergence using MULTIPLE signals
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    cond = np.max(eigvals) / np.min(eigvals)
    
    # Trigger exploration on BOTH low variance AND high condition
    is_converging = (rel_var < 1e-3) and (cond > 1e4)
    
    if is_converging:
        # AGGRESSIVE exploration: perturb along WEAKEST eigendirections
        # (opposite of current approach which perturbs strongest)
        k_perturb = min(self.dim, max(5, self.dim // 2))
        
        for i in range(k_perturb):
            # Perturb weakest directions most (they need the most help)
            strength = 3.0 * (i + 1) / k_perturb  # stronger for weaker eigenvectors
            self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])
        
        # Add substantial diagonal noise for uniform exploration
        self.C += 1.0 * np.eye(self.dim)
        
        # Reset evolution path completely
        self.pc = np.zeros(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    else:
        # Standard update with diversity-scaled learning rates
        ccov_scale = 1.0 + 20.0 * min(rel_var, 0.1)
        ccov_scale = np.clip(ccov_scale, 0.1, 10.0)
        
        cc_adapt = np.clip(self.cc * ccov_scale, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
        
        # Prevent extreme conditioning
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.min(eigvals_check)
        if cond_check > 1e7:
            self.C *= 0.5
```