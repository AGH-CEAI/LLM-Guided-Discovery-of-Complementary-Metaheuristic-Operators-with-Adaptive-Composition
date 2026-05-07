**Idea: Eigenspace Perturbation with Condition-Aware Adaptation**

This approach directly manipulates the covariance eigenspectrum to escape the catastrophic convergence causing 10^1-10^2 errors on the hardest tasks. Unlike evolution-path-based variants, it monitors eigenvalue spread and injects exploration along minor eigendirections when ill-conditioning is detected, combined with adaptive learning rates that scale with problem conditioning.

```python
def _adapt_covariance_original(self):
    """Eigenspace perturbation with condition-aware adaptation for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Compute eigendecomposition for condition monitoring
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
    except np.linalg.LinAlgError:
        eigvals = np.diag(self.C)
        eigvecs = np.eye(self.dim)
    
    eig_min = np.maximum(np.min(eigvals), 1e-12)
    eig_max = np.max(eigvals)
    cond = eig_max / eig_min
    eig_spread = eig_min / (eig_max + 1e-12)
    
    # Detect catastrophic convergence (primary failure mode on hardest tasks)
    is_trapped = (cond > 1e4) or (eig_spread < 1e-4) or (eig_min < 1e-8)
    
    # Adaptive learning rates based on condition number
    if cond > 1e6:
        ccov_scale = 0.1
        cc_scale = 0.5
    elif cond > 1e3:
        ccov_scale = 0.3
        cc_scale = 0.7
    else:
        ccov_scale = 1.0
        cc_scale = 1.0
    
    ccov_eff = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
    cc_eff = np.clip(self.cc * cc_scale, 0.001, 0.3)
    
    # Rebuild evolution path with scaled learning rate
    self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Standard rank-mu update
    self.C = ((1.0 - ccov_eff) * self.C + 
              ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))
    
    # Perturb eigenspectrum if trapped
    if is_trapped:
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            
            # Clip eigenvalues to prevent collapse
            eigvals = np.clip(eigvals, 1e-10, None)
            
            # Inject exploration along minor eigendirections
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            exploration_budget = 0.1 * max_eig
            
            # Distribute exploration across bottom 50% of eigendirections
            n_explore = max(1, self.dim // 2)
            sorted_indices = np.argsort(eigvals)
            for idx in sorted_indices[:n_explore]:
                eigvals[idx] += exploration_budget / n_explore
            
            # Rebuild covariance from perturbed spectrum
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        except np.linalg.LinAlgError:
            # Fallback: add isotropic exploration
            self.C += 0.05 * np.eye(self.dim)
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
```