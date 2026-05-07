**Idea: Restart-Triggered Covariance Reset with Isotropic Exploration**
When stagnation is detected (mean movement < threshold), this variant aggressively resets the covariance to isotropic (scaled identity), forcing exploration in all directions simultaneously. This is fundamentally different from archive-guided perturbation (variant_01) which continuously monitors trapping — this approach uses explicit stagnation triggers to make discrete, dramatic covariance resets that break out of local optima more forcefully.

```python
def _adapt_covariance(self):
    """Restart-triggered covariance reset with isotropic exploration for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute mean displacement magnitude (key stagnation indicator)
    mean_displacement = np.linalg.norm(y_mean)
    
    # Adaptive thresholds based on dimension
    stagnation_threshold = 0.1 * np.sqrt(self.dim)
    extreme_stagnation = mean_displacement < stagnation_threshold / 5.0
    
    # Track stagnation history
    if not hasattr(self):
        self.cov_stagnation_count = 0
    if mean_displacement < stagnation_threshold:
        self.cov_stagnation_count += 1
    else:
        self.cov_stagnation_count = max(0, self.cov_stagnation_count - 1)
    
    # Detect severe stagnation
    is_stagnated = self.cov_stagnation_count > max(3, self.dim // 10)
    
    # Evolution path update with adaptive rate
    if is_stagnated:
        # Increase pc learning rate when stagnant to speed up covariance adaptation
        cc_adaptive = min(self.cc * 2.0, 0.3)
    else:
        cc_adaptive = self.cc
    
    self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Base covariance update (standard CMA-ES)
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # RESTART-TRIGGERED ISOTROPIC RESET (core mechanism)
    if is_stagnated or extreme_stagnation:
        # Reset to isotropic covariance (explore uniformly in all directions)
        current_trace = np.trace(self.C) / self.dim
        self.C = current_trace * np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.cov_stagnation_count = 0  # Reset counter after reset
        
        # Increase step size to aid exploration after reset
        self.sigma = min(self.sigma * 2.0, 5.0)
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
    
    # Full reset on extreme ill-conditioning
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-10)
    
    if cond > 1e8 or eig_min < 1e-10:
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.L = None
```