**Idea: Success-Ratio Rank-Mu Weighting**

A fundamentally different covariance adaptation that replaces static rank-based weights with dynamically computed success-ratio weights based on how well each selected individual improved over the population median. This targets the failure mode where static weights fail to emphasize the most consistently improving directions across generations, which is likely why all variants get stuck at ~1e+1 error on the worst tasks.

```python
def _adapt_covariance_original(self):
    """Success-ratio covariance adaptation with dynamic rank weighting."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    # Compute fitness median for success ratio baseline
    median_fit = np.median(self.fitness)
    
    # Compute success-ratio weights dynamically (key difference from static weights)
    success_weights = np.zeros(self.mu)
    for i in range(self.mu):
        fitness_i = self.fitness[i]
        # Ratio of improvement over median (capped for stability)
        improvement_ratio = (median_fit - fitness_i) / (abs(median_fit) + 1e-10)
        # Only positive improvements contribute, scaled logarithmically
        if improvement_ratio > 0:
            success_weights[i] = np.log1p(improvement_ratio * 10.0)
    
    # Normalize success weights
    sum_sw = np.sum(success_weights)
    if sum_sw > 1e-15:
        success_weights /= sum_sw
        # Blend with original log-based weights for stability
        final_weights = 0.6 * success_weights + 0.4 * self.weights[:self.mu]
        final_weights /= np.sum(final_weights)
    else:
        final_weights = self.weights[:self.mu]
    
    # Rank-mu update with success-based weighting
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += final_weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rate based on condition number
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.maximum(np.min(eigvals), 1e-15)
    eig_max = np.max(eigvals)
    cond = eig_max / eig_min
    
    # Increase learning rate when poorly conditioned (harder problems)
    if cond > 1e4:
        ccov_eff = np.clip(self.ccov * 2.0, 1e-8, 0.5)
        cc_eff = np.clip(self.cc * 1.5, 0.001, 0.3)
    elif cond > 1e2:
        ccov_eff = np.clip(self.ccov * 1.2, 1e-8, 0.5)
        cc_eff = self.cc
    else:
        ccov_eff = self.ccov
        cc_eff = self.cc
    
    # Re-compute pc with adaptive rate
    self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Combine updates
    self.C = ((1.0 - ccov_eff) * self.C + 
              ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```