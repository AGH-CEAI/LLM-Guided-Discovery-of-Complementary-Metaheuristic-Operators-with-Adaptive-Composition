**Idea: sep-CMA-ES Baseline**
A separable covariance adaptation strategy that independently adapts each dimension's variance, providing robust performance across diverse problem types with improved numerical stability.

```python
def _adapt_covariance_variant_09(self):
    """sep-CMA-ES: Separable covariance adaptation for robust dimension-wise updates."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Adaptive learning rates based on recent improvement
    if not hasattr(self, 'recent_improvements'):
        self.recent_improvements = []
    
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    self.recent_improvements.append(improvement)
    if len(self.recent_improvements) > 10:
        self.recent_improvements.pop(0)
    
    avg_improvement = np.mean(self.recent_improvements) if self.recent_improvements else 0.0
    improv_scale = np.clip(1.0 + np.log1p(avg_improvement * 1e6), 0.5, 3.0)
    
    cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
    ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
    
    # Update evolution path
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Separable rank-one update: per-dimension contribution from evolution path
    pc_sq = self.pc ** 2
    
    # Separable rank-mu update: per-dimension weighted population variance
    rank_mu_diag = np.zeros(self.dim)
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu_diag += self.weights[i] * (diff ** 2)
    
    # Update diagonal of covariance matrix (separable adaptation)
    for d in range(self.dim):
        self.C[d, d] = ((1.0 - ccov_adapt) * self.C[d, d] +
                        ccov_adapt * pc_sq[d] +
                        (1.0 - 1.0 / self.mueff) * ccov_adapt * rank_mu_diag[d])
        self.C[d, d] = max(self.C[d, d], 1e-10)
    
    # Keep only diagonal (separable approach)
    np.fill_diagonal(self.C, np.diag(self.C))
    
    self.C = self._ensure_positive_definite(self.C)
```