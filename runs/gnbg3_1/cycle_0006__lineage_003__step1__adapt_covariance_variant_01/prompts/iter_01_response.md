**Idea: Diagonal-Only (sep-CMA) Covariance Adaptation**
A fundamentally different approach using only diagonal covariance elements with separate learning rates for each dimension, providing faster adaptation on axis-aligned problems and improved numerical stability.

```python
def _adapt_covariance_variant_01(self):
    """Diagonal-only (sep-CMA) covariance adaptation with per-dimension learning rates."""
    # Initialize diagonal covariance if needed
    if not hasattr(self, 'diag_C') or self.diag_C is None:
        self.diag_C = np.ones(self.dim)
    
    # Compute mean shift in normalized space
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Per-dimension rank-1 update using evolution path
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Separate learning rates for rank-1 and rank-mu updates
    cc_1 = 1.0 / (self.dim + 2.0)
    cc_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    
    # Rank-1 update for diagonal (squared values)
    rank_one_contrib = self.pc ** 2
    
    # Rank-mu update for diagonal
    rank_mu_contrib = np.zeros(self.dim)
    total_weight = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu_contrib += self.weights[i] * (diff ** 2)
        total_weight += abs(self.weights[i])
    
    if total_weight > 0:
        rank_mu_contrib /= total_weight
    
    # Adaptive learning rate based on dimensionality
    dim_adapt = np.log(self.dim + 1.0) / np.log(100.0)
    cc_1_adapt = cc_1 * (0.5 + 0.5 * dim_adapt)
    cc_mu_adapt = cc_mu * (1.5 - 0.5 * dim_adapt)
    
    # Clip learning rates for stability
    cc_1_adapt = np.clip(cc_1_adapt, 1e-10, 0.5)
    cc_mu_adapt = np.clip(cc_mu_adapt, 1e-10, 0.5)
    
    # Update diagonal covariance
    self.diag_C = (1.0 - cc_1_adapt - cc_mu_adapt) * self.diag_C
    self.diag_C += cc_1_adapt * rank_one_contrib
    self.diag_C += cc_mu_adapt * rank_mu_contrib
    
    # Ensure positive diagonal with minimum variance
    min_var = 1e-10 * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    self.diag_C = np.maximum(self.diag_C, min_var)
    
    # Build full covariance matrix from diagonal
    self.C = np.diag(self.diag_C)
    
    # Update Cholesky factor for sampling (diagonal case)
    self.L = np.diag(np.sqrt(self.diag_C))
```