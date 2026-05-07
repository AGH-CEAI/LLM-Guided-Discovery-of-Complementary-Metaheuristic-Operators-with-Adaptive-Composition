**Idea: Orthogonal-Escape Evolution Path**
When stagnation is detected, inject a secondary evolution path perpendicular to the main search direction, forcing the covariance to explore orthogonal subspaces and escape local optima on multimodal functions.

```python
def _adapt_covariance_variant_01(self):
    """Orthogonal-escape evolution path for multimodal function escape."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute perpendicular direction to pc for escape
    if not hasattr(self, 'p_escape'):
        self.p_escape = None
    
    # Trigger escape mechanism at 50% of stagnation threshold
    if self.stagnation_counter >= self.max_stagnation // 2:
        if self.p_escape is None and np.dot(self.pc, self.pc) > 1e-20:
            # Compute perpendicular direction using QR decomposition
            v = self.pc / (np.linalg.norm(self.pc) + 1e-20)
            random_vec = np.random.randn(self.dim)
            random_vec -= np.dot(random_vec, v) * v
            norm_r = np.linalg.norm(random_vec)
            if norm_r > 1e-10:
                self.p_escape = random_vec / norm_r
    
    # Apply main evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Rank-one and rank-mu updates
    rank_one = np.outer(self.pc, self.pc)
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine main and escape updates
    if self.p_escape is not None:
        rank_escape = np.outer(self.p_escape, self.p_escape)
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one + 
                  0.5 * self.ccov * rank_escape +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        # Decay escape path
        self.p_escape *= 0.95
    else:
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one + 
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```