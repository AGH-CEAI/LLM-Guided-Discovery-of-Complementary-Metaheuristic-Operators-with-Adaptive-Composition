**Idea: Multi-Evolution-Path Covariance with Active Damping**

The worst unsolved tasks (17, 16, 6) have errors in the 10^2-10^3 range, indicating the current single evolution path (pc) and rank-mu update aren't capturing the full problem geometry. The key insight is that CMA-ES's covariance adaptation relies on cumulating information about successful search directions, but with only one path, it can miss important correlations—especially on ill-conditioned or multi-modal landscapes. This variant introduces **dual active evolution paths**: one tracking step-size success (as before) and a new **weighted success path** that exponentially weights recent successful steps with higher weight, plus an **inter-step correlation path** that captures how consecutive steps interact. This gives the covariance matrix richer directional information to navigate the worst-case landscapes.

```python
def _adapt_covariance(self):
    """
    Adapt covariance using dual active evolution paths with exponential history weighting.
    - pc_weighted: tracks exponentially-weighted recent mean shifts (better for rugged landscapes)
    - p_cross: captures inter-step correlations (helps with non-separable rotation)
    """
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Primary evolution path (standard)
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Secondary weighted evolution path: exponentially weighted recent steps
    # Gives more weight to recent successful directions (alpha = 0.05)
    alpha = 0.05
    if not hasattr(self, 'pc_weighted'):
        self.pc_weighted = np.zeros(self.dim)
    self.pc_weighted = (1.0 - alpha) * self.pc_weighted + alpha * y_mean
    
    # Tertiary path: inter-step correlation (captures rotation in search)
    # Helps on non-separable functions where step direction changes
    beta = 0.02
    if not hasattr(self, 'p_cross'):
        self.p_cross = np.zeros(self.dim)
    if hasattr(self, 'prev_y_mean'):
        cross_contribution = np.sqrt(beta) * (y_mean - self.prev_y_mean)
        self.p_cross = (1.0 - beta) * self.p_cross + cross_contribution
    self.prev_y_mean = y_mean.copy()
    
    # Rank-1 updates from all three paths
    rank_one_primary = np.outer(self.pc, self.pc)
    rank_one_weighted = np.outer(self.pc_weighted, self.pc_weighted)
    rank_one_cross = np.outer(self.p_cross, self.p_cross)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine with different learning rates
    # Primary path: standard rate
    # Weighted path: 0.3 * ccov (slower, supplementary)
    # Cross path: 0.1 * ccov (minor correction for rotation)
    ccov_primary = self.ccov
    ccov_weighted = 0.3 * self.ccov
    ccov_cross = 0.1 * self.ccov
    
    self.C = ((1.0 - ccov_primary) * self.C + 
              ccov_primary * rank_one_primary +
              ccov_weighted * rank_one_weighted +
              ccov_cross * rank_one_cross +
              (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
    
    # Ensure positive definiteness with tighter bound
    self.C = 0.5 * (self.C + self.C.T)
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-12:
        self.C += (1e-9 - min_eig) * np.eye(self.dim)
    
    # Clip condition number for robustness
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / np.min(eigvals)
    if cond > 1e7:
        # Dampen extreme eigenvalues
        self.C *= 0.5
```