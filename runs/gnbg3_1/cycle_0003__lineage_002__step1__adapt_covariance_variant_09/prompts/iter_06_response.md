**Idea: Eigenvalue Floor with Active Conditioning**

A fundamentally different approach that directly monitors and corrects the covariance matrix's eigenvalue distribution. The key insight is that tasks with errors ~10^1-10^2 are likely suffering from premature eigenvalue collapse (near-zero variance in some directions) causing the algorithm to get stuck in local optima. This variant enforces a minimum eigenvalue floor, actively conditions the matrix, and uses eigenvalue momentum to prevent re-collapse.

```python
def _adapt_covariance_variant_09(self):
    """Eigenvalue floor with active conditioning - prevents premature collapse."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Base covariance update
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # Enforce minimum eigenvalue floor to prevent collapse
    eigvals, eigvecs = np.linalg.eigh(self.C)
    min_eig = np.min(eigvals)
    eig_floor = 1e-6 * np.max(eigvals)
    
    if min_eig < eig_floor:
        # Raise small eigenvalues to floor
        eigvals[eigvals < eig_floor] = eig_floor
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Active condition number control
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    max_cond = 1e6
    
    if cond > max_cond:
        # Compress condition number by scaling extreme eigenvalues
        target_spread = max_cond
        min_target = np.min(eigvals)
        max_target = min_target * target_spread
        
        # Compress large eigenvalues
        mask_large = eigvals > max_target
        if np.any(mask_large):
            eigvals[mask_large] = max_target
        
        # Expand small eigenvalues if needed
        if np.min(eigvals) < min_target:
            eigvals[eigvals < min_target] = min_target
        
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    self.C = self._ensure_positive_definite(self.C)
```