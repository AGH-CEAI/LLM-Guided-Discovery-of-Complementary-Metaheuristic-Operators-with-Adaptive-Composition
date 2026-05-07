**Idea: Eigendecomposition-Based Anisotropic Expansion**

For the worst unsolved tasks (17, 16, 11, 23, 18, 20, 19) with errors in the 10-600 range, the Cholesky-based variant_01 actually performs *worse* than the original—likely because it lacks mechanisms to escape deceptive local optima. This variant will use eigendecomposition to detect when the covariance matrix becomes too narrow (high condition number), then aggressively expand along principal axes to escape local basins. This is fundamentally different from the current approach which maintains a fixed Cholesky structure without any stagnation-triggered exploration boost.

```python
def _adapt_covariance_variant_01(self):
    """Eigendecomposition-based anisotropic expansion for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    total_weight = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_weight += abs(w)
    
    if total_weight > 0:
        rank_mu /= total_weight
    
    ccov_1 = 1.0 / (self.dim + 2.0)
    ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
              ccov_1 * rank_one + 
              ccov_mu * rank_mu * np.sum(self.weights[:self.mu]))
    
    # Detect narrow covariance (high condition number) indicating potential local optima trap
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)
        cond = np.max(eigvals) / np.min(eigvals)
        
        # Threshold for "too narrow" covariance - tune based on problem scale
        narrow_threshold = 1e5
        
        if cond > narrow_threshold:
            # Anisotropic expansion: scale up smallest eigenvalues significantly
            min_eig = np.min(eigvals)
            expansion_factor = 10.0  # Strong expansion to escape local basin
            
            # Create expansion vector: scale smallest eigvals more
            expansion = np.ones_like(eigvals)
            expansion[eigvals < np.median(eigvals)] = expansion_factor
            
            new_eigvals = eigvals * expansion
            new_eigvals = np.maximum(new_eigvals, min_eig * 0.1)
            
            # Reconstruct covariance with expanded eigenvalues
            self.C = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
            self.C = 0.5 * (self.C + self.C.T)
            
        # Also trigger expansion if population diversity is very low
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = pop_variance / (expected_var + 1e-10)
        
        if diversity_ratio < 1e-4:
            # Broad isotropic expansion when diversity collapses
            min_eig = np.min(eigvals)
            self.C += 0.5 * min_eig * np.eye(self.dim)
            
    except np.linalg.LinAlgError:
        pass
    
    self.C = self._ensure_positive_definite(self.C)
```