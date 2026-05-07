**Idea: Spectral Eigenvalue Collapse Detection**
Uses covariance matrix eigendecomposition to detect when population has collapsed into a low-dimensional subspace (high condition number, low effective rank). Restart generates new population along principal eigendirections to maintain structure while restoring diversity.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on spectral properties of population covariance matrix.
    Detects population collapse via condition number and effective rank of covariance."""
    
    # Compute population covariance matrix
    centroid = population.mean(axis=0)
    centered = population - centroid
    cov = np.cov(centered.T)
    
    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
    except np.linalg.LinAlgError:
        # Fallback: reinitialize if eigendecomposition fails
        best_idx = np.nanargmin(fitness)
        return self._initialize_population(), fitness[best_idx], population[best_idx].copy()
    
    # Filter positive eigenvalues for numerical stability
    pos_mask = eigenvalues > 1e-10
    if pos_mask.sum() < 2:
        # Nearly rank-1 or rank-0 covariance - severe collapse
        best_idx = np.nanargmin(fitness)
        new_pop = self._initialize_population()
        return new_pop, fitness[best_idx], population[best_idx].copy()
    
    pos_eigenvals = eigenvalues[pos_mask]
    
    # Condition number: ratio of largest to smallest positive eigenvalue
    cond_num = pos_eigenvals[0] / pos_eigenvals[-1]
    
    # Effective dimensionality: proportion of variance explained by principal components
    # Using participation ratio: sum(eigenvalues^2) / sum(eigenvalues)^2
    eigenvals_sq_sum = np.sum(pos_eigenvals ** 2)
    eigenvals_sum_sq = np.sum(pos_eigenvals) ** 2
    effective_dim = eigenvals_sq_sum / (eigenvals_sum_sq + 1e-10)
    
    # Threshold-based restart decision
    cond_threshold = 1000.0
    effective_dim_threshold = 0.3
    
    if cond_num > cond_threshold or effective_dim < effective_dim_threshold:
        # Population has collapsed - spectral restart needed
        best_idx = np.nanargmin(fitness)
        best_fit = fitness[best_idx]
        best_sol = population[best_idx].copy()
        
        # Sort eigenvectors by eigenvalue (descending)
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigvec_sorted = eigenvectors[:, sort_idx]
        eigval_sorted = eigenvalues[sort_idx]
        
        # Generate new population along eigendirections
        # Preserves covariance structure but adds diversity proportional to eigenvalue magnitudes
        new_pop = np.empty((self.NP, self.dim))
        new_pop[0] = best_sol.copy()
        
        # Compute scaling factors from eigenvalues
        max_eigenval = eigval_sorted[0]
        scales = np.sqrt(eigval_sorted / (max_eigenval + 1e-10))
        
        for i in range(1, self.NP):
            # Random linear combination of eigenvectors weighted by eigenvalue scales
            direction = np.zeros(self.dim)
            for d in range(min(self.dim, pos_mask.sum())):
                scale = scales[d] * (1.0 + 0.5 * np.random.rand())
                direction += np.random.randn() * eigvec_sorted[:, d] * scale
            new_pop[i] = best_sol + direction
        
        new_pop = np.clip(new_pop, -100.0, 100.0)
        return new_pop, best_fit, best_sol
    
    # No restart needed
    return (None, None, None)
```