**Idea: Eigendirection Weighted Mutation**

Use eigendecomposition of the population covariance matrix to identify principal axes, then weight mutation contributions by eigenvalue magnitude multiplied by fitness correlation along each axis. This adapts mutation scale per eigendirection based on the population's spectral structure and its relationship to fitness, addressing anisotropy and effective dimensionality in the search landscape (Category B: Spectral/linear-algebraic, targeting task 16).

```python
def _mutate_compass_batch(self, population, fitness):
    """Spectral mutation: eigendecomposition-weighted directional mutation."""
    NP, dim = population.shape
    
    # Compute population covariance matrix
    centroid = population.mean(axis=0)
    centered = population - centroid
    cov = (centered.T @ centered) / max(NP - 1, 1)
    
    # Eigendecomposition to get principal axes
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return population + np.random.randn(NP, dim) * 0.5
    
    # Sort by descending eigenvalue (descending variance)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project individuals onto principal axes
    projections = centered @ eigenvectors
    
    # Compute fitness correlation per principal axis (Pearson r with fitness)
    fitness_cent = fitness - fitness.mean()
    correlations = np.zeros(dim)
    for d in range(dim):
        proj_cent = projections[:, d] - projections[:, d].mean()
        denom = np.sqrt(proj_cent.var() + 1e-10) * np.sqrt(fitness_cent.var() + 1e-10)
        if denom > 1e-10:
            correlations[d] = np.dot(proj_cent, fitness_cent) / denom
        else:
            correlations[d] = 0.0
    
    # Weight each eigendirection by eigenvalue magnitude * |fitness correlation|
    # Stronger variance axes with clearer fitness alignment get higher weight
    weights = np.abs(eigenvalues) * np.abs(correlations)
    weights = np.maximum(weights, 1e-10)
    weights /= (weights.max() + 1e-10)
    
    # Generate spectral mutation: weighted combination of eigendirections
    # Each individual gets mutation proportional to its projection on weighted axes
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Compute weighted mutation along principal axes
        weighted_proj = projections[i] * weights
        # Transform back to original space
        mutation = eigenvectors @ weighted_proj
        
        # Add small random component scaled by average eigenvalue
        avg_eig = np.mean(np.abs(eigenvalues))
        mutation += np.random.randn(dim) * avg_eig * 0.1 * self.F
        
        mutants[i] = population[i] + mutation
    
    return np.clip(mutants, -100.0, 100.0)
```