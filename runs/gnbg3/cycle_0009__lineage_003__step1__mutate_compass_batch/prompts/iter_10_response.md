**Idea: Eigendecomposition-Guided Compass Mutation**

Uses PCA/eigendecomposition of the population covariance matrix to identify principal variance axes, then scales mutation components inversely to eigenvalues to preferentially explore less-varied (potentially underexplored) directions.

```python
def _mutate_compass_batch(self, population, fitness):
    """Spectral compass mutation via eigendecomposition of population covariance."""
    NP, dim = self.NP, self.dim
    mutants = np.empty_like(population)
    
    # Compute population covariance matrix (B: spectral analysis)
    centered = population - population.mean(axis=0)
    cov = np.cov(centered.T)
    
    # Eigendecomposition: eigenvectors = principal axes, eigenvalues = variance
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        # Fallback: use random perturbations
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            r = others[np.random.choice(len(others), 3, replace=False)]
            mutants[i] = population[i] + self.F * (population[r[0]] - population[r[1]])
        return mutants
    
    # Sort eigenvalues ascending (smallest = least variance = underexplored)
    idx = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[idx], 1e-12)
    eigenvectors = eigenvectors[:, idx]
    
    # Compute effective dimensionality (B: effective dimensionality signal)
    eigenvalue_sum = eigenvalues.sum()
    explained_ratio = np.cumsum(eigenvalues) / eigenvalue_sum
    n_essential = np.searchsorted(explained_ratio, 0.95) + 1
    n_essential = min(n_essential, dim)
    
    # Condition number for anisotropy scaling (B: anisotropy metric)
    cond = eigenvalues[-1] / eigenvalues[0]
    if cond > 1e6:
        # Highly anisotropic: use only essential dimensions
        use_dim = n_essential
    else:
        use_dim = min(dim, max(3, NP // 2))
    
    # Spectral scaling: inverse eigenvalue weighting (B: subspace alignment)
    # Smaller eigenvalue = less variance = underexplored = more mutation potential
    spectral_weights = 1.0 / (eigenvalues[:use_dim] + 1e-10)
    spectral_weights = spectral_weights / spectral_weights.sum()
    
    # Generate spectral mutants for each individual
    for i in range(NP):
        # Sample random coefficients weighted by inverse eigenvalues
        z = np.random.randn(use_dim)
        z = z * spectral_weights * np.sqrt(eigenvalues[:use_dim])
        
        # Project onto principal components
        perturbation = np.dot(eigenvectors[:, :use_dim], z)
        
        # Adaptive scale based on condition number (B: condition number adaptation)
        scale = self.F * (1.0 + np.log1p(cond) * 0.05)
        
        mutant = population[i] + scale * perturbation
        mutants[i] = np.clip(mutant, -100.0, 100.0)
    
    return mutants
```