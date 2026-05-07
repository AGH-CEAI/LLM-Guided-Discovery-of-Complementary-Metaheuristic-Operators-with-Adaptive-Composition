**Idea: Eigen-Direction Modulated Mutation**

Category B (Spectral / linear-algebraic) — Use eigendecomposition of population covariance to modulate mutation strength per eigenvector direction. Directions with small eigenvalues (neglected subspace) receive amplified mutation to escape local optima; dominant directions receive reduced mutation for exploitation.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
    """
    Generate mutation vectors using eigendecomposition of population covariance.
    Modulate mutation strength along each eigenvector based on eigenvalue magnitude.
    """
    np_pop, dim = population.shape
    trials = np.zeros((np_pop, dim))
    
    # Compute population covariance matrix
    centroid = np.mean(population, axis=0)
    centered = population - centroid
    cov = (centered.T @ centered) / max(np_pop - 1, 1)
    
    # Regularize covariance for numerical stability
    cov_reg = cov + 1e-6 * np.eye(dim)
    
    # Eigendecomposition: cov = V @ diag(eigvals) @ V.T
    try:
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        # Sort eigenvalues ascending (smallest = most neglected direction)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    except np.linalg.LinAlgError:
        # Fallback to standard rand/1 if eigendecomposition fails
        r1 = ring_prev
        r2 = ring_next
        current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
        trials = population[r1] + current_F * (population[r2] - population)
        return trials, current_F
    
    # Compute condition number for anisotropy detection
    eigvals_safe = np.clip(eigvals, 1e-12, None)
    cond_num = np.max(eigvals_safe) / np.min(eigvals_safe)
    
    # Compute spectral weights: inverse of normalized eigenvalue
    # Small eigenvalues -> large weight (explore neglected subspace)
    eigvals_norm = eigvals_safe / (np.sum(eigvals_safe) + 1e-12)
    spectral_weights = 1.0 / (eigvals_norm + 0.01)
    spectral_weights = spectral_weights / np.sum(spectral_weights)  # Normalize
    
    # Anisotropy factor: when highly anisotropic, emphasize small-eigenvalue directions
    anisotropy_factor = np.log1p(cond_num) / np.log1p(dim * dim)
    anisotropy_factor = np.clip(anisotropy_factor, 0.0, 1.0)
    
    # Base mutation: ring-based rand/1
    r1 = ring_prev
    r2 = ring_next
    current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
    base_mutation = population[r1] + current_F * (population[r2] - population)
    
    # Project base mutation to eigenbasis
    base_mutation_centered = base_mutation - centroid
    proj_coeffs = base_mutation_centered @ eigvecs  # (np_pop, dim) in eigenbasis
    
    # Modulate mutation coefficients by spectral weights
    # High anisotropy -> stronger modulation toward small-eigenvalue directions
    modulation_strength = 0.3 + 0.4 * anisotropy_factor
    modulation = 1.0 + modulation_strength * (spectral_weights - np.mean(spectral_weights)) / (np.std(spectral_weights) + 1e-8)
    modulation = np.clip(modulation, 0.3, 2.5)
    
    # Apply modulation in eigenbasis
    modulated_coeffs = proj_coeffs * modulation
    
    # Transform back to original space
    modulated_mutation = modulated_coeffs @ eigvecs.T + centroid
    
    # Blend: modulated vs base based on anisotropy
    blend_weight = 0.3 + 0.4 * anisotropy_factor
    blend_weight = np.clip(blend_weight, 0.2, 0.7)
    trials = (1 - blend_weight) * base_mutation + blend_weight * modulated_mutation
    
    # Fitness-weighted directional component for additional exploitation
    weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
    weights = weights / np.sum(weights)
    centroid_weighted = np.sum(population * weights[:, np.newaxis], axis=0)
    
    directional_component = centroid_weighted - centroid
    trials = trials + 0.15 * directional_component
    
    return trials, current_F
```