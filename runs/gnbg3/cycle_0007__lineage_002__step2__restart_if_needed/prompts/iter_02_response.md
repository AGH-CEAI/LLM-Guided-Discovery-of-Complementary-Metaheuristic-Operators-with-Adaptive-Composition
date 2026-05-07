**Idea: Eigenvalue-Collapse Restart with Anisotropy Correction**

A spectral restart that monitors population covariance eigenvalue decay. If the effective dimensionality collapses (few dominant eigenvalues, rest near zero) or condition number becomes extreme, reinitialize by expanding along the neglected eigendirections to restore full-dimensional search capacity.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on covariance eigenvalue spectrum analysis.
    
    Category B: Spectral/linear-algebraic
    Detects population collapse via eigenvalue decay and condition number.
    Reinitializes by expanding along low-variance eigendirections.
    """
    NP, dim = population.shape
    
    # Compute population covariance matrix
    centroid = population.mean(axis=0)
    centered = population - centroid
    cov = (centered.T @ centered) / max(NP - 1, 1)
    
    # Eigendecomposition of covariance
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except np.linalg.LinAlgError:
        # Fallback: random reinit
        return self._initialize_population(), np.full(NP, np.inf), np.zeros(dim)
    
    # Numerical stability: clip small negative eigenvalues (can occur due to precision)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    
    # Effective dimensionality: count eigenvalues above threshold
    total_variance = eigenvalues.sum()
    if total_variance < 1e-12:
        # Population is a point - full random reinit
        return self._initialize_population(), np.full(NP, np.inf), np.zeros(dim)
    
    # Relative variance contribution
    rel_var = eigenvalues / total_variance
    cumsum_var = np.cumsum(rel_var)
    
    # Effective rank using exponential entropy
    p = rel_var / rel_var.sum()  # normalize
    entropy = -np.sum(p * np.log(p + 1e-12))
    effective_rank = np.exp(entropy)
    
    # Condition number (ratio of largest to smallest eigenvalue)
    cond_number = eigenvalues[0] / max(eigenvalues[-1], 1e-12)
    
    # Check collapse criteria
    # 1) Effective rank too low (population in subspace)
    rank_collapsed = effective_rank < dim * 0.3
    
    # 2) Condition number too extreme
    cond_collapsed = cond_number > 1e6
    
    # 3) Most variance in few directions (anisotropy)
    first_two_explain = cumsum_var[min(1, len(cumsum_var)-1)]
    anisotropy = first_two_explain > 0.95
    
    # 4) Smallest eigenvalues nearly zero (flat directions)
    smallest_ratio = eigenvalues[-1] / max(eigenvalues[0], 1e-12)
    flat_directions = smallest_ratio < 1e-6
    
    should_restart = rank_collapsed or cond_collapsed or anisotropy or flat_directions
    
    if not should_restart:
        return (None, None, None)
    
    # Restart strategy: expand along neglected eigendirections
    new_population = np.empty_like(population)
    
    # Determine which eigendirections are underutilized
    small_var_mask = rel_var < 0.05  # directions with <5% variance
    n_directions_to_expand = max(dim // 3, 2)  # expand at least dim/3 directions
    
    if small_var_mask.sum() < n_directions_to_expand:
        # If not enough small directions, take the smallest ones
        expand_indices = np.argsort(rel_var)[:n_directions_to_expand]
    else:
        expand_indices = np.where(small_var_mask)[0][:n_directions_to_expand]
    
    # For each individual, expand along the neglected directions
    for i in range(NP):
        # Base: keep some current population structure
        if np.random.rand() < 0.3 and NP > 1:
            # Keep best individual's influence with probability
            best_idx = np.argmin(fitness)
            base = population[best_idx].copy()
        else:
            base = centroid.copy()
        
        # Add noise weighted by inverse of eigenvalue (more noise in low-var directions)
        noise = np.zeros(dim)
        for d in range(dim):
            if d in expand_indices:
                # Expand more in neglected directions
                scale = np.sqrt(eigenvalues[0] / max(eigenvalues[d], 1e-12)) * 5.0
            else:
                # Small perturbation in well-represented directions
                scale = np.sqrt(eigenvalues[d]) * 0.5
            noise[d] = np.random.randn() * scale
        
        new_population[i] = base + noise
    
    # Clip to bounds
    new_population = np.clip(new_population, -100.0, 100.0)
    
    # Update best solution tracking
    best_idx = np.argmin(fitness)
    x_opt = population[best_idx].copy()
    f_opt = fitness[best_idx]
    
    # Reset stagnation counter
    self.stagnation_counter = 0
    
    return new_population, f_opt, x_opt
```