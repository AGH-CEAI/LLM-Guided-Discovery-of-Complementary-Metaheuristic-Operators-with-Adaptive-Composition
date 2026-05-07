Looking at this problem, I need to design a Category B (Spectral/linear-algebraic) mutation strategy that uses matrix decompositions and spectral properties of the population to guide mutation directions.

**Idea: SVD-Guided Principal Component Mutation**
Use singular value decomposition of the centered population matrix to extract principal directions, then construct mutations aligned with the population's major axes of variance. This leverages the effective dimensionality and singular value spectrum to weight mutation contributions.

```python
def _mutate_compass_batch(self, population, fitness):
    """SVD-guided mutation using population's spectral decomposition."""
    NP, dim = population.shape
    
    # Compute population covariance via SVD (B: spectral decomposition)
    centroid = population.mean(axis=0)
    centered = population - centroid
    
    # SVD captures both principal axes (V) and effective dimensionality via singular values (S)
    # This is fundamentally different from geometric distance-based approaches
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback for rank-deficient or near-singular populations
        return np.random.uniform(-100, 100, (NP, dim))
    
    # Compute effective rank from singular value spectrum (B: effective dimensionality)
    # Using entropy-based effective rank: exp(H) where H is normalized entropy of squared singular values
    eps = 1e-12
    s_squared = S**2
    s_norm = s_squared / (s_squared.sum() + eps)
    entropy = -np.sum(s_norm * np.log(s_norm + eps))
    effective_rank = int(np.round(np.exp(entropy)))
    effective_rank = max(1, min(effective_rank, dim))
    
    # Compute condition number for adaptive scaling (B: condition number)
    cond_number = S[0] / (S[-1] + eps) if S[-1] > eps else 1e6
    cond_clipped = min(cond_number, 1e4)
    
    # Adaptive scaling: reduce step size for ill-conditioned populations
    scale_factor = self.F / np.log1p(cond_clipped)
    
    # Rank singular values to identify major/minor axes (B: rank-based spectral reasoning)
    n_major = max(1, min(effective_rank, 3))
    n_minor = max(1, dim - effective_rank)
    
    # Normalize singular values for weighting
    S_normalized = S / (S[0] + eps)
    
    # Generate mutations per individual using spectral structure
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Select eigenvectors weighted by singular value magnitudes
        # Major axes (high variance): contribute scaled mutations
        major_indices = np.random.choice(
            min(effective_rank, len(S)), 
            size=min(n_major, effective_rank), 
            replace=False, 
            p=S_normalized[:effective_rank] / (S_normalized[:effective_rank].sum() + eps)
        )
        
        # Minor axes (low variance): small perturbation for diversity
        minor_indices = np.random.choice(
            max(effective_rank, 0), 
            max(0, min(n_minor, dim - effective_rank)), 
            replace=False
        ) if dim > effective_rank else np.array([], dtype=int)
        
        # Construct mutation vector from spectral components (B: subspace alignment)
        mutation = np.zeros(dim)
        
        # Add major direction contributions (weighted by singular values)
        for idx in major_indices:
            sign = np.random.choice([-1, 1])
            mutation += sign * scale_factor * S_normalized[idx] * Vt[idx]
        
        # Add minor direction perturbations (small, diversity-inducing)
        if len(minor_indices) > 0:
            for idx in minor_indices:
                sign = np.random.choice([-1, 1])
                mutation += sign * 0.1 * scale_factor * S_normalized[idx] * Vt[idx]
        
        # Apply mutation centered at current individual (maintain population structure)
        mutants[i] = population[i] + mutation
    
    return np.clip(mutants, -100.0, 100.0)
```