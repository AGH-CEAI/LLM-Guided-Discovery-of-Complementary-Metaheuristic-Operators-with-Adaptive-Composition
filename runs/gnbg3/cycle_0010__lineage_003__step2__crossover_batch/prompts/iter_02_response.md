**Idea: SVD-Weighted Spectral Crossover**

Crossover rate per dimension is modulated by singular value magnitudes from SVD of the population matrix. Dimensions with larger singular values (more population variance) receive higher crossover probability, aligning genetic exchange with principal axes of variation (category B: spectral/linear-algebraic).
```python
def _crossover_batch(self, population, mutants):
    """Binomial crossover with spectral-weighted per-dimension CR using SVD."""
    NP, dim = population.shape
    trials = np.empty_like(population)
    
    # Compute SVD of population matrix to capture variance structure
    # Use economy SVD for efficiency
    try:
        # Population matrix: each row is an individual
        U, s, Vt = np.linalg.svd(population, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback on singular matrix: use uniform crossover
        mask = np.random.rand(NP, dim) < self.CR
        trials = np.where(mask, mutants, population)
        return trials
    
    # Compute effective dimensionality: count singular values > threshold
    # This tells us how many principal directions contain meaningful variance
    threshold = s[0] * 1e-6 if s[0] > 0 else 1e-12
    effective_dim = np.sum(s > threshold)
    effective_dim = max(1, min(effective_dim, dim))
    
    # Base CR: use instance attribute, default 0.85
    base_cr = getattr(self, 'CR', 0.85)
    
    # Compute spectral weights: higher singular values -> higher crossover probability
    # Weight = normalized singular value (emphasizes principal components)
    s_normalized = s / (s.sum() + 1e-12)  # Normalized, shape (min(NP, dim),)
    
    # For each dimension, compute probability of crossover
    # Dimensions aligned with principal components get higher probability
    if dim <= len(s_normalized):
        spectral_weights = s_normalized[:dim]
    else:
        # More dimensions than singular values: pad with small values
        spectral_weights = np.zeros(dim)
        spectral_weights[:len(s_normalized)] = s_normalized
        spectral_weights[len(s_normalized):] = s_normalized[-1] if len(s_normalized) > 0 else 0.0
    
    # Normalize weights to sum to dim (so average weight = 1)
    if spectral_weights.sum() > 0:
        spectral_weights = spectral_weights * dim / spectral_weights.sum()
    
    # Compute per-dimension crossover probability
    # Blend base CR with spectral weight: higher weight -> higher probability
    # This aligns crossover with directions of population variance
    cr_per_dim = np.clip(base_cr * (0.5 + 0.5 * spectral_weights), 0.0, 0.99)
    
    # Generate crossover mask: each element independently crossed with probability cr_per_dim
    # Add small per-individual noise to cr_per_dim for diversity
    cr_noise = np.random.uniform(-0.05, 0.05, dim)
    cr_per_dim_noisy = np.clip(cr_per_dim + cr_noise, 0.0, 0.99)
    
    # Create mask: shape (NP, dim), each row is same mask (same CR per dim for all individuals)
    mask = np.random.rand(NP, dim) < cr_per_dim_noisy
    
    # Ensure at least one dimension is crossed for each individual (standard DE practice)
    force_cross = np.random.randint(0, dim, size=NP)
    force_mask = np.zeros((NP, dim), dtype=bool)
    force_mask[np.arange(NP), force_cross] = True
    mask = mask | force_mask
    
    # Apply crossover: trial = population where mask is False, mutant where mask is True
    trials = np.where(mask, mutants, population)
    
    return trials
```