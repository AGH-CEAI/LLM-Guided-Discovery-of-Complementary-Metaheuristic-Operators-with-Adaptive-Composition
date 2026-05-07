**Idea: Pairwise Distance Diversity**
A fundamentally different strategy using average pairwise Euclidean distance between population members, normalized by the expected distance for a uniform distribution. This is simpler, more intuitive, and doesn't rely on covariance matrix structure.

```python
def _compute_diversity(self):
    """Compute diversity using average pairwise Euclidean distance.
    
    Measures the average Euclidean distance between all pairs of individuals
    in the population. Normalized by expected distance for a uniformly
    distributed population in a hypercube: E[d] = range * sqrt(dim/3).
    
    - High values (~1.0): Population well-spread (good diversity)
    - Low values (~0.0): Population clustered (poor diversity)
    """
    n = len(self.population)
    if n < 2:
        return 1e-15
    
    # Compute all pairwise squared Euclidean distances efficiently
    # diffs[i,j,:] = population[i,:] - population[j,:]
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    
    # Average over all off-diagonal pairs (exclude self-distances)
    n_float = float(n)
    total_pairs = n_float * (n_float - 1.0)
    avg_sq_dist = (np.sum(sq_dists) - np.trace(sq_dists)) / max(total_pairs, 1.0)
    
    # Convert to Euclidean distance
    avg_dist = np.sqrt(max(avg_sq_dist, 1e-15))
    
    # Expected average pairwise distance for uniform distribution in hypercube
    # For uniform in [0,1]^d: E[||x-y||^2] = d/3, so E[||x-y||] = sqrt(d/3)
    bound_range = self.ub[0] - self.lb[0]
    expected_dist = bound_range * np.sqrt(self.dim / 3.0)
    expected_dist = max(expected_dist, 1e-10)
    
    # Normalized diversity metric
    diversity = avg_dist / expected_dist
    
    return float(np.clip(diversity, 1e-15, None))
```