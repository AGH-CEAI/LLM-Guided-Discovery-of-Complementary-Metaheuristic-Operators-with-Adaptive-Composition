**Idea: Effective Dimensionality Collapse Detection with Axis-Aligned Reinitialization**

A geometric restart strategy that detects when the population collapses into a low-dimensional subspace by analyzing the effective rank of the population matrix via singular value analysis on centered data. If effective dimensionality falls below a threshold, trigger a structured restart that reinitializes the population along the principal axes to maximize geometric spread while preserving elite solutions.

```python
def _restart_if_needed(self, population, fitness):
    """Restart triggered by geometric collapse detection."""
    NP, dim = population.shape
    
    # Compute centroid and center the population
    centroid = population.mean(axis=0)
    centered = population - centroid
    
    # Compute singular values to assess effective dimensionality (geometric property)
    # Using SVD on centered data measures how spread out the population is in different directions
    try:
        # np.linalg.svd is numerically stable
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        s = np.ones(dim)
    
    # Total variance captured by each singular value
    total_var = np.sum(s**2)
    if total_var < 1e-20:
        effective_dim = 1.0
    else:
        cumvar = np.cumsum(s**2) / total_var
        # Effective dimensionality: number of components needed for 95% variance
        effective_dim = np.searchsorted(cumvar, 0.95) + 1
    
    # Geometric spread metric: average pairwise distance normalized by expected range
    pairwise_dists = np.linalg.norm(centered[:, np.newaxis, :] - centered[np.newaxis, :, :], axis=2)
    np.fill_diagonal(pairwise_dists, np.nan)
    mean_spread = np.nanmean(pairwise_dists)
    
    # Axis-aligned spread: range in each dimension
    axis_spread = population.max(axis=0) - population.min(axis=0)
    min_axis_spread = np.min(axis_spread)
    
    # Decision to restart: effective dim too low OR population collapsed in any axis
    collapse_threshold = max(2, dim * 0.15)
    spread_threshold = 5.0  # Geometric spread too small
    
    should_restart = effective_dim < collapse_threshold or mean_spread < spread_threshold or min_axis_spread < 1e-4
    
    if not should_restart:
        return (None, None, None)
    
    # Structured reinitialization along principal axes
    # Find top-k principal directions
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_axes = Vt[:int(np.ceil(effective_dim))]
    except np.linalg.LinAlgError:
        principal_axes = np.eye(dim)[:1]
    
    # Keep top 20% elite solutions
    n_elite = max(1, int(0.2 * NP))
    elite_indices = np.argsort(fitness)[:n_elite]
    elite_solutions = population[elite_indices].copy()
    
    # Reinitialize rest along principal axes with controlled spread
    new_pop = np.empty((NP, dim))
    
    # Place elite solutions
    for i, elite_idx in enumerate(elite_indices):
        new_pop[i] = elite_solutions[i]
    
    # Reinitialize remaining individuals
    n_new = NP - n_elite
    
    # Compute current spread to guide new sampling
    current_range = np.maximum(axis_spread, 1.0)
    
    for i in range(n_new):
        idx = n_elite + i
        # Start from centroid
        new_point = centroid.copy()
        
        # Add perturbation along principal axes (geometrically informed)
        for axis in principal_axes:
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-10:
                axis = axis / axis_norm
                # Scale perturbation by current spread in that direction
                proj_range = np.abs(np.dot(axis, current_range))
                perturbation = np.random.randn() * max(proj_range * 0.3, 0.5)
                new_point += perturbation * axis
        
        # Fill remaining dimensions with random perturbation
        remaining_dims = dim - len(principal_axes)
        if remaining_dims > 0:
            new_point += np.random.randn(remaining_dims) * 2.0
        
        # Clip to bounds
        new_pop[idx] = np.clip(new_point, -100.0, 100.0)
    
    # Reset stagnation tracking
    self.stagnation_counter = 0
    
    # Compute new best fitness
    new_fitness = fitness.copy()
    new_best_idx = np.argmin(new_fitness)
    new_best_fitness = new_fitness[new_best_idx]
    new_best_x = new_pop[new_best_idx].copy()
    
    return (new_pop, new_best_fitness, new_best_x)
```