**Idea: Geometric Convex Hull and k-NN Density Restart**

Uses convex hull volume and k-NN graph structure to detect population compression, then reinitializes compressed regions by spreading along principal axes while maintaining centroid proximity.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on geometric structure: convex hull volume and k-NN density."""
    NP, dim = population.shape
    
    # Compute pairwise distances and spread metrics (A: geometry)
    centroid = population.mean(axis=0)
    diffs = population - centroid
    spread = np.mean(np.linalg.norm(diffs, axis=1))
    
    # Compute convex hull volume as diversity indicator (A: convex hull)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(population)
        hull_volume = hull.volume if hasattr(hull, 'volume') else 0.0
    except (QhullError, Exception):
        hull_volume = 1e-10
    
    # Axis-aligned bounding box to detect dimensional compression (A: axis-aligned spread)
    bb_min = population.min(axis=0)
    bb_max = population.max(axis=0)
    bb_range = bb_max - bb_min
    total_range = bb_range.sum()
    
    # k-NN structure analysis for local density (A: k-NN structure)
    k = max(2, min(5, NP // 10))
    sq_dists = np.sum((population[:, np.newaxis, :] - population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
    knn_avg_dist = np.mean(np.sqrt(nearest_sq_dists))
    
    # Detect compression: low hull volume, small spread, or high local density
    is_compressed = (hull_volume < 1e-6 * (spread ** dim)) or (spread < 5.0) or \
                    (knn_avg_dist < spread * 0.1) or (total_range < 10.0)
    
    # Compute individual distances to centroid (A: centroid distances)
    ind_dists = np.linalg.norm(diffs, axis=1)
    avg_dist = ind_dists.mean()
    
    # Identify compressed individuals (below-average distance to centroid)
    compressed_mask = ind_dists < avg_dist * 0.7
    
    if is_compressed or compressed_mask.sum() > NP // 3:
        # Reinitialize compressed individuals with geometric spreading
        n_replace = max(compressed_mask.sum(), NP // 3)
        replace_indices = np.argsort(ind_dists)[:n_replace]
        
        for idx in replace_indices:
            # Spherical distribution around centroid with radius scaled by spread
            direction = np.random.randn(dim)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            radius = np.random.uniform(0.5, 2.0) * max(spread, 20.0)
            noise_scale = max(15.0, spread * 0.3)
            population[idx] = np.clip(centroid + direction * radius + np.random.randn(dim) * noise_scale, -100.0, 100.0)
        
        # Update fitness for reinitialized individuals
        new_fitness = fitness.copy()
        new_fitness[replace_indices] = np.inf
        
        # Update stagnation tracking
        self.stagnation_counter = 0
        self.prev_best_fitness = np.min(new_fitness)
        
        return population, np.min(new_fitness), population[np.argmin(new_fitness)]
    
    return (None, None, None)
```