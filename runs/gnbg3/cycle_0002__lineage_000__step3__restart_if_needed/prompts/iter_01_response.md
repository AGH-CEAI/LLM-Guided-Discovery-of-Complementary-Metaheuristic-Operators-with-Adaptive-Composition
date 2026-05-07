Looking at this problem, I need to implement `_restart_if_needed` using geometric/spatial reasoning about the population layout. The key insight is that when stagnation occurs, I should use the geometric structure of the population to guide reinitialization rather than purely random restart.

**Idea: Convex Hull Centroid Injection Restart**

Use convex hull geometry to identify extreme boundary points and cluster centroids, then reinitialize the population by preserving geometrically important points while filling gaps along principal directions. This maintains search space coverage based on literal spatial positions.

```python
def _restart_if_needed(self, population, fitness):
    """Restart using convex hull and geometric clustering."""
    NP, dim = population.shape
    
    # Compute population centroid (geometric center)
    centroid = population.mean(axis=0)
    
    # Compute pairwise distances to centroid (geometric measure)
    centroid_dists = np.linalg.norm(population - centroid, axis=1)
    
    # Compute all pairwise distances for geometric structure
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Identify extreme points using maximum distance from centroid
    n_extreme = max(2, NP // 4)
    extreme_mask = np.zeros(NP, dtype=bool)
    sorted_by_dist = np.argsort(centroid_dists)[::-1]
    extreme_mask[sorted_by_dist[:n_extreme]] = True
    extreme_indices = np.where(extreme_mask)[0]
    
    # Use k-NN structure for cluster identification
    k = max(3, min(8, NP // 5))
    nearest = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute local density (count of neighbors within median distance)
    median_dist = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
    local_density = np.sum(np.sqrt(sq_dists) < median_dist, axis=1)
    
    # Select cluster representatives: point with lowest local density in each region
    n_clusters = max(2, NP // 6)
    cluster_rep_indices = []
    remaining = np.ones(NP, dtype=bool)
    remaining[extreme_indices] = False
    
    for _ in range(n_clusters):
        if not remaining.any():
            break
        # Pick lowest density point from remaining
        densities = np.where(remaining, local_density, np.inf)
        rep_idx = np.argmin(densities)
        cluster_rep_indices.append(rep_idx)
        # Mark nearby points as "covered"
        neighbor_mask = np.isin(nearest[rep_idx], np.where(remaining)[0])
        nearby = nearest[rep_idx][neighbor_mask]
        remaining[nearby] = False
        remaining[rep_idx] = False
    
    # Build new population: extreme points + cluster reps + axis-filling
    new_pop = []
    selected = set(extreme_indices.tolist() + cluster_rep_indices)
    
    # Keep extreme points (geometrically important boundary)
    for idx in extreme_indices:
        new_pop.append(population[idx].copy())
    
    # Add cluster representatives (geometric centers)
    for idx in cluster_rep_indices:
        if len(new_pop) < NP:
            new_pop.append(population[idx].copy())
    
    # Fill remaining with points along principal axes from centroid
    axis_ranges = population.max(axis=0) - population.min(axis=0)
    axis_ranges = np.maximum(axis_ranges, 1e-6)
    
    while len(new_pop) < NP:
        t = np.random.rand(dim)
        # Interpolate from centroid toward random corner of the bounding box
        corner = population.min(axis=0) + t * axis_ranges
        point = centroid + 0.5 * (corner - centroid) + np.random.randn(dim) * np.sqrt(axis_ranges) * 0.1
        new_pop.append(np.clip(point, -100.0, 100.0))
    
    return np.array(new_pop[:NP]), centroid_dists.min(), extreme_indices
```