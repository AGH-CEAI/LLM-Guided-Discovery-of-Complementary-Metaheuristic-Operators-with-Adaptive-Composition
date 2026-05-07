Looking at the problem, I need to design a geometrically-driven mutation for `_mutate_compass_batch` that operates on the spatial layout of the population. The key is to use distances, k-NN structure, and centroid relationships to guide mutations toward unexplored regions.

**Idea: Convex Hull Boundary + Centroid Pull + Local Repulsion**

This approach uses three geometric components per individual: (1) direction toward/away from population centroid weighted by distance from center, (2) local k-NN repulsion to escape dense clusters, and (3) boundary exploration toward convex hull extremes. All reasoning is purely spatial — no fitness signals or eigenvalues.

```python
def _mutate_compass_batch(self, population, fitness):
    """
    Geometric compass mutation using centroid, k-NN structure, and boundary analysis.
    Category A: Geometry / spatial - operates on pairwise distances, centroid relationships,
    and k-NN structure to generate spatially-guided mutations.
    """
    NP, dim = population.shape
    
    # Compute centroid of population
    centroid = population.mean(axis=0)
    
    # Compute distances to centroid for each individual
    centroid_diffs = population - centroid
    centroid_distances = np.linalg.norm(centroid_diffs, axis=1)
    
    # Compute axis-aligned spread (range per dimension)
    dim_min = population.min(axis=0)
    dim_max = population.max(axis=0)
    axis_spread = dim_max - dim_min
    axis_spread = np.maximum(axis_spread, 1e-10)
    
    # Normalize spread for axis weighting
    spread_weights = axis_spread / axis_spread.sum()
    
    # Compute pairwise distances (vectorized) for k-NN structure
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    k = max(2, min(5, NP // 5))
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute local density from k-NN distances
    knn_distances = np.sqrt(sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices])
    local_density = np.mean(knn_distances, axis=1)
    local_density = np.maximum(local_density, 1e-10)
    
    # Find boundary candidates (farthest from centroid)
    farthest_idx = np.argmax(centroid_distances)
    
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Direction 1: Centroid attraction/repulsion
        # Closer individuals get stronger pull toward centroid
        centroid_direction = centroid - population[i]
        centroid_dist = centroid_distances[i]
        centroid_direction = centroid_direction / (centroid_dist + 1e-10)
        centroid_weight = centroid_dist / (centroid_distances.mean() + 1e-10)
        
        # Direction 2: Local k-NN repulsion (escape dense clusters)
        nn_indices = nearest_indices[i]
        nn_points = population[nn_indices]
        local_center = nn_points.mean(axis=0)
        local_direction = population[i] - local_center
        local_dist = np.linalg.norm(local_direction)
        local_direction = local_direction / (local_dist + 1e-10)
        # Denser neighborhoods get stronger repulsion
        density_weight = 1.0 / (local_density[i] + 1e-10)
        density_weight = density_weight / (density_weight.mean() + 1e-10)
        
        # Direction 3: Boundary exploration (toward farthest hull point)
        boundary_direction = population[farthest_idx] - population[i]
        boundary_dist = np.linalg.norm(boundary_direction)
        boundary_direction = boundary_direction / (boundary_dist + 1e-10)
        # Individuals near centroid get stronger boundary push
        boundary_weight = 1.0 - (centroid_dist / (centroid_distances.max() + 1e-10))
        
        # Combine weighted directions
        mutation_direction = (
            0.5 * centroid_weight * centroid_direction +
            0.3 * density_weight * local_direction +
            0.2 * boundary_weight * boundary_direction
        )
        
        # Scale by axis spread (explore along wider dimensions more)
        mutation_direction = mutation_direction * (1.0 + spread_weights)
        
        # Apply mutation
        mutants[i] = population[i] + self.F * mutation_direction
    
    return np.clip(mutants, -100.0, 100.0)
```