**Idea: k-NN Geometric Centroid Mutation**

Use k-nearest-neighbor relationships to compute local geometric centroids and derive mutation directions purely from spatial structure, without fitness-based weighting.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
    """
    Generate mutation vectors using k-NN geometric structure.
    For each individual, finds k nearest neighbors and moves toward their centroid.
    Purely geometric: uses pairwise distances and spatial relationships only.
    """
    np_pop, dim = population.shape
    trials = np.zeros((np_pop, dim))
    
    # Compute all pairwise squared distances (vectorized for efficiency)
    # Shape: (np_pop, np_pop)
    sq_dists = np.sum(population ** 2, axis=1, keepdims=True) + \
               np.sum(population ** 2, axis=1, keepdims=True).T - \
               2 * np.dot(population, population.T)
    np.fill_diagonal(sq_dists, np.inf)  # Exclude self-distance
    
    # k-NN structure: find k nearest neighbors for each individual
    k = min(5, np_pop - 1)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute k-NN centroid for each individual
    # Gather neighbor coordinates: shape (np_pop, k, dim)
    knn_coords = population[knn_indices]
    knn_centroid = np.mean(knn_coords, axis=1)  # Shape: (np_pop, dim)
    
    # Compute distance from each individual to its k-NN centroid
    dist_to_centroid = np.linalg.norm(population - knn_centroid, axis=1, keepdims=True)
    
    # Compute average distance between all individuals and their k-NN centroids
    avg_dist = np.mean(dist_to_centroid) + 1e-10
    
    # Direction vector toward k-NN centroid
    direction = knn_centroid - population  # Shape: (np_pop, dim)
    
    # Normalize direction and weight by relative position
    dir_norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-10
    direction_normalized = direction / dir_norm
    
    # Weight: individuals far from centroid get stronger pull, those near get exploration boost
    distance_ratio = dist_to_centroid / avg_dist
    weight = np.clip(distance_ratio, 0.3, 2.0)
    
    # Adaptive F with perturbation
    current_F = self.F * (1.0 + 0.1 * np.random.randn())
    current_F = np.clip(current_F, 0.1, 2.0)
    
    # Compute axis-aligned spread for perturbation diversity
    axis_spread = np.std(population, axis=0) + 1e-10
    
    # Geometric mutation: move toward k-NN centroid with distance-weighted strength
    perturbation = 0.1 * current_F * axis_spread * np.random.randn(np_pop, dim)
    trials = population + weight * direction_normalized * current_F * avg_dist + perturbation
    
    return trials, current_F
```