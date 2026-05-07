**Idea: k-NN Directed Spatial Mutation**
Category A: Geometry / spatial — Use k-nearest-neighbor structure to direct mutations toward locally best regions, with step size modulated by local spatial density.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
    np_pop, dim = population.shape
    trials = np.zeros((np_pop, dim))
    
    k = min(5, np_pop - 1)
    
    # Compute pairwise squared distances (vectorized)
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Find k nearest neighbors for each individual
    knn_indices = np.zeros((np_pop, k), dtype=int)
    for i in range(np_pop):
        knn_indices[i] = np.argpartition(dists_sq[i], k - 1)[:k]
    
    # Compute k-NN centroid weighted by inverse fitness
    knn_centroids = np.zeros((np_pop, dim))
    for i in range(np_pop):
        neighbor_fitness = fitness[knn_indices[i]]
        inv_fit = self.upper - neighbor_fitness + 1.0
        weights = inv_fit / (np.sum(inv_fit) + 1e-30)
        knn_centroids[i] = np.sum(population[knn_indices[i]] * weights[:, np.newaxis], axis=0)
    
    # Compute local spatial density (mean dist to k-NN)
    local_density = np.zeros(np_pop)
    for i in range(np_pop):
        local_density[i] = np.mean(np.sqrt(dists_sq[i, knn_indices[i]]))
    
    # Diversity weight: sparse regions get larger steps
    min_d, max_d = np.min(local_density), np.max(local_density)
    range_d = max_d - min_d + 1e-30
    diversity_weight = 0.5 + 1.5 * (max_d - local_density) / range_d
    diversity_weight = np.clip(diversity_weight, 0.3, 2.0)
    
    # k-NN directed mutation with density scaling
    current_F = self.F * (1.0 + 0.1 * np.random.randn())
    current_F = np.clip(current_F, 0.1, 2.0)
    
    knn_drift = knn_centroids - population
    trials = population + current_F * knn_drift * diversity_weight[:, np.newaxis]
    
    # Add small geometric perturbation for exploration
    trials += 0.05 * current_F * np.random.randn(np_pop, dim)
    
    return trials, current_F
```