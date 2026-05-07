**Idea: k-NN Graph Connectivity-Guided Mutation**

One-line description: Build a k-NN graph over the population, compute per-individual connectivity and local density, then use graph structure to select mutation parents from different clusters and bias mutations toward sparse (under-explored) regions.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
    np_pop, dim = population.shape
    trials = np.zeros((np_pop, dim))
    
    # Build k-NN graph based on Euclidean distance
    k = max(3, min(10, np_pop // 4))
    
    # Compute pairwise distances
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Find k-nearest neighbors
    knn_indices = np.argsort(dists_sq, axis=1)[:, :k]
    
    # Graph-based property 1: local connectivity (inverse mean distance to k-NN)
    local_density = np.zeros(np_pop)
    for i in range(np_pop):
        local_density[i] = np.mean(dists_sq[i, knn_indices[i]])
    local_density = np.clip(local_density, 1e-10, None)
    connectivity = 1.0 / local_density
    
    # Normalize connectivity to [0, 1]
    conn_min, conn_max = np.min(connectivity), np.max(connectivity)
    if conn_max > conn_min:
        conn_norm = (connectivity - conn_min) / (conn_max - conn_min)
    else:
        conn_norm = np.ones(np_pop) * 0.5
    
    # Graph-based property 2: find individuals in sparse regions (low connectivity)
    # These are under-explored — bias mutation toward global best
    sparse_mask = conn_norm < 0.3
    
    # Find global best and second-best for cross-cluster guidance
    best_idx = np.argmin(fitness)
    sorted_idx = np.argsort(fitness)
    second_best_idx = sorted_idx[1] if np_pop > 1 else best_idx
    
    # Build graph-based parent indices: select from different local neighborhoods
    # For each individual, find one k-NN parent and one "far" parent from different region
    graph_parent1 = np.array([knn_indices[i, np.random.randint(k)] for i in range(np_pop)])
    
    # Second parent: pick from k-NN of a distant individual (promotes cross-cluster mating)
    far_indices = np.argsort(dists_sq, axis=1)[:, -1]  # farthest neighbor
    graph_parent2 = far_indices[graph_parent1]  # far from first graph parent
    
    # Adaptive F
    current_F = self.F * (1.0 + 0.1 * np.random.randn())
    current_F = np.clip(current_F, 0.1, 2.0)
    
    # Graph-guided mutation for all individuals
    p1 = population[graph_parent1]
    p2 = population[graph_parent2]
    
    # Base graph-guided mutation: rand/1 using graph-derived parents
    graph_mutation = p1 + current_F * (p2 - population)
    
    # Sparse-region correction: additionally bias toward best for under-explored areas
    best_vector = population[best_idx]
    bias_strength = np.maximum(0, 0.3 - conn_norm) * 2.0  # 0 to 0.3 based on sparsity
    
    for i in range(np_pop):
        if sparse_mask[i]:
            # Under-explored: mix graph mutation with best-guided mutation
            trials[i] = (1 - bias_strength[i]) * graph_mutation[i] + \
                        bias_strength[i] * (population[i] + current_F * (best_vector - population[i]))
        else:
            trials[i] = graph_mutation[i]
    
    # Final clipping to bounds
    trials = np.clip(trials, self.lower, self.upper)
    
    return trials, current_F
```