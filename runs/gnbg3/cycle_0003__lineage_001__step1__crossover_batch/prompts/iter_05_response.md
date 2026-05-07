**Idea: k-NN Graph-Distance Adaptive Crossover**

Use a k-nearest-neighbor graph to compute graph-based distances between each mutant and population members, then modulate the per-dimension crossover probability based on topological proximity. Mutants that are well-connected (short graph distance) to their target in the population topology will contribute more dimensions, while isolated mutants contribute fewer.

```python
def _crossover_batch(self, population, mutants):
    """Graph-distance-adaptive binomial crossover using k-NN topology."""
    NP, dim = population.shape
    
    # Build k-NN graph over population
    k = max(2, min(5, NP // 4))
    
    # Compute pairwise squared distances (vectorized)
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute graph-based distance from each mutant to its target
    # Graph distance considers indirect paths through shared neighbors
    graph_dist = np.zeros(NP)
    
    for i in range(NP):
        target = population[i]
        mutant = mutants[i]
        
        # Direct Euclidean distance
        direct_dist = np.linalg.norm(mutant - target)
        
        # Graph-based distance: consider shared neighbors
        mutant_neighbors = set(knn_indices[i])
        
        if len(mutant_neighbors) == 0:
            graph_dist[i] = direct_dist
            continue
        
        # Check second-order connections (neighbors of neighbors)
        second_order = set()
        for neighbor in mutant_neighbors:
            second_order.update(knn_indices[neighbor])
        second_order.discard(i)
        second_order = second_order - mutant_neighbors
        
        # Graph distance approximation: direct + penalty for lack of graph connectivity
        if len(second_order) == 0:
            graph_penalty = 1.5
        else:
            # More second-order neighbors = better graph connectivity = lower distance
            connectivity_factor = len(second_order) / float(k * k)
            graph_penalty = 1.0 / (1.0 + connectivity_factor)
        
        graph_dist[i] = direct_dist * graph_penalty
    
    # Normalize graph distances
    max_dist = graph_dist.max()
    if max_dist > 1e-10:
        norm_dist = graph_dist / max_dist
    else:
        norm_dist = np.zeros(NP)
    
    # Compute adaptive CR per individual based on graph topology
    # Short graph distance -> high CR (trust mutant more)
    # Long graph distance -> low CR (preserve target features)
    adaptive_CR = self.CR * (1.0 - 0.3 * norm_dist)
    adaptive_CR = np.clip(adaptive_CR, 0.1, 0.95)
    
    # Generate trial population via binomial crossover
    trials = np.empty_like(population)
    
    # Random dimensions to always inherit from mutant
    jr = np.random.randint(0, dim, size=NP)
    
    for i in range(NP):
        cr = adaptive_CR[i]
        mask = np.random.rand(dim) < cr
        mask[jr[i]] = True
        trials[i] = np.where(mask, mutants[i], population[i])
    
    return trials
```