**Idea: k-NN Graph Connectivity Survivor Selection**
One-line description: Build a k-nearest-neighbor graph on the combined population and trials, compute per-individual connectivity scores, and use these to bias survivor selection toward individuals that preserve graph structure — preventing diversity collapse on rugged/deceptive tasks.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Graph-connectivity-aware survivor selection using k-NN topology."""
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    pop_size = len(population)
    combined = np.vstack([population, trials])
    
    # Build k-NN graph on combined population to assess structural connectivity
    k = min(5, pop_size - 1)
    
    # Compute pairwise Euclidean distances for the combined set
    # Vectorized for speed: broadcast to (2N, 2N, D), then reduce
    diffs = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]  # (2N, 2N, D)
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))  # (2N, 2N)
    
    # For each individual, compute mean distance to its k-nearest neighbors
    # (k+1 because the individual itself has distance 0)
    sorted_indices = np.argpartition(dist_matrix, k + 1, axis=1)
    knn_connectivity = np.zeros(2 * pop_size)
    for i in range(2 * pop_size):
        neighbor_dists = dist_matrix[i, sorted_indices[i, :k + 1]]
        knn_connectivity[i] = np.mean(neighbor_dists)
    
    pop_conn = knn_connectivity[:pop_size]           # connectivity of current population
    trial_conn = knn_connectivity[pop_size:]         # connectivity of trial individuals
    
    improved_mask = trial_fitness < fitness
    
    # --- PHASE 1: Clear fitness improvements ---
    # Always accept trials that strictly improve fitness
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    # --- PHASE 2: Connectivity-preserving replacement for near-ties ---
    # When trial ≈ parent in fitness, prefer the one that maintains better
    # graph structure (lower knn_connectivity = denser local neighborhood)
    not_improved = ~improved_mask
    if np.any(not_improved):
        fitness_diffs = fitness - trial_fitness
        max_diff = np.max(fitness_diffs[not_improved]) + 1e-10
        
        # "Near-tie" threshold: within 20% of the worst-case gap
        near_tie_threshold = 0.2 * max_diff
        near_tie_mask = not_improved & (fitness_diffs < near_tie_threshold)
        
        if np.any(near_tie_mask):
            idx_nt = np.where(near_tie_mask)[0]
            
            # Compute connectivity advantage for each near-tie pair
            conn_advantage = pop_conn[idx_nt] - trial_conn[idx_nt]  # positive = trial is better
            
            # Accept trial if it improves connectivity by at least 5%
            connectivity_threshold = 0.05 * pop_conn[idx_nt]
            accept_connectivity = conn_advantage > connectivity_threshold
            
            accept_idx = idx_nt[accept_connectivity]
            new_population[accept_idx] = trials[accept_idx]
            new_fitness[accept_idx] = trial_fitness[accept_idx]
            
            # Mark accepted indices as improved for adaptation tracking
            improved_mask[accept_idx] = True
    
    return new_population, new_fitness, improved_mask
```