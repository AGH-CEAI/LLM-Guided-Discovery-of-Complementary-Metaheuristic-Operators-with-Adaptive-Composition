**Idea: Betweenness-Centrality Survivor Selection**
Use k-NN graph betweenness centrality to preserve structurally important individuals that maintain population connectivity, preventing premature convergence on ill-conditioned tasks.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection biased by graph betweenness centrality to maintain population connectivity."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    
    pop_size, dim = population.shape
    
    # Build k-NN graph and compute betweenness centrality for each individual
    k = max(2, min(5, pop_size // 4))
    dist_matrix = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        diffs = population - population[i]
        dist_matrix[i] = np.sqrt(np.sum(diffs * diffs, axis=1))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # k-NN adjacency
    knn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
    row_idx = np.repeat(np.arange(pop_size), k)
    col_idx = knn_indices.flatten()
    data = np.ones(len(row_idx))
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(pop_size, pop_size))
    adj = adj + adj.T
    adj = (adj > 0).astype(float)
    
    # Compute betweenness centrality via Floyd-Warshall path counts
    n_paths = np.zeros((pop_size, pop_size))
    dists, preds = shortest_path(adj, directed=False, unweighted=True, return_predecessors=True)
    dists = np.nan_to_num(dists, nan=np.inf)
    
    for start in range(pop_size):
        for end in range(pop_size):
            if start != end and dists[start, end] < np.inf:
                current = end
                while current != start:
                    prev = preds[start, current]
                    if prev < 0:
                        break
                    n_paths[start, prev] += 1
                    n_paths[prev, end] += 1
                    current = prev
    
    centrality = np.zeros(pop_size)
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if dists[i, j] < np.inf:
                c_ij = n_paths[i, j] + n_paths[j, i]
                centrality[i] += c_ij
                centrality[j] += c_ij
    centrality = centrality / (centrality.max() + 1e-10)
    
    # Selection: prefer individuals with higher centrality when fitness is similar
    new_population = population.copy()
    new_fitness = fitness.copy()
    improved_mask = np.zeros(pop_size, dtype=bool)
    
    for i in range(pop_size):
        trial_f = trial_fitness[i]
        parent_f = fitness[i]
        
        if trial_f < parent_f:
            new_population[i] = trials[i]
            new_fitness[i] = trial_f
            improved_mask[i] = True
        elif np.abs(trial_f - parent_f) < 1e-14:
            # For equal fitness, probabilistically prefer higher centrality
            trial_centrality = 0.5  # New individual has neutral centrality
            if np.random.random() < (centrality[i] - trial_centrality + 0.5):
                new_population[i] = trials[i]
                new_fitness[i] = trial_f
                improved_mask[i] = True
    
    return new_population, new_fitness, improved_mask
```