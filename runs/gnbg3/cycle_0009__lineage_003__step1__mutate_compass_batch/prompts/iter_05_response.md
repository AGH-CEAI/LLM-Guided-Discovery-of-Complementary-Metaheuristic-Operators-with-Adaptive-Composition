**Idea: PageRank-Guided Graph-Centric Mutation**

Build a directed k-NN graph over the population, compute PageRank to identify influential individuals, then use graph-centric directions (toward high-PR high-fitness nodes) to guide mutations. This leverages graph topology rather than raw geometry.

```python
def _mutate_compass_batch(self, population, fitness):
    """Vectorized compass mutation using k-NN graph topology and PageRank centrality."""
    NP, dim = population.shape
    
    # Compute fitness ranking
    order = np.argsort(fitness)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(NP)
    fitness_ranks = ranks / max(NP - 1, 1)
    
    # Build directed k-NN graph (each point points to its k nearest)
    k = max(3, min(7, NP // 6))
    
    # Compute pairwise squared distances
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each point
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Build directed adjacency matrix (row i -> column j means i points to j)
    adj = np.zeros((NP, NP))
    for i in range(NP):
        adj[i, nearest_indices[i]] = 1.0
    
    # Normalize rows to create stochastic transition matrix
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    transition = adj / row_sums
    
    # Add uniform teleportation (damping factor 0.85)
    damping = 0.85
    teleport = np.ones((NP, NP)) / NP
    stochastic = damping * transition + (1 - damping) * teleport
    
    # Compute PageRank via power iteration
    pr = np.ones(NP) / NP
    for _ in range(30):
        pr_new = stochastic.T @ pr
        pr_new = pr_new / (np.linalg.norm(pr_new) + 1e-10)
        if np.abs(pr_new - pr).max() < 1e-8:
            break
        pr = pr_new
    
    # Identify high-influence, high-fitness individuals (graph-informed targets)
    combined_score = 0.6 * (1.0 - fitness_ranks) + 0.4 * pr
    target_idx = np.argmax(combined_score)
    target = population[target_idx]
    
    # Compute local edge lengths for adaptive scaling
    local_edges = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
    avg_edge = np.mean(np.sqrt(local_edges))
    
    # Generate mutations using graph-centric directions
    mutants = np.empty_like(population)
    
    for i in range(NP):
        i_rank = fitness_ranks[i]
        i_pr = pr[i]
        
        # Select mutation strategy based on graph position
        if i_rank < 0.2 and i_pr > 0.5 * pr.max():
            # Top fitness AND high influence: conservative exploration
            # Small perturbation, mostly exploit current position
            direction = target - population[i]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction = direction / dir_norm
            mutants[i] = population[i] + 0.3 * self.F * direction * avg_edge
            mutants[i] += np.random.randn(dim) * avg_edge * 0.05
            
        elif i_rank > 0.6:
            # Low fitness: move aggressively toward graph-informed target
            # Use combined graph+rank information for direction
            direction = target - population[i]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction = direction / dir_norm
            
            # Scale by inverse PageRank (less connected individuals move more)
            move_scale = 0.8 + 0.4 * (1.0 - i_pr / (pr.max() + 1e-10))
            mutants[i] = population[i] + move_scale * self.F * direction * avg_edge
            mutants[i] += np.random.randn(dim) * avg_edge * 0.15
            
        else:
            # Medium fitness: balanced exploration-exploitation
            # Mix of best individual direction and random exploration
            best_idx = np.argmin(fitness_ranks)
            direction_best = population[best_idx] - population[i]
            direction_target = target - population[i]
            direction = 0.6 * direction_best + 0.4 * direction_target
            
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction = direction / dir_norm
            
            mutants[i] = population[i] + self.F * direction * avg_edge
            mutants[i] += np.random.randn(dim) * avg_edge * 0.1
    
    return np.clip(mutants, -100.0, 100.0)
```