Looking at the context, I need to write a `_restart_if_needed` function using topology/graph-based reasoning. The current implementation uses k-NN graphs in `_adapt_parameters`, so I should take a different angle - analyzing graph connectivity and clustering to detect fragmented populations and replace poorly-connected individuals.

**Idea: k-NN Connectivity-Based Restart**
Use k-NN graph connectivity analysis to detect isolated individuals and fragmented clusters, then replace them with mutations of the best solutions to restore population cohesion while maintaining diversity.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on k-NN graph connectivity analysis."""
    NP, dim = population.shape
    
    # Build k-NN graph to analyze population topology
    k = max(2, min(5, NP // 8))
    
    # Compute pairwise distances
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute connectivity: how many times each individual appears in others' neighborhoods
    connectivity = np.zeros(NP)
    for i in range(NP):
        neighbors = nearest_indices[i]
        for n in neighbors:
            connectivity[n] += 1
    
    # Compute local clustering coefficient for each node
    clustering = np.zeros(NP)
    for i in range(NP):
        neighbors = set(nearest_indices[i])
        if len(neighbors) < 2:
            clustering[i] = 0.0
            continue
        edges = 0
        possible = 0
        for ni in neighbors:
            for nj in neighbors:
                if ni < nj:
                    possible += 1
                    if nj in set(nearest_indices[ni]):
                        edges += 1
        clustering[i] = edges / max(possible, 1)
    
    # Identify isolated individuals: low connectivity OR low clustering AND low fitness rank
    fitness_ranks = self._compute_fitness_ranking(fitness)
    isolation_score = (1.0 - connectivity / connectivity.max()) * 0.5 + \
                      (1.0 - clustering) * 0.3 + \
                      fitness_ranks * 0.2
    
    # Find indices to replace (most isolated)
    n_replace = max(NP // 4, 3)
    replace_indices = np.argsort(isolation_score)[-n_replace:]
    
    # Get best individuals for mutation sources
    n_best = min(5, NP)
    best_indices = np.argsort(fitness)[:n_best]
    
    # Create replacement individuals via topology-guided mutation
    new_individuals = []
    for idx in replace_indices:
        # Select source based on graph distance: prefer individuals far in graph from idx
        best_source = None
        best_graph_dist = -1
        for b_idx in best_indices:
            # Approximate graph distance via common neighbors
            n1 = set(nearest_indices[idx])
            n2 = set(nearest_indices[b_idx])
            common = len(n1 & n2)
            graph_dist_approx = 1.0 / (common + 0.1)  # Lower common neighbors = further
            if graph_dist_approx > best_graph_dist:
                best_graph_dist = graph_dist_approx
                best_source = b_idx
        
        if best_source is None:
            best_source = best_indices[0]
        
        # Mutation: directional jump toward unexplored regions
        source = population[best_source]
        direction = population[best_source] - population[idx]
        mutation_scale = np.random.uniform(0.8, 1.5)
        
        # Add noise proportional to local edge length
        local_edges = sq_dists[idx, nearest_indices[idx]]
        avg_local_dist = np.mean(np.sqrt(local_edges))
        
        new_point = source + mutation_scale * self.F * direction + \
                    np.random.randn(dim) * avg_local_dist * 0.5
        new_individuals.append(np.clip(new_point, -100.0, 100.0))
    
    # Inject new individuals into population
    for i, idx in enumerate(replace_indices):
        population[idx] = new_individuals[i]
    
    # Update best solution
    best_idx = np.argmin(fitness)
    x_opt = population[best_idx].copy()
    f_opt = fitness[best_idx]
    
    # Reset stagnation counter
    self.stagnation_counter = 0
    
    return population, f_opt, x_opt
```