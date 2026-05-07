**Idea: k-NN Graph Connectivity-Guided Crossover**
One-line description: Build a k-NN graph over the population, compute per-individual connectivity and clustering coefficients, then use these topology properties to modulate CR per-dimension and guide which dimensions inherit from mutants vs. population.

```python
def _crossover_batch(self, population, mutants):
    """Graph topology-guided crossover using k-NN connectivity and clustering."""
    NP, dim = self.dim
    
    # Build k-NN graph over population
    k = max(2, min(5, NP // 10))
    
    # Compute pairwise squared distances
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute per-individual connectivity (normalized degree in k-NN graph)
    connectivity = np.zeros(NP)
    for i in range(NP):
        connectivity[i] = np.sum(sq_dists[i, nearest_indices[i]] < np.inf)
    connectivity = connectivity / k  # Normalize
    
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
    
    # Compute graph-based CR modifier per individual
    # High clustering + high connectivity = well-integrated -> higher CR for exploration
    # Low clustering + low connectivity = isolated -> lower CR for exploitation
    graph_influence = 0.5 * (connectivity / (connectivity.max() + 1e-8)) + \
                      0.5 * clustering
    graph_CR_modifier = 1.0 + 0.3 * (graph_influence - 0.5)
    
    # Generate trial vectors
    trials = np.empty_like(population)
    
    for i in range(NP):
        # Per-individual CR influenced by graph topology
        cr_i = np.clip(self.CR * graph_CR_modifier[i], 0.1, 0.95)
        
        # Determine number of dimensions to inherit from mutant
        n_dims_to_cross = int(np.ceil(cr_i * dim))
        
        # Select dimensions: use graph-based preference
        # For dimensions where neighbors have similar values, prefer mutant (exploration)
        # For dimensions with high variance among neighbors, prefer target (exploitation)
        dim_variance = np.var(population[nearest_indices[i]], axis=0)
        
        # Probability of taking from mutant increases with low neighbor variance
        mutant_prob = 1.0 - np.clip(dim_variance / (dim_variance.max() + 1e-8), 0, 1)
        mutant_prob = mutant_prob / (mutant_prob.sum() + 1e-8)  # Normalize
        
        # Select dimensions based on graph-informed probability
        if n_dims_to_cross > 0 and n_dims_to_cross < dim:
            cross_dims = np.random.choice(dim, size=n_dims_to_cross, replace=False, p=mutant_prob)
        elif n_dims_to_cross >= dim:
            cross_dims = np.arange(dim)
        else:
            # Random crossover for at least one dimension
            cross_dims = np.array([np.random.randint(dim)])
        
        # Create trial vector
        mask = np.zeros(dim, dtype=bool)
        mask[cross_dims] = True
        
        trials[i] = np.where(mask, mutants[i], population[i])
    
    return trials
```