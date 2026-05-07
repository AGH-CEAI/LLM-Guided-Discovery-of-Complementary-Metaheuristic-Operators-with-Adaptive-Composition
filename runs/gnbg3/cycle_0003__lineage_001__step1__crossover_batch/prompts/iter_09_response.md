**Idea: k-NN Centroid Modulated Crossover**

Crossover rate and binomial mask are modulated by the geometric relationship between each mutant and its k-NN centroid in the population—mutants extending the local neighborhood (far from centroid) trigger higher CR for exploration, while mutants collapsing toward it trigger lower CR for exploitation. Per-dimension spread further weights the mask toward high-variance dimensions.

```python
def _crossover_batch(self, population, mutants):
    NP, dim = population.shape
    trials = np.empty_like(population)
    
    # Build k-NN graph on population (k based on population size)
    k = max(2, min(5, NP // 10))
    
    # Compute pairwise distances within population
    pop_diff = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    pop_sq_dists = np.sum(pop_diff ** 2, axis=2)
    np.fill_diagonal(pop_sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest_indices = np.argsort(pop_sq_dists, axis=1)[:, :k]
    
    # Compute k-NN centroid for each individual
    knn_centroids = np.zeros_like(population)
    for i in range(NP):
        neighbors = nearest_indices[i]
        knn_centroids[i] = population[neighbors].mean(axis=0)
    
    # Compute distance from mutant to its target's k-NN centroid
    mutant_to_knn_dist = np.linalg.norm(mutants - knn_centroids, axis=1)
    
    # Compute average population spread (centroid distance)
    pop_centroid = population.mean(axis=0)
    avg_spread = np.mean(np.linalg.norm(population - pop_centroid, axis=1)) + 1e-8
    
    # Normalize: spatial modulation factor
    spatial_modulation = np.clip(mutant_to_knn_dist / avg_spread, 0.0, 3.0)
    
    # Per-dimension axis-aligned spread weight
    dim_std = np.std(population, axis=0)
    dim_spread = dim_std / (dim_std.mean() + 1e-8)
    
    for i in range(NP):
        cr = np.clip(self.CR * (1.0 + spatial_modulation[i] * 0.3), 0.1, 0.95)
        j_rand = np.random.randint(dim)
        
        for j in range(dim):
            effective_cr = cr * (0.5 + 0.5 * dim_spread[j])
            if j == j_rand or np.random.rand() < effective_cr:
                trials[i, j] = mutants[i, j]
            else:
                trials[i, j] = population[i, j]
    
    return trials
```