Looking at the unsolved tasks, the worst ones (Tasks 16, 23, 20, 19, 21, 17, 18, 14, 22) have errors in the range 1-100+, suggesting they're likely multimodal, ill-conditioned, or compositional functions where the algorithm gets trapped in local optima. The current strategies (current-to-pbest/1 and rand-to-pbest/1) both converge toward pbest, which can be a local optimum on these hard functions.

I need a fundamentally different approach that:
1. Uses **weighted differences of multiple random vectors** for better exploration of the landscape
2. Includes a **rank-based directional component** that leverages fitness ranking more aggressively
3. Adds **dimensional perturbation** to escape narrow basins on non-separable functions

The key insight: on hard multimodal functions, we need more diverse search directions. I'll implement a mixture of DE/rand/2 (two difference vectors for broader jumps) and a "directed" strategy that uses fitness-weighted centroids instead of just pbest.

**Idea: Dual Difference Vectors with Fitness-Weighted Centroid**
Uses DE/rand/2 and a fitness-weighted centroid strategy for stronger exploration on multimodal/ill-conditioned functions.
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    
    # Fitness-weighted centroid using top half of population
    k = max(3, n // 2)
    top_idx = sorted_idx[:k]
    top_fit = fitness[top_idx]
    # Convert to weights (lower fitness = higher weight)
    max_fit = np.max(top_fit)
    min_fit = np.min(top_fit)
    range_fit = max_fit - min_fit + 1e-30
    weights = (max_fit - top_fit) / range_fit + 0.01
    weights /= np.sum(weights)
    centroid = np.sum(weights[:, np.newaxis] * population[top_idx], axis=0)
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    # Generate 5 distinct random indices per individual
    all_r = np.zeros((n, 5), dtype=int)
    for j in range(5):
        for i in range(n):
            while True:
                idx = np.random.randint(0, n)
                if idx != i and all(idx != all_r[i, :j]):
                    all_r[i, j] = idx
                    break
    
    # Strategy 1 (strategy_mask == True): current-to-centroid + rand/2 
    # Two difference vectors for broader exploration
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        # Stronger scaling for the centroid direction
        fw = f_col[idx1] * 0.8
        mutants[idx1] = (
            population[idx1]
            + fw * (centroid[np.newaxis, :] - population[idx1])
            + f_col[idx1] * (population[all_r[idx1, 0]] - population[all_r[idx1, 1]])
            + f_col[idx1] * 0.5 * (population[all_r[idx1, 2]] - population[all_r[idx1, 3]])
        )
    
    # Strategy 2 (strategy_mask == False): DE/rand/2 with best guidance
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=len(idx2))]
        
        base = all_r[idx2, 0]
        mutants[idx2] = (
            population[base]
            + f_col[idx2] * (population[pbest_idx] - population[base])
            + f_col[idx2] * (population[all_r[idx2, 1]] - population[all_r[idx2, 2]])
            + f_col[idx2] * 0.5 * (population[all_r[idx2, 3]] - population[all_r[idx2, 4]])
        )
    
    # Add small Gaussian perturbation to help escape narrow basins
    gen = getattr(self, 'generation', 0)
    noise_scale = max(0.001, 0.05 / (1.0 + gen / 100.0))
    noise = np.random.normal(0, noise_scale * (self.ub - self.lb), (n, dim))
    mutants += noise
    
    return mutants
```