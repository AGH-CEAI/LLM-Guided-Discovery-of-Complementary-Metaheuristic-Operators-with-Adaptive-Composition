Looking at the unsolved tasks, the errors are stuck at very high values (1e+1 to 1e+2), suggesting the algorithm is trapped in local optima on multimodal, non-separable, and possibly ill-conditioned landscapes. The existing variants all use pbest-guided mutations which converge prematurely on deceptive landscapes.

**Key insight:** Tasks 16, 20, 23, 19 have errors ~10-100, indicating the population converges to wrong basins entirely. We need a mutation strategy that:
1. Uses **rank-weighted direction estimation** to exploit fitness landscape gradient information
2. Employs **multiple difference vectors** for better exploration on high-dimensional non-separable problems
3. Includes a **dimension-adaptive perturbation** where each coordinate gets scaled noise based on population spread, preventing premature convergence in narrow dimensions

This is fundamentally different from prior variants because it combines: (a) a weighted centroid-based mutation (not just pbest), (b) two independent difference vectors for richer directional information, and (c) an adaptive Gaussian perturbation scaled by per-dimension population spread.

**Idea: Rank-Weighted Centroid with Double Differences and Dimension-Adaptive Noise**
Uses fitness-rank-weighted centroid as attractor with two difference vectors and per-dimension diversity-scaled perturbation to escape local optima on multimodal non-separable landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    # Rank-based weights (higher weight for better fitness)
    sorted_idx = np.argsort(fitness)
    ranks = np.empty(n)
    ranks[sorted_idx] = np.arange(n)
    # Exponential weighting favoring top individuals
    weights = np.exp(-ranks / max(1, n * 0.3))
    weights /= weights.sum() + 1e-30
    
    # Weighted centroid (better than single pbest)
    centroid = np.sum(weights[:, np.newaxis] * population, axis=0)
    
    # Per-dimension standard deviation for adaptive noise
    dim_std = np.std(population, axis=0) + 1e-30
    
    # p-best for directed component
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    # Union for difference vectors
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    # Generate 4 distinct random indices
    indices = np.arange(n)
    r1 = self._random_indices_not_equal(n, indices)
    r2 = np.zeros(n, dtype=int)
    r3 = np.zeros(n, dtype=int)
    r4 = np.zeros(n, dtype=int)
    for i in range(n):
        exc = {i, r1[i]}
        for arr in [r2, r3]:
            while True:
                idx = np.random.randint(0, union_size)
                if idx not in exc:
                    arr[i] = idx
                    exc.add(idx)
                    break
        while True:
            idx = np.random.randint(0, n)
            if idx not in exc:
                r4[i] = idx
                break
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    # Strategy 1 (mask=True): centroid-guided with double differences
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        noise = np.random.normal(0, 1, (len(idx1), dim)) * dim_std[np.newaxis, :] * 0.1
        mutants[idx1] = (
            population[idx1]
            + f_col[idx1] * (centroid[np.newaxis, :] - population[idx1])
            + f_col[idx1] * 0.5 * (population[r1[idx1]] - union[r2[idx1]])
            + f_col[idx1] * 0.5 * (population[r4[idx1]] - union[r3[idx1]])
            + noise
        )
    
    # Strategy 2 (mask=False): pbest-guided with centroid blend and perturbation
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        blend = 0.5 * (population[pbest_idx[idx2]] + centroid[np.newaxis, :])
        noise = np.random.normal(0, 1, (len(idx2), dim)) * dim_std[np.newaxis, :] * 0.05
        mutants[idx2] = (
            population[idx2]
            + f_col[idx2] * (blend - population[idx2])
            + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            + noise
        )
    
    return mutants
```