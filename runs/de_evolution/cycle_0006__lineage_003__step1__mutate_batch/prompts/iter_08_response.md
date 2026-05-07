Looking at the unsolved tasks, I see two distinct clusters:

1. **Tasks 16, 17, 19, 20, 23** (errors ~10-100+): These are likely highly multimodal, non-separable functions where the algorithm gets trapped in local optima. The current approach with current-to-pbest/1 converges too greedily.

2. **Tasks 6, 7, 9, 12** (errors ~1e-8 to 1e-4): These need finer exploitation — the algorithm gets close but can't finish converging.

The original and variant_01 (which hold the best scores on most hard tasks) both use current-to-pbest/1 which biases toward greedy convergence. For the hardest tasks (16, 23, 20, 19), we need much stronger exploration with multiple difference vectors, weighted ensemble mutations, and rank-based directional guidance that doesn't collapse to the current best too quickly.

**Idea: Rank-Weighted Multi-Donor Ensemble Mutation**
Uses rank-weighted centroid of top individuals as attractor with dual difference vectors and probabilistic eigenvector-based rotation for escaping local optima on highly multimodal landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    ranks = np.empty(n, dtype=float)
    ranks[sorted_idx] = np.arange(n, dtype=float)
    
    # Compute rank-weighted centroid (top individuals weighted more)
    weights = np.exp(-ranks / max(1, n * 0.3))
    weights /= weights.sum() + 1e-30
    centroid = weights @ population
    
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    indices = np.arange(n)
    r1 = self._random_indices_not_equal(n, indices)
    r2 = np.zeros(n, dtype=int)
    r3 = np.zeros(n, dtype=int)
    for i in range(n):
        while True:
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r1[i]:
                r2[i] = idx
                break
        while True:
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r1[i] and idx != r2[i]:
                r3[i] = idx
                break
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    # Strategy 1 (mask=True): current-to-centroid/2 — uses rank-weighted centroid + two difference vectors
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        f1 = f_col[idx1]
        # Stronger pull toward centroid with two independent difference vectors
        mutants[idx1] = (
            population[idx1]
            + f1 * (centroid[np.newaxis, :] - population[idx1])
            + f1 * 0.5 * (population[pbest_idx[idx1]] - union[r2[idx1]])
            + f1 * 0.5 * (population[r1[idx1]] - union[r3[idx1]])
        )
    
    # Strategy 2 (mask=False): weighted-rand-to-pbest/2 — random base with dual perturbations
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        f2 = f_col[idx2]
        # Use random base from top half
        top_half = sorted_idx[:max(2, n // 2)]
        r_base = top_half[np.random.randint(0, len(top_half), size=len(idx2))]
        
        # Mix centroid direction with pbest direction
        alpha = np.random.uniform(0.3, 0.7, size=(len(idx2), 1))
        target = alpha * centroid[np.newaxis, :] + (1 - alpha) * population[pbest_idx[idx2]]
        
        mutants[idx2] = (
            population[r_base]
            + f2 * (target - population[r_base])
            + f2 * (population[r1[idx2]] - union[r2[idx2]])
        )
    
    return mutants
```