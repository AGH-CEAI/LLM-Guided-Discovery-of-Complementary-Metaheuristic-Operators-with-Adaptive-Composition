Looking at the unsolved tasks, I see two distinct failure modes:

1. **Tasks 16, 23, 20, 19, 21, 17, 18, 14** (errors ~1-100): These are likely highly multimodal, non-separable, or ill-conditioned functions where the algorithm gets trapped in local optima. The current strategies (current-to-pbest/1 and rand-to-pbest/1) both bias heavily toward the current best, causing premature convergence.

2. **Tasks 6, 7, 12, 3, 9** (errors ~1e-8 to 1e-4): These need better fine-tuning in the endgame.

The key insight: for the worst tasks, we need **much more diverse exploration** with multiple fundamentally different mutation strategies, including strategies that DON'T always gravitate toward pbest. I'll implement a multi-strategy approach combining: (1) DE/rand/2 for deep exploration, (2) current-to-pbest with weighted difference vectors using rank-based selection, (3) a "best/2" strategy for exploitation, and (4) a trigonometric mutation for escaping local optima. Each individual gets randomly assigned one of these four strategies.

**Idea: Quad-Strategy with Rand2 Trigonometric and Rank-Weighted Mutations**
Four diverse strategies including DE/rand/2, trigonometric mutation, rank-weighted current-to-pbest, and best/2 to escape local optima on multimodal problems.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    sorted_idx = np.argsort(fitness)
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    best_idx = sorted_idx[0]
    
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    def rand_idx_not_in(exclude_sets, size, max_val):
        result = np.empty(size, dtype=int)
        for i in range(size):
            while True:
                idx = np.random.randint(0, max_val)
                if idx not in exclude_sets[i]:
                    result[i] = idx
                    break
        return result
    
    indices = np.arange(n)
    r1 = self._random_indices_not_equal(n, indices)
    r2 = rand_idx_not_in([{i, r1[i]} for i in range(n)], n, union_size)
    r3 = rand_idx_not_in([{i, r1[i]} for i in range(n)], n, n)
    r4 = rand_idx_not_in([{i, r1[i], r3[i]} for i in range(n)], n, n)
    r5 = rand_idx_not_in([{i, r1[i], r3[i], r4[i]} for i in range(n)], n, n)
    
    # Rank-based weights for weighted difference
    ranks = np.empty(n)
    ranks[sorted_idx] = np.arange(n)
    weights = np.exp(-ranks / (n / 3.0))
    weights /= weights.sum()
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    strat = np.random.randint(0, 4, size=n)
    
    # Strategy 0: current-to-pbest/1 (standard SHADE)
    m0 = strat == 0
    if np.any(m0):
        idx = np.where(m0)[0]
        mutants[idx] = population[idx] + f_col[idx] * (population[pbest_idx[idx]] - population[idx]) + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
    
    # Strategy 1: DE/rand/2 - pure exploration
    m1 = strat == 1
    if np.any(m1):
        idx = np.where(m1)[0]
        mutants[idx] = population[r3[idx]] + f_col[idx] * (population[r4[idx]] - population[r5[idx]]) + 0.5 * f_col[idx] * (population[r1[idx]] - union[r2[idx]])
    
    # Strategy 2: best/2 - aggressive exploitation
    m2 = strat == 2
    if np.any(m2):
        idx = np.where(m2)[0]
        mutants[idx] = population[best_idx] + f_col[idx] * (population[r3[idx]] - population[r4[idx]]) + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
    
    # Strategy 3: Trigonometric mutation for escaping local optima
    m3 = strat == 3
    if np.any(m3):
        idx = np.where(m3)[0]
        p1, p2, p3 = population[r1[idx]], population[r3[idx]], population[r4[idx]]
        f1, f2, f3 = fitness[r1[idx]], fitness[r3[idx]], fitness[r4[idx]]
        pp = np.abs(f1) + np.abs(f2) + np.abs(f3) + 1e-30
        w1 = (np.abs(f1) / pp)[:, np.newaxis]
        w2 = (np.abs(f2) / pp)[:, np.newaxis]
        w3 = (np.abs(f3) / pp)[:, np.newaxis]
        centroid = (p1 + p2 + p3) / 3.0
        mutants[idx] = centroid + (p1 - p2) * w1 + (p2 - p3) * w2 + (p3 - p1) * w3
    
    return mutants
```