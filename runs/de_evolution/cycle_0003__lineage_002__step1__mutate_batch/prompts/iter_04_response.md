Looking at the unsolved tasks, the errors are stuck at very high values (1e+2 down to 1e+0), suggesting the algorithm is trapped in local optima and lacks sufficient exploration. The current variants all use pbest-directed mutations which converge prematurely on multimodal, non-separable, ill-conditioned landscapes.

**Key diagnosis:** Tasks 16 (159), 20 (18.5), 23 (14.9), 19 (6.4) have enormous errors — the population is converging to wrong basins. We need a mutation strategy that:
1. Has strong global exploration via multiple difference vectors
2. Uses rank-weighted greedy perturbation to escape local optima
3. Incorporates a "current-to-best/2" style with additional diversity from random pairs

The fundamental difference from prior variants: instead of just pbest-directed movement, I'll use a **weighted multi-donor strategy** combining (a) movement toward the best with an aggressive scaling, (b) two independent difference vectors for exploration, and (c) a fitness-rank-based adaptive scaling that gives worse individuals larger step sizes to escape bad basins.

**Idea: Rank-Adaptive Multi-Donor Exploration**
Fitness-rank-scaled mutation with best/2 using two independent difference vectors for maximal exploration on hard multimodal tasks.
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    # Rank-based scaling: worst individuals get larger F multipliers
    ranks = np.empty(n, dtype=float)
    ranks[sorted_idx] = np.arange(n, dtype=float)
    rank_scale = 0.5 + 0.5 * (ranks / max(n - 1, 1))  # range [0.5, 1.0]
    
    # Best individual
    best_idx = sorted_idx[0]
    
    # p-best for diversity
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    # Union for donor selection
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    # Select 4 distinct random indices for two difference vectors
    r1 = np.empty(n, dtype=int)
    r2 = np.empty(n, dtype=int)
    r3 = np.empty(n, dtype=int)
    r4 = np.empty(n, dtype=int)
    for i in range(n):
        candidates = list(range(n))
        candidates.remove(i)
        chosen = np.random.choice(candidates, size=min(2, len(candidates)), replace=False)
        r1[i] = chosen[0]
        r3[i] = chosen[1] if len(chosen) > 1 else chosen[0]
        # r2, r4 from union
        attempts = 0
        while attempts < 100:
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r1[i]:
                r2[i] = idx
                break
            attempts += 1
        else:
            r2[i] = np.random.randint(0, union_size)
        attempts = 0
        while attempts < 100:
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r3[i] and idx != r2[i]:
                r4[i] = idx
                break
            attempts += 1
        else:
            r4[i] = np.random.randint(0, union_size)
    
    f_col = f_values[:, np.newaxis]
    rs = rank_scale[:, np.newaxis]
    
    mutants = np.empty((n, dim))
    
    # Strategy 1 (mask True): current-to-best/2 with rank scaling
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        mutants[idx1] = (
            population[idx1]
            + rs[idx1] * f_col[idx1] * (population[best_idx] - population[idx1])
            + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            + 0.5 * f_col[idx1] * (population[r3[idx1]] - union[r4[idx1]])
        )
    
    # Strategy 2 (mask False): rand-to-pbest/2 with rank scaling
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
        mutants[idx2] = (
            population[r_base]
            + rs[idx2] * f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
            + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            + 0.5 * f_col[idx2] * (population[r3[idx2]] - union[r4[idx2]])
        )
    
    return mutants
```