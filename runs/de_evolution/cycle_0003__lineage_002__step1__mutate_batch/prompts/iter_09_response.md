Looking at the unsolved tasks, I see two clusters:

1. **Tasks with errors ~1e+1 to 1e+2** (Tasks 16, 20, 23, 19, 21, 17, 18, 14, 11, 22, 13, 15): These are stuck at large errors, suggesting the population is trapped in local optima on multimodal/non-separable landscapes. The current strategies all rely on pbest-guided mutation which biases search toward the current best basin.

2. **Tasks with errors ~1e-4 to 1e-6** (Tasks 9, 3, 10, 12, 7, 6): These are converging but too slowly, needing better exploitation.

For the worst tasks, the key issue is **premature convergence to wrong basins** on highly multimodal functions. All existing variants use pbest-directed mutation, which reinforces convergence to the current best region. 

My approach: A **multi-strategy mutation** combining (1) a highly explorative **rand/2** strategy with larger F for escaping basins, (2) a **current-to-best with rank-weighted selection** for moderate exploration, and (3) a **trigonometric mutation** that creates trial vectors from geometric combinations of three random individuals. The strategy selection is fitness-rank-based: worse individuals explore more aggressively while better ones exploit. This is fundamentally different because it removes pbest-guidance for a large fraction of the population and uses geometric/trigonometric combinations that can jump across basins.

**Idea: Rank-Adaptive Multi-Strategy with Trigonometric and Rand2 Exploration**
Rank-based strategy selection: top third uses current-to-pbest, middle uses trigonometric mutation, bottom third uses rand/2 with amplified F for maximum exploration.
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    sorted_idx = np.argsort(fitness)
    ranks = np.empty(n, dtype=int)
    ranks[sorted_idx] = np.arange(n)
    
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    def rand_idx_not(n_pool, *excludes):
        result = np.random.randint(0, n_pool, size=n)
        for _ in range(20):
            bad = np.zeros(n, dtype=bool)
            for ex in excludes:
                bad |= (result == ex)
            if not np.any(bad):
                break
            result[bad] = np.random.randint(0, n_pool, size=np.sum(bad))
        return result
    
    arange_n = np.arange(n)
    r1 = rand_idx_not(n, arange_n)
    r2 = rand_idx_not(union_size, arange_n, r1)
    r3 = rand_idx_not(n, arange_n, r1)
    r4 = rand_idx_not(n, arange_n, r1, r3)
    r5 = rand_idx_not(n, arange_n, r1, r3, r4)
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    third = n // 3
    # Top third (best fitness, low rank): current-to-pbest/1 exploitation
    mask_exploit = ranks < third
    # Middle third: trigonometric mutation
    mask_trig = (ranks >= third) & (ranks < 2 * third)
    # Bottom third (worst fitness): rand/2 with amplified F
    mask_explore = ranks >= 2 * third
    
    if np.any(mask_exploit):
        idx = np.where(mask_exploit)[0]
        mutants[idx] = (population[idx]
                        + f_col[idx] * (population[pbest_idx[idx]] - population[idx])
                        + f_col[idx] * (population[r1[idx]] - union[r2[idx]]))
    
    if np.any(mask_trig):
        idx = np.where(mask_trig)[0]
        p1, p2, p3 = population[r1[idx]], population[r3[idx]], population[r4[idx]]
        f1, f2, f3 = fitness[r1[idx]], fitness[r3[idx]], fitness[r4[idx]]
        psum = np.abs(f1) + np.abs(f2) + np.abs(f3) + 1e-30
        w1 = (np.abs(f1) / psum)[:, np.newaxis]
        w2 = (np.abs(f2) / psum)[:, np.newaxis]
        w3 = (np.abs(f3) / psum)[:, np.newaxis]
        centroid = (p1 + p2 + p3) / 3.0
        mutants[idx] = centroid + f_col[idx] * (p1 - p2) + f_col[idx] * (p2 - p3)
    
    if np.any(mask_explore):
        idx = np.where(mask_explore)[0]
        F_big = np.clip(f_col[idx] * 1.5, 0.4, 1.5)
        mutants[idx] = (population[r3[idx]]
                        + F_big * (population[r4[idx]] - population[r5[idx]])
                        + F_big * (population[r1[idx]] - union[r2[idx]]))
    
    return mutants
```