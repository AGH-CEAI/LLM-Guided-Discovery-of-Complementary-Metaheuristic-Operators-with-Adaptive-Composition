Looking at the unsolved tasks, I see errors ranging from ~1e+2 down to ~1e-8. The worst tasks (16, 23, 20, 19) have errors stuck at 1e+1 to 1e+2, suggesting highly multimodal or compositional functions where the algorithm gets trapped in local optima. The existing variants use current-to-pbest/1 and rand-to-pbest/1, which both bias toward the current best region. For these hard multimodal tasks, we need:

1. **More aggressive exploration** via DE/rand/2 (uses 5 random vectors, no bias toward current best)
2. **Weighted mutation toward best with rank-based scaling** for exploitation when close
3. **Eigenvector-based rotation** to handle non-separable functions (tasks 16-23 are likely rotated/composed)

The key mechanism: a mix of DE/rand/2 for global exploration (breaking out of local optima on tasks 16, 23, 20) and a covariance-guided current-to-best for non-separable functions, fundamentally different from the pbest-only approaches used before.

**Idea: Eigenvector-Guided Rand2 and Covariance Mutation**
Combines DE/rand/2 for exploration with covariance-matrix-adapted current-to-best for non-separable multimodal landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    f_col = f_values[:, np.newaxis]
    
    sorted_idx = np.argsort(fitness)
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    # Compute covariance-based rotation from top half of population
    n_elite = max(3, n // 3)
    elite = population[sorted_idx[:n_elite]]
    center = np.mean(elite, axis=0)
    diffs = elite - center
    if n_elite > dim and dim <= 100:
        cov = np.cov(diffs.T) + 1e-10 * np.eye(dim)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_eigvals = np.sqrt(eigvals)
        except:
            eigvecs = np.eye(dim)
            sqrt_eigvals = np.ones(dim)
    else:
        eigvecs = np.eye(dim)
        sqrt_eigvals = np.ones(dim) * (np.std(population, axis=0).mean() + 1e-10)
    
    mutants = np.empty((n, dim))
    
    # Strategy 1 (mask=True): DE/rand/2 — strong exploration for multimodal
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        k = len(idx1)
        # Generate 5 distinct random indices
        r = np.zeros((k, 5), dtype=int)
        for j in range(5):
            r[:, j] = np.random.randint(0, n, size=k)
            for attempt in range(50):
                bad = r[:, j] == idx1
                for jj in range(j):
                    bad |= r[:, j] == r[:, jj]
                if not np.any(bad):
                    break
                r[bad, j] = np.random.randint(0, n, size=np.sum(bad))
        
        # DE/rand/2: x_r0 + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)
        mutants[idx1] = (population[r[:, 0]]
                         + f_col[idx1] * (population[r[:, 1]] - population[r[:, 2]])
                         + 0.5 * f_col[idx1] * (population[r[:, 3]] - population[r[:, 4]]))
    
    # Strategy 2 (mask=False): Covariance-guided current-to-pbest
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        k = len(idx2)
        
        # Direction toward pbest in eigenvector space
        diff_pbest = population[pbest_idx[idx2]] - population[idx2]
        
        # Random perturbation in eigenvector space
        z = np.random.standard_normal((k, dim))
        rotated_noise = z @ np.diag(sqrt_eigvals) @ eigvecs.T
        scale = np.std(population, axis=0).mean() + 1e-10
        rotated_noise *= 0.1 * scale / (np.std(rotated_noise) + 1e-10)
        
        r1 = np.random.randint(0, n, size=k)
        r2 = np.random.randint(0, n, size=k)
        
        mutants[idx2] = (population[idx2]
                         + f_col[idx2] * diff_pbest
                         + 0.5 * f_col[idx2] * (population[r1] - population[r2])
                         + f_col[idx2] * rotated_noise)
    
    return mutants
```