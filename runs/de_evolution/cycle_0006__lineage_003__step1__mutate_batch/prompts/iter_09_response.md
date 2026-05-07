Looking at the unsolved tasks, I see errors spanning from ~1e+2 (Task 16) down to ~5e-8 (Task 6). The worst tasks (16, 23, 20, 19, 21, 17, 18, 14) have errors stuck at 1-100+, suggesting they're multimodal, ill-conditioned, or have deceptive landscapes where current-to-pbest gets trapped in local optima.

The original and variant_01 (which are best on most unsolved tasks) both use current-to-pbest/1 and rand-to-pbest/1 — both heavily biased toward pbest. For highly multimodal/non-separable functions, this greedy attraction to pbest causes premature convergence.

My approach: **Weighted multi-donor mutation with rank-based diverse attractors and eigenvector-guided perturbation**. Instead of just pbest, I use multiple ranked donors with decreasing weights, combined with a covariance-adapted perturbation to navigate ill-conditioned landscapes. This breaks the single-attractor bias while adapting to the local landscape geometry.

**Idea: Eigenvector-Guided Multi-Donor Mutation**
Uses covariance-based eigenvector rotation with multiple ranked donors and adaptive exploration noise to escape local optima on multimodal ill-conditioned landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    f_col = f_values[:, np.newaxis]
    
    sorted_idx = np.argsort(fitness)
    
    # Compute simplified covariance direction from top individuals
    n_top = max(3, n // 4)
    top_pop = population[sorted_idx[:n_top]]
    center = np.mean(top_pop, axis=0)
    
    # Compute covariance eigenvectors for rotation
    try:
        if dim <= 100 and n_top >= 3:
            diffs = top_pop - center
            cov = np.dot(diffs.T, diffs) / max(1, n_top - 1) + 1e-10 * np.eye(dim)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            # Normalize eigenvalues for scaling
            sqrt_eigvals = np.sqrt(eigvals / np.max(eigvals))
        else:
            eigvecs = np.eye(dim)
            sqrt_eigvals = np.ones(dim)
    except:
        eigvecs = np.eye(dim)
        sqrt_eigvals = np.ones(dim)
    
    # Multiple donor strategy: pick 3 different ranked donors per individual
    p1 = max(2, int(0.1 * n))
    p2 = max(3, int(0.3 * n))
    p3 = max(4, int(0.6 * n))
    
    donor1_idx = sorted_idx[np.random.randint(0, p1, size=n)]
    donor2_idx = sorted_idx[np.random.randint(0, p2, size=n)]
    
    r1 = self._random_indices_not_equal(n, np.arange(n))
    
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    r2 = np.zeros(n, dtype=int)
    for i in range(n):
        for _ in range(100):
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r1[i]:
                r2[i] = idx
                break
    
    mutants = np.empty((n, dim))
    
    # Strategy 1: Eigenvector-guided current-to-multi-pbest
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        # Weighted combination of two donors
        w1, w2 = 0.7, 0.3
        direction = (w1 * (population[donor1_idx[idx1]] - population[idx1]) +
                     w2 * (population[donor2_idx[idx1]] - population[idx1]))
        diff_vec = population[r1[idx1]] - union[r2[idx1]]
        # Eigenvector-guided noise
        noise = np.random.normal(0, 1, (len(idx1), dim))
        rotated_noise = noise @ eigvecs.T * sqrt_eigvals[np.newaxis, :]
        rotated_noise = rotated_noise @ eigvecs
        noise_scale = 0.1 * f_col[idx1]
        
        mutants[idx1] = (population[idx1] + f_col[idx1] * direction +
                         f_col[idx1] * diff_vec + noise_scale * rotated_noise)
    
    # Strategy 2: rand/2 with eigenvector perturbation for exploration
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
        r3 = self._random_indices_not_equal(n, np.arange(n))[idx2]
        
        mutants[idx2] = (population[r_base] +
                         f_col[idx2] * (population[donor1_idx[idx2]] - population[r3]) +
                         f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]]))
    
    return mutants
```