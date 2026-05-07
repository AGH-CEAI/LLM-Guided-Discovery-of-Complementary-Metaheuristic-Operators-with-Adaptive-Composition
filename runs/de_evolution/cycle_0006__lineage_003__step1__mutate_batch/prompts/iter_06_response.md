Looking at the unsolved tasks, I see errors ranging from ~1e+2 down to ~1e-8. The worst tasks (16, 23, 20, 19, 21, 17, 18, 14) have errors stuck at 1-100+, suggesting multimodal/ill-conditioned landscapes where the current strategies get trapped in local optima. The original and variant_01 (which are similar current-to-pbest strategies) perform best on these, but can't escape.

The key insight: current-to-pbest/1 converges too greedily toward pbest, losing diversity on multimodal landscapes. I need a strategy that:
1. Uses **rank-based weighted recombination** of multiple donors (like CMA-ES inspiration) for better search direction estimation
2. Employs **eigenvalue-based rotation** via population covariance to handle non-separable ill-conditioned functions
3. Includes a **proximity-avoiding mechanism** where individuals far from the current best explore more aggressively

This is fundamentally different from all prior variants which used standard DE mutation (base + F*diff). Instead, this creates mutants by sampling from a covariance-adapted distribution centered on a weighted centroid of top individuals, combined with a DE-style difference vector for diversity.

**Idea: Covariance-Guided Weighted Centroid Mutation**
Mutants generated from rank-weighted centroid of top individuals plus covariance-adapted perturbation and DE difference vector.
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    
    # Rank-weighted centroid from top half
    top_k = max(3, n // 3)
    top_pop = population[sorted_idx[:top_k]]
    # Log-linear weights
    weights = np.log(top_k + 0.5) - np.log(np.arange(1, top_k + 1))
    weights = weights / np.sum(weights)
    centroid = np.sum(weights[:, np.newaxis] * top_pop, axis=0)
    
    # Compute covariance of top individuals for rotation-aware perturbation
    diffs_cov = top_pop - centroid
    if top_k > dim and dim <= 50:
        try:
            cov = np.cov(diffs_cov, rowvar=True)
            if cov.ndim < 2:
                cov = np.eye(dim) * (np.std(population, axis=0) ** 2 + 1e-20)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except:
            sqrt_cov = np.diag(np.std(population, axis=0) + 1e-20)
    else:
        sqrt_cov = np.diag(np.std(population, axis=0) + 1e-20)
    
    # p-best for directed component
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    # r1 != i for difference vector
    r1 = self._random_indices_not_equal(n, np.arange(n))
    
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    r2 = np.array([np.random.choice([j for j in range(union_size) if j != i and j != r1[i]]) for i in range(n)])
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    # Strategy 1 (mask=True): centroid + covariance perturbation + pbest direction
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        n1 = len(idx1)
        z = np.random.standard_normal((n1, dim))
        cov_perturb = z @ sqrt_cov.T
        mutants[idx1] = (
            centroid
            + 0.5 * f_col[idx1] * (population[pbest_idx[idx1]] - population[idx1])
            + 0.5 * f_col[idx1] * cov_perturb
            + 0.3 * f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
        )
    
    # Strategy 2 (mask=False): current-to-centroid with DE difference
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        mutants[idx2] = (
            population[idx2]
            + f_col[idx2] * (centroid - population[idx2])
            + 0.5 * f_col[idx2] * (population[pbest_idx[idx2]] - union[r2[idx2]])
        )
    
    return mutants
```