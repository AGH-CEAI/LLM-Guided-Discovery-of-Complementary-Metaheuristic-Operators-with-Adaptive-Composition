Looking at the unsolved tasks, I notice the worst ones (Tasks 16, 20, 23, 19, 21, 17, 18, 14) have errors in the range 1-100+, suggesting these are likely high-dimensional, multimodal, or ill-conditioned functions where the algorithm gets trapped in local optima. The current variants all use pbest-directed mutations which can converge prematurely on deceptive landscapes.

The key insight: existing variants all bias toward the current best solutions (pbest), which on multimodal/deceptive landscapes leads to premature convergence. For the hardest tasks, we need a fundamentally different approach that combines:

1. **Rank-weighted centroid mutation** - instead of targeting a single pbest, use a weighted centroid of top solutions to get a smoother gradient signal
2. **Eigenvector-based rotation** - learn the covariance structure and mutate along principal axes (CMA-ES inspired), crucial for non-separable rotated functions
3. **Multi-donor differential vectors** - use 2 difference vectors instead of 1 for better exploration

This is fundamentally different from all prior variants which use standard DE/current-to-pbest mutations without any covariance adaptation.

**Idea: Eigenvector-Guided Weighted Centroid Mutation with Double Differences**
Uses covariance-adapted mutation along principal axes with rank-weighted centroid targeting and dual difference vectors for non-separable multimodal landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    sorted_idx = np.argsort(fitness)
    
    # Compute rank-weighted centroid of top half
    top_k = max(3, n // 3)
    top_pop = population[sorted_idx[:top_k]]
    # Rank weights: best gets highest weight
    weights = np.log(top_k + 0.5) - np.log(np.arange(1, top_k + 1))
    weights /= weights.sum()
    centroid = weights @ top_pop
    
    # Compute covariance and eigenvectors for rotation-aware mutation
    try:
        diff = top_pop - centroid
        cov = (diff * weights[:, np.newaxis]).T @ diff
        cov += 1e-10 * np.eye(dim)  # regularize
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-20)
        # Scaling matrix: sqrt of eigenvalues
        D = np.sqrt(eigvals)
    except:
        eigvecs = np.eye(dim)
        D = np.ones(dim)
    
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    # Generate 3 distinct random indices per individual
    r1 = self._random_indices_not_equal(n, np.arange(n))
    r2 = self._random_indices_not_equal(n, r1)
    # Ensure r2 != i
    for i in range(n):
        while r2[i] == i:
            r2[i] = np.random.randint(0, n)
    r3 = self._random_indices_not_equal(n, r2)
    for i in range(n):
        while r3[i] == i or r3[i] == r1[i]:
            r3[i] = np.random.randint(0, n)
    
    # Strategy 1 (mask=True): Eigenvector-rotated centroid mutation
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        # Direction toward centroid + eigenvector-rotated difference
        diff_vec = population[r1[idx1]] - population[r2[idx1]]
        # Rotate difference vector through eigenbasis
        rotated = (diff_vec @ eigvecs) * (D / (D.mean() + 1e-30))
        rotated = rotated @ eigvecs.T
        
        mutants[idx1] = (
            population[idx1]
            + f_col[idx1] * (centroid - population[idx1])
            + 0.5 * f_col[idx1] * rotated
        )
    
    # Strategy 2 (mask=False): pbest with double difference vectors
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        mutants[idx2] = (
            population[idx2]
            + f_col[idx2] * (population[pbest_idx[idx2]] - population[idx2])
            + 0.5 * f_col[idx2] * (population[r1[idx2]] - population[r2[idx2]])
            + 0.5 * f_col[idx2] * (population[r2[idx2]] - population[r3[idx2]])
        )
    
    return mutants
```