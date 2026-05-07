Looking at the unsolved tasks, I see errors stuck at ~1e+1 to ~1e+2 for tasks 16, 20, 23, 19, 21, 17, 18, 14 — these are likely high-dimensional, multimodal, or ill-conditioned functions where standard DE gets trapped in local optima. The existing variants all use pbest-based strategies which converge fast but get stuck in basins.

The key insight: for these hard tasks, we need **much more aggressive exploration** with multiple differential vectors, **rank-based weighted recombination** (like CMA-ES), and **eigenspace-aware perturbation** to handle rotated/ill-conditioned landscapes. I'll implement a weighted mutation that combines information from multiple top individuals (not just one pbest) with an additional exploration term using the population's covariance structure.

**Idea: Eigenvector-Guided Multi-Donor Weighted Mutation**
Uses principal component directions from top individuals for rotation-invariant exploration, plus weighted multi-donor recombination for better landscape modeling on ill-conditioned/multimodal problems.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    
    # Weighted centroid of top individuals (like CMA-ES)
    n_top = max(3, n // 4)
    top_idx = sorted_idx[:n_top]
    top_pop = population[top_idx]
    
    # Log-linear weights for weighted recombination
    weights = np.log(n_top + 0.5) - np.log(np.arange(1, n_top + 1))
    weights = weights / np.sum(weights)
    weighted_mean = np.sum(weights[:, np.newaxis] * top_pop, axis=0)
    
    # Compute covariance-based directions for rotation invariance
    if dim <= 100 and n_top >= dim // 2:
        diffs = top_pop - weighted_mean
        cov = np.dot((weights[:, np.newaxis] * diffs).T, diffs)
        cov += 1e-10 * np.eye(dim)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except:
            sqrt_cov = None
    else:
        sqrt_cov = None
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    r1 = self._random_indices_not_equal(n, np.arange(n))
    
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    r2 = np.array([np.random.choice([j for j in range(union_size) if j != i and j != r1[i]]) for i in range(n)])
    
    for i in range(n):
        if strategy_mask[i]:
            # Strategy 1: Weighted mean guidance + eigenvector perturbation
            base = population[i] + f_values[i] * (weighted_mean - population[i])
            diff = f_values[i] * (population[r1[i]] - union[r2[i]])
            
            if sqrt_cov is not None and np.random.random() < 0.4:
                z = np.random.standard_normal(dim)
                eigen_perturb = 0.1 * f_values[i] * sqrt_cov @ z
                mutants[i] = base + diff + eigen_perturb
            else:
                mutants[i] = base + diff
        else:
            # Strategy 2: current-to-pbest with double difference vectors
            r3 = np.random.randint(0, n)
            while r3 == i or r3 == r1[i]:
                r3 = np.random.randint(0, n)
            r4 = np.random.randint(0, union_size)
            while r4 == i or r4 == r1[i] or r4 == r2[i]:
                r4 = np.random.randint(0, union_size)
            
            mutants[i] = (
                population[i]
                + f_values[i] * (population[pbest_idx[i]] - population[i])
                + f_values[i] * (population[r1[i]] - union[r2[i]])
                + 0.5 * f_values[i] * (population[r3] - union[r4])
            )
    
    return mutants
```