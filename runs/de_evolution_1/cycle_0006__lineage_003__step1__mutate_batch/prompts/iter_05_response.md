Looking at the unsolved tasks, I see errors ranging from ~1e+2 down to ~1e-8. The worst tasks (16, 23, 20, 19, 21, 17, 18, 14) have errors stuck at 1-100+, suggesting they're highly multimodal, non-separable, and/or ill-conditioned functions where the current strategies get trapped in local optima. The current variants use only pbest-directed mutations, which converge greedily toward the current best region. For these hard multimodal/compositional functions, we need:

1. **Rank-based weighted recombination** (like CMA-ES) to exploit fitness landscape structure
2. **Multiple difference vectors** for better exploration of complex landscapes  
3. **Eigenvector-based rotation** to handle non-separable, ill-conditioned functions
4. **Larger perturbations** with adaptive scaling to escape deep local optima

The key insight: for Tasks 16-23 (likely composition/hybrid functions), the search needs to explore broadly while still being directed. I'll implement a mutation that combines eigenvector-adapted difference vectors with a weighted centroid direction, fundamentally different from the pbest-directed approaches used so far.

**Idea: Eigenvector-Adapted Weighted Centroid Mutation**
Uses covariance-based rotation of difference vectors plus fitness-weighted centroid direction for non-separable multimodal landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    """Eigenvector-adapted mutation with weighted centroid direction."""
    n = len(population)
    dim = self.dim
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    # Compute fitness-weighted centroid (better than pbest for multimodal)
    sorted_idx = np.argsort(fitness)
    n_top = max(3, n // 3)
    top_idx = sorted_idx[:n_top]
    top_pop = population[top_idx]
    top_fit = fitness[top_idx]
    
    # Weights based on rank (not fitness value - more robust)
    weights = np.log(n_top + 0.5) - np.log(np.arange(1, n_top + 1))
    weights /= np.sum(weights)
    centroid = np.sum(weights[:, np.newaxis] * top_pop, axis=0)
    
    # Try eigenvector rotation if dim is manageable
    use_eigen = dim <= 100 and n >= dim + 1
    if use_eigen:
        try:
            cov = np.cov(population.T) + 1e-10 * np.eye(dim)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            # Scaling matrix: sqrt of eigenvalues for isotropic sampling in eigen-space
            D = np.sqrt(eigenvalues)
        except:
            use_eigen = False
    
    # Union for difference vectors
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    for i in range(n):
        F = f_values[i]
        
        if strategy_mask[i]:
            # Strategy 1: Centroid-directed + eigenvector-rotated perturbation
            direction = centroid - population[i]
            
            # Two random difference vectors from union
            candidates = list(range(n))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            r3 = np.random.randint(0, union_size)
            while r3 == i:
                r3 = np.random.randint(0, union_size)
            
            diff = population[r1] - union[r3]
            
            if use_eigen:
                # Rotate difference vector through eigenspace for better exploration
                diff_eigen = eigenvectors.T @ diff
                diff_eigen *= (D / (np.mean(D) + 1e-30))  # Normalize by mean scale
                diff = eigenvectors @ diff_eigen
            
            mutants[i] = population[i] + F * direction + 0.5 * F * diff
        else:
            # Strategy 2: rand/2 with centroid bias for broader exploration
            candidates = list(range(n))
            candidates.remove(i)
            chosen = np.random.choice(candidates, 4, replace=False)
            r1, r2, r3, r4 = chosen
            
            base = 0.5 * (population[r1] + centroid)  # Blend random with centroid
            mutants[i] = base + F * (population[r2] - population[r3]) + 0.5 * F * (population[r4] - population[i])
    
    return mutants
```