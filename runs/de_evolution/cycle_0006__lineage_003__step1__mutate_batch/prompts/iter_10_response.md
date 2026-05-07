Looking at the unsolved tasks, I see errors stuck at magnitudes 1e+0 to 1e+2, suggesting these are likely multimodal, non-separable, ill-conditioned functions (like Schwefel, Katsuura, Lunacek, composition functions in CEC benchmarks). The existing variants all use pbest-based mutation which converges prematurely on these landscapes.

**Key insight:** For the hardest tasks (16, 23, 20, 19, 21), errors are stuck at 1-100+, meaning the search is trapped in wrong basins entirely. We need a mutation strategy that:
1. Uses **weighted difference vectors from multiple pairs** to better approximate the fitness landscape gradient
2. Incorporates **rank-based weighted recombination** (like CMA-ES) to exploit population structure
3. Adds **dimension-wise perturbation with adaptive step sizes** to break through narrow valleys

The fundamental difference from all prior variants: instead of simple `pbest - current` differences, this uses a **weighted centroid of top individuals minus weighted centroid of bottom individuals** as the primary search direction, combined with orthogonal random perturbation for escaping ridges.

**Idea: Weighted Centroid Directional Mutation with Orthogonal Perturbation**
Uses fitness-weighted centroid differences and orthogonal random perturbations to navigate complex multimodal landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    
    # Compute weighted centroids of top and bottom halves
    n_top = max(2, n // 4)
    n_bot = max(2, n // 4)
    
    top_idx = sorted_idx[:n_top]
    bot_idx = sorted_idx[-n_bot:]
    
    # Fitness-based weights for top individuals (better = higher weight)
    top_fit = fitness[top_idx]
    fit_range = np.max(top_fit) - np.min(top_fit) + 1e-30
    top_weights = (np.max(top_fit) - top_fit) / fit_range + 0.1
    top_weights /= np.sum(top_weights)
    
    centroid_top = np.sum(population[top_idx] * top_weights[:, np.newaxis], axis=0)
    centroid_bot = np.mean(population[bot_idx], axis=0)
    
    # Primary direction: from bad centroid toward good centroid
    direction = centroid_top - centroid_bot
    dir_norm = np.linalg.norm(direction) + 1e-30
    direction_unit = direction / dir_norm
    
    f_col = f_values[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
    
    r1 = self._random_indices_not_equal(n, np.arange(n))
    r2 = self._random_indices_not_equal(n, r1)
    # Ensure r2 != i
    clash = r2 == np.arange(n)
    r2[clash] = (r2[clash] + 1) % n
    
    for i in range(n):
        if strategy_mask[i]:
            # Weighted centroid direction + pbest attraction + random diff
            rand_perturb = np.random.randn(dim)
            # Remove component along direction to get orthogonal perturbation
            proj = np.dot(rand_perturb, direction_unit) * direction_unit
            ortho_perturb = rand_perturb - proj
            ortho_norm = np.linalg.norm(ortho_perturb) + 1e-30
            ortho_perturb = ortho_perturb / ortho_norm * dir_norm * 0.3
            
            mutants[i] = (
                population[i]
                + f_values[i] * 0.5 * (centroid_top - population[i])
                + f_values[i] * 0.5 * (population[pbest_idx[i]] - population[i])
                + f_values[i] * 0.3 * ortho_perturb
            )
        else:
            # Rank-weighted mutation: use multiple difference vectors
            # Sample 3 random distinct individuals weighted by fitness rank
            ranks = np.zeros(n)
            ranks[sorted_idx] = np.arange(n, dtype=float)
            weights = np.exp(-ranks / (n / 3.0))
            weights[i] = 0
            weights /= np.sum(weights) + 1e-30
            
            chosen = np.random.choice(n, size=3, replace=False, p=weights)
            
            mutants[i] = (
                population[chosen[0]]
                + f_values[i] * (population[pbest_idx[i]] - population[chosen[0]])
                + f_values[i] * (population[chosen[1]] - population[chosen[2]])
                + f_values[i] * 0.1 * np.random.randn(dim)
            )
    
    return mutants
```