Looking at the unsolved tasks, the errors are stuck at very high values (1e+1 to 1e+2), suggesting the algorithm gets trapped in local optima on multimodal, non-separable, ill-conditioned landscapes. The existing variants all use pbest-guided mutations which converge prematurely on deceptive landscapes.

**Key analysis:**
- Tasks 16, 20, 23 have errors ~10-100: severe multimodality/deceptive landscapes needing much more exploration
- Tasks 11-22 stuck at ~1-10: likely ill-conditioned + multimodal, needing both exploration and exploitation
- Current variants all bias toward pbest, causing premature convergence

**My approach:** A multi-strategy mutation combining (1) weighted recombination of multiple donors for better exploration of the landscape, (2) rank-based differential weighting where worse individuals get larger perturbations, and (3) a "directed mutation" component using the centroid of top individuals. This fundamentally differs from prior pbest-based approaches by using ensemble-based direction estimation and fitness-proportional step sizes.

**Idea: Rank-Weighted Multi-Donor Ensemble Mutation**
Uses rank-based adaptive scaling with multi-donor differential vectors and centroid-directed search for escaping deep local optima on multimodal landscapes.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    
    sorted_idx = np.argsort(fitness)
    ranks = np.empty(n, dtype=float)
    ranks[sorted_idx] = np.arange(n, dtype=float)
    
    # Rank-based scaling: worse individuals explore more
    rank_scale = 0.5 + 1.5 * (ranks / max(n - 1, 1))  # [0.5, 2.0]
    
    p = max(2, int(np.ceil(self.p_best_rate * n)))
    top_p = sorted_idx[:p]
    centroid_top = np.mean(population[top_p], axis=0)
    
    # Union for donor selection
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    f_col = f_values[:, np.newaxis]
    rank_col = rank_scale[:, np.newaxis]
    mutants = np.empty((n, dim))
    
    for i in range(n):
        f_i = f_values[i]
        
        if strategy_mask[i]:
            # Strategy 1: Centroid-directed + dual differential vectors
            # Pick two pairs of donors
            candidates = list(range(n))
            candidates.remove(i)
            np.random.shuffle(candidates)
            r1, r2, r3, r4 = candidates[0], candidates[1], candidates[2 % len(candidates)], candidates[3 % len(candidates)]
            
            r5 = np.random.randint(0, union_size)
            while r5 == i:
                r5 = np.random.randint(0, union_size)
            
            # Weighted combination: centroid direction + two diff vectors
            w_centroid = 0.5 * f_i * rank_scale[i]
            w_diff1 = 0.5 * f_i
            w_diff2 = 0.3 * f_i
            
            mutants[i] = (population[i] 
                         + w_centroid * (centroid_top - population[i])
                         + w_diff1 * (population[r1] - population[r2])
                         + w_diff2 * (population[r3] - union[r5]))
        else:
            # Strategy 2: Random base with aggressive exploration
            # Pick random base from top half
            top_half = max(3, n // 2)
            base_idx = sorted_idx[np.random.randint(0, top_half)]
            
            candidates = list(range(n))
            candidates.remove(i)
            if base_idx in candidates:
                candidates.remove(base_idx)
            np.random.shuffle(candidates)
            r1 = candidates[0]
            r2 = candidates[1 % len(candidates)]
            
            r3 = np.random.randint(0, union_size)
            while r3 == i or r3 == r1:
                r3 = np.random.randint(0, union_size)
            
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            scale = f_i * rank_scale[i]
            mutants[i] = (population[base_idx]
                         + 0.7 * f_i * (population[pbest_idx] - population[base_idx])
                         + scale * (population[r1] - union[r3])
                         + 0.1 * np.random.randn(dim) * (rank_scale[i] ** 2))
    
    return mutants
```