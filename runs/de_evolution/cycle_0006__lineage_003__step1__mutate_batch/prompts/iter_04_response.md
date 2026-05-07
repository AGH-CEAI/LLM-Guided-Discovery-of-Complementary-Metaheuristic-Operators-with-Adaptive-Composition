Looking at the unsolved tasks, I see errors ranging from ~1e+2 down to ~1e-8, suggesting different failure modes:

1. **Tasks 16, 23, 20, 19** (errors 10-100+): Likely highly multimodal/compositional functions where the algorithm gets trapped in local optima. These need much stronger exploration with diverse search directions.
2. **Tasks 11-22** (errors 1-10): Moderately difficult, possibly ill-conditioned or rotated functions needing better directional information.
3. **Tasks 6, 7, 12, 3, 9** (errors 1e-8 to 1e-4): Close to convergence but need fine-tuning precision.

The existing variants use current-to-pbest/1 and rand-to-pbest/1 — both are greedy toward pbest. For highly multimodal functions, this creates premature convergence. I'll implement a **multi-strategy approach** combining: (1) **rand/2** for maximum exploration on the hardest tasks, (2) **best/2** with weighted differences for intermediate tasks, and (3) a **rank-weighted directional mutation** that uses fitness-rank-based weighting of multiple difference vectors to create more informed search directions — fundamentally different from simple pbest-guided mutations.

**Idea: Rank-Weighted Multi-Donor Mutation with rand/2 Exploration**
Uses rank-weighted combinations of multiple difference vectors plus a rand/2 strategy for strong exploration on multimodal landscapes.
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    n = len(population)
    dim = self.dim
    f_col = f_values[:, np.newaxis]
    
    sorted_idx = np.argsort(fitness)
    ranks = np.empty(n, dtype=float)
    ranks[sorted_idx] = np.arange(n, dtype=float)
    # Weights: better rank = higher weight (linear)
    weights = (n - ranks) / n  # best=1.0, worst~=0.0
    
    mutants = np.empty((n, dim))
    
    # Strategy 1 (mask=True): Rank-weighted multi-donor directional mutation
    # Pick 4 random distinct donors, form weighted combination
    mask1 = strategy_mask
    if np.any(mask1):
        idx1 = np.where(mask1)[0]
        ni = len(idx1)
        
        # Pick 5 distinct random indices per individual
        donors = np.zeros((ni, 5), dtype=int)
        for k in range(5):
            candidates = np.random.randint(0, n, size=ni)
            # Simple retry to avoid self
            for _ in range(10):
                bad = candidates == idx1
                if not np.any(bad):
                    break
                candidates[bad] = np.random.randint(0, n, size=np.sum(bad))
            donors[:, k] = candidates
        
        # Rank-weighted centroid of donors
        donor_weights = weights[donors]  # (ni, 5)
        donor_weights_sum = donor_weights.sum(axis=1, keepdims=True) + 1e-30
        donor_weights_norm = donor_weights / donor_weights_sum  # (ni, 5)
        
        # Weighted centroid
        donor_vecs = population[donors]  # (ni, 5, dim)
        centroid = np.einsum('ij,ijk->ik', donor_weights_norm, donor_vecs)  # (ni, dim)
        
        # Best donor direction + random perturbation
        best_donor_idx = np.argmax(donor_weights, axis=1)
        best_donor = population[donors[np.arange(ni), best_donor_idx]]
        worst_donor_idx = np.argmin(donor_weights, axis=1)
        worst_donor = population[donors[np.arange(ni), worst_donor_idx]]
        
        mutants[idx1] = centroid + f_col[idx1] * (best_donor - worst_donor)
    
    # Strategy 2 (mask=False): rand/2 — strong exploration
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        ni2 = len(idx2)
        
        # Pick 5 distinct random individuals
        rs = np.zeros((ni2, 5), dtype=int)
        for k in range(5):
            rs[:, k] = np.random.randint(0, n, size=ni2)
        
        # rand/2: x_r0 + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)
        mutants[idx2] = (
            population[rs[:, 0]]
            + f_col[idx2] * (population[rs[:, 1]] - population[rs[:, 2]])
            + f_col[idx2] * (population[rs[:, 3]] - population[rs[:, 4]])
        )
    
    return mutants
```