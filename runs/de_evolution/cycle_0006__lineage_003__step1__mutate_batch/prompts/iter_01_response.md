**Idea: Weighted Current-to-PBest with Rank-Based F Scaling**
Enhanced current-to-pbest/1 mutation where F is scaled by fitness rank (worse individuals get larger F for more exploration) and p-best rate adapts per individual based on generation progress.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    """Generate mutant vectors using rank-weighted current-to-pbest/1 with adaptive p."""
    n = len(population)
    dim = self.dim
    
    # Sort by fitness for p-best selection and ranking
    sorted_idx = np.argsort(fitness)
    
    # Rank-based F scaling: worse individuals use larger F values
    ranks = np.empty(n, dtype=int)
    ranks[sorted_idx] = np.arange(n)  # rank 0 = best, rank n-1 = worst
    rank_ratio = ranks / max(n - 1, 1)  # 0 to 1
    
    # Scale F: best individuals use smaller F (exploitation), worst use larger (exploration)
    f_scaled = f_values * (0.5 + 0.5 * rank_ratio)
    f_scaled = np.clip(f_scaled, 0.01, 1.0)
    
    # Adaptive p-best rate per individual: top individuals use smaller p (more greedy)
    # bottom individuals use larger p (more diverse targets)
    generation = getattr(self, 'generation', 0)
    progress = min(1.0, generation / 500.0)
    p_min = max(2, int(np.ceil(0.05 * n)))
    p_max = max(2, int(np.ceil(0.25 * n)))
    
    # Per-individual p values
    p_values = np.clip(
        (p_min + (p_max - p_min) * rank_ratio * (1.0 - 0.5 * progress)).astype(int),
        2, n
    )
    
    # Select p-best indices for each individual with their own p
    pbest_idx = np.array([sorted_idx[np.random.randint(0, p_values[i])] for i in range(n)])
    
    # Select r1 != i
    r1 = self._random_indices_not_equal(n, np.arange(n))
    
    # Union of population and archive for r2
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)
    
    # Select r2 from union, r2 != i and r2 != r1
    r2 = np.zeros(n, dtype=int)
    for i in range(n):
        attempts = 0
        while attempts < 100:
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r1[i]:
                r2[i] = idx
                break
            attempts += 1
        else:
            r2[i] = np.random.randint(0, union_size)
    
    f_col = f_scaled[:, np.newaxis]
    
    # Weighted current-to-pbest/1 for all individuals
    # v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2_union)
    mutants = (
        population
        + f_col * (population[pbest_idx] - population)
        + f_col * (population[r1] - union[r2])
    )
    
    # For strategy_mask == False, add a second difference vector for more exploration
    mask2 = ~strategy_mask
    if np.any(mask2):
        idx2 = np.where(mask2)[0]
        r3 = self._random_indices_not_equal(n, np.arange(n))[idx2]
        r4 = self._random_indices_not_equal(n, r1)[idx2]
        # Add weighted second difference vector (smaller weight)
        weight2 = 0.5 * f_col[idx2]
        mutants[idx2] += weight2 * (population[r3] - population[r4])
    
    return mutants
```