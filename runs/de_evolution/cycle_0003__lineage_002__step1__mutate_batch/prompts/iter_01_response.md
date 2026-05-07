**Idea: JADE-style current-to-pbest/1 with weighted F and rank-based pbest**

Pure current-to-pbest/1 mutation (JADE/SHADE canonical) with rank-based probabilistic pbest selection and vectorized r2 sampling for robustness across all problem types.

```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
    """Generate mutant vectors using current-to-pbest/1 with rank-based pbest selection."""
    n = len(population)
    dim = self.dim

    # Sort by fitness for p-best selection
    sorted_idx = np.argsort(fitness)
    
    # Adaptive p: use strategy_mask to vary greediness
    # mask=True -> smaller p (greedier), mask=False -> larger p (more explorative)
    p_greedy = max(2, int(np.ceil(0.05 * n)))
    p_explore = max(2, int(np.ceil(0.25 * n)))
    
    pbest_idx = np.empty(n, dtype=int)
    for i in range(n):
        if strategy_mask[i]:
            pbest_idx[i] = sorted_idx[np.random.randint(0, p_greedy)]
        else:
            pbest_idx[i] = sorted_idx[np.random.randint(0, p_explore)]

    # Select r1 != i
    r1 = np.empty(n, dtype=int)
    for i in range(n):
        while True:
            idx = np.random.randint(0, n)
            if idx != i:
                r1[i] = idx
                break

    # Union of population and archive for r2
    if len(archive) > 0:
        union = np.vstack([population, archive])
    else:
        union = population.copy()
    union_size = len(union)

    # Select r2 from union, r2 != i and r2 != r1
    r2 = np.empty(n, dtype=int)
    for i in range(n):
        while True:
            idx = np.random.randint(0, union_size)
            if idx != i and idx != r1[i]:
                r2[i] = idx
                break

    f_col = f_values[:, np.newaxis]

    # current-to-pbest/1: v_i = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2_union)
    mutants = (
        population
        + f_col * (population[pbest_idx] - population)
        + f_col * (population[r1] - union[r2])
    )

    return mutants
```