Looking at the error patterns, the worst unsolved tasks (17, 16, 11, 19, 23 with errors 10^1 to 10^3) suggest the current mutation operators are failing to escape local optima and explore the search space effectively. The current approach uses uniform random sampling for r1/r2 selection, which may not provide sufficient directional guidance toward better regions.

**Idea: Distance-Weighted Elite-Driven Mutation**
Instead of uniform random sampling for mutation vectors, this approach selects candidates based on their distance from elite individuals, weighted by inverse distance. This creates a "gravity toward excellence" effect that pushes mutations toward promising regions while maintaining diversity through probabilistic sampling. Uses multiple adaptive F values and optional mixing with global best for balance between exploitation and exploration.

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using distance-weighted elite-driven mutation."""
    np_pop = len(population)
    
    # Get sorted indices by fitness for elite selection
    sorted_idx = np.argsort(fitness)
    n_elite = max(3, int(np_pop * 0.15))  # Top 15% as elite
    elite_idx = sorted_idx[:n_elite]
    
    # Get best individual
    best_idx = sorted_idx[0]
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Compute pairwise distances from each individual to elite members
    # Use sum of distances as diversity measure
    elite_pop = population[elite_idx]  # (n_elite, dim)
    
    # Vectorized distance computation from all individuals to all elite
    diff = population[:, np.newaxis, :] - elite_pop[np.newaxis, :, :]  # (np_pop, n_elite, dim)
    dists_to_elite = np.sqrt(np.sum(diff ** 2, axis=2))  # (np_pop, n_elite)
    sum_dists = np.sum(dists_to_elite, axis=1) + 1e-10  # (np_pop,)
    
    # Distance-weighted selection for r1: prefer individuals far from elite (more diverse)
    # Weight = sum of distances to elite members
    weights = sum_dists / (np.sum(sum_dists) + 1e-10)
    r1 = np.random.choice(np_pop, size=np_pop, replace=True, p=weights)
    
    # For r2, use uniform random (standard approach, maintains some diversity)
    r2 = np.random.choice(np_pop, size=np_pop, replace=True)
    r3 = np.random.choice(np_pop, size=np_pop, replace=True)
    
    # Adaptive F with multiple scales for diverse exploration
    f_diverse = np.random.uniform(0.4, 1.2, size=np_pop)  # Wider range
    f_small = np.clip(self.f_base + self.f_adapt * np.random.randn(np_pop), 0.1, 2.0)
    f_large = np.clip(self.f_base * 1.5 + self.f_adapt * 0.3 * np.random.randn(np_pop), 0.3, 2.5)
    
    # Apply mutation based on selected operator
    if selected_operator == 0:  # rand/1 with distance-weighted base
        mutants = (population[r1] + f_diverse[:, np.newaxis] * 
                  (population[r2] - population[r3]))
    elif selected_operator == 1:  # best/1 with elite diversity
        # Mix best with distance-weighted perturbation
        mix = np.random.rand(np_pop, 1)
        mutants = (population[best_idx] * (1 - mix) + 
                  population[r1] * mix + 
                  f_small[:, np.newaxis] * (population[r2] - population[r3]))
    elif selected_operator == 2:  # current-to-pbest/1 with distance weighting
        # Sample pbest from elite
        elite_sample = np.random.choice(n_elite, size=np_pop)
        pbest = elite_pop[elite_sample]
        mutants = (population + f_small[:, np.newaxis] * 
                  (pbest - population[r1] + population[r2] - population[r3]))
    else:  # rand-to-best/1 with multi-scale
        # Blend multiple scaling factors
        mutants = (population[r1] + f_small[:, np.newaxis] * 
                   (population[best_idx] - population[r1]) + 
                   f_large[:, np.newaxis] * (population[r2] - population[r3]))
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```