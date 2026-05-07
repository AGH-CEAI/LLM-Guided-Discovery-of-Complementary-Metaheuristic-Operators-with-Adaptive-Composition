**Idea: Multi-Strategy Ensemble Mutation with Fitness-Weighted Selection**

Instead of selecting ONE mutation strategy per batch, generate candidates using multiple fundamentally different approaches (random direction, gradient-following, opposition-based, and dimension-wise), then probabilistically select based on current fitness landscape properties (variance and best/worst ratio). This ensemble approach provides broader coverage of problem types and reduces the risk of getting stuck in deceptive local optima that plague single-strategy variants.

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using multi-strategy ensemble with fitness-weighted selection."""
    np_pop = len(population)
    
    # Build ring topology indices
    indices = np.arange(np_pop)
    ring_prev = np.roll(indices, 1)
    ring_next = np.roll(indices, -1)
    
    # Get sorted indices for various selection schemes
    sorted_idx = np.argsort(fitness)
    pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
    worst_idx = sorted_idx[-1]
    
    # Compute fitness landscape properties for adaptive strategy selection
    f_min, f_max = np.min(fitness), np.max(fitness)
    f_range = max(f_max - f_min, 1e-10)
    f_variance = np.var(fitness)
    best_worst_ratio = (f_min + 1e-10) / (f_max + 1e-10)
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Efficient batch index sampling
    valid_mask = np.ones((np_pop, np_pop), dtype=bool)
    np.fill_diagonal(valid_mask, False)
    
    for i in range(np_pop):
        valid_mask[i, i] = False
    
    # Sample r1, r2, r3 for each individual
    r1 = np.array([np.random.choice(np.where(valid_mask[i])[0]) for i in range(np_pop)])
    r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                   for i in range(np_pop)])
    r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) 
                   for i in range(np_pop)])
    
    # Adaptive F with dynamic bounds based on landscape
    base_f = self.f_base * (1.0 + 0.5 * np.log1p(f_variance) / (np_pop + 1))
    f_values = base_f + self.f_adapt * np.random.randn(np_pop)
    f_values = np.clip(f_values, 0.05, 2.5)
    
    # Strategy 0: Random direction (rand/3/bin - different from rand/1)
    mutant_rand = population[r1] + f_values[:, np.newaxis] * (
        population[r2] - population[r3])
    
    # Strategy 1: Gradient-following (current-to-pbest/2 with better indices)
    pbest = population[np.random.choice(pbest_idx, np_pop)]
    mutant_gradient = population + f_values[:, np.newaxis] * (
        pbest - population[r1] + population[r2] - population[r3])
    
    # Strategy 2: Opposition-based (toward better, away from worst)
    best_idx = sorted_idx[0]
    mutant_oppose = population[r1] + f_values[:, np.newaxis] * (
        population[best_idx] - population[r2] + population[r3] - population[r1])
    
    # Strategy 3: Dimension-wise精英 (elite-guided with dimensional diversity)
    elite = population[sorted_idx[:max(1, int(np_pop * 0.05))]]  # Top 5%
    elite_choice = elite[np.random.randint(0, len(elite), size=np_pop)]
    mutant_elite = elite_choice + f_values[:, np.newaxis] * (
        population[r1] - population[r2])
    
    # Compute strategy selection probabilities based on fitness landscape
    # High variance = exploration needed, low ratio = stuck in local optima
    explore_weight = min(0.5, f_variance / (f_range + 1e-10))
    exploit_weight = 1.0 - explore_weight
    
    # Probabilities: [rand, gradient, oppose, elite]
    if best_worst_ratio < 0.1:  # Stuck - favor exploration
        probs = np.array([0.35, 0.25, 0.25, 0.15])
    elif best_worst_ratio > 0.5:  # Good progress - balance
        probs = np.array([0.2, 0.35, 0.15, 0.3])
    else:  # Normal - moderate exploration
        probs = np.array([0.3, 0.3, 0.2, 0.2])
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Generate random choices for each individual
    strategy_choices = np.random.choice(4, size=np_pop, p=probs)
    
    # Assemble final mutants based on strategy choices
    for i in range(np_pop):
        if strategy_choices[i] == 0:
            mutants[i] = mutant_rand[i]
        elif strategy_choices[i] == 1:
            mutants[i] = mutant_gradient[i]
        elif strategy_choices[i] == 2:
            mutants[i] = mutant_oppose[i]
        else:
            mutants[i] = mutant_elite[i]
    
    # Clip to bounds with numerical safety
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    # Handle any NaN/Inf from numerical issues
    nan_mask = np.any(np.isnan(mutants) | np.isinf(mutants), axis=1)
    if np.any(nan_mask):
        mutants[nan_mask] = population[nan_mask] + np.random.uniform(
            -0.1, 0.1, (np.sum(nan_mask), self.dim))
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```