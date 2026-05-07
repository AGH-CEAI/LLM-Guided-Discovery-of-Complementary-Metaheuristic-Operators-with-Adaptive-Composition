**Idea: Ring-Guided Adaptive Mutation**

This strategy actively uses the ring topology (currently computed but unused) to create mutations along the "valley" direction between ring neighbors, combined with adaptive weighting. For the worst unsolved tasks (errors 1e+1 to 1e+3), this provides structured diversity that escapes local optima by moving individuals toward better ring neighbors while maintaining exploration via standard DE components.

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Ring-Guided Adaptive Mutation with structured neighborhood exploration."""
    np_pop = len(population)
    
    # Ring topology indices (now actively used!)
    indices = np.arange(np_pop)
    ring_prev = np.roll(indices, 1)
    ring_next = np.roll(indices, -1)
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Sample indices for mutation (excluding current)
    r1 = np.array([np.random.choice(np.delete(np.arange(np_pop), i)) for i in range(np_pop)])
    r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) for i in range(np_pop)])
    r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) for i in range(np_pop)])
    
    # Adaptive F values
    f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
    f_values = np.clip(f_values, 0.3, 1.5)
    
    # For each individual, compute ring-guided direction
    for i in range(np_pop):
        # Compare fitness of ring neighbors
        f_curr = fitness[i]
        f_prev = fitness[ring_prev[i]]
        f_next = fitness[ring_next[i]]
        
        # Direction toward better neighbor (valley climbing)
        if f_prev <= f_next and f_prev < f_curr:
            # Move toward previous neighbor
            ring_dir = population[ring_prev[i]] - population[i]
            ring_weight = 0.5
        elif f_next < f_prev and f_next < f_curr:
            # Move toward next neighbor
            ring_dir = population[ring_next[i]] - population[i]
            ring_weight = 0.5
        else:
            # Current is best among ring - use inter-neighbor direction for exploration
            ring_dir = population[ring_next[i]] - population[ring_prev[i]]
            ring_weight = 0.3
        
        # Apply mutation based on selected operator
        if selected_operator == 0:  # rand/1 with ring guidance
            base = population[r1[i]]
            de_vec = population[r2[i]] - population[r3[i]]
            mutants[i] = base + f_values[i] * de_vec + ring_weight * ring_dir
            
        elif selected_operator == 1:  # current-to-ring-best with ring guidance
            base = population[i]
            de_vec = population[r1[i]] - population[r2[i]]
            # Move toward better of two ring neighbors
            if f_prev <= f_next:
                target = population[ring_prev[i]]
            else:
                target = population[ring_next[i]]
            mutants[i] = base + f_values[i] * de_vec + 0.7 * (target - base) + 0.3 * ring_dir
            
        elif selected_operator == 2:  # ring-assisted current-to-pbest
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[0]
            base = population[i]
            pbest_vec = population[pbest_idx] - population[i]
            de_vec = population[r1[i]] - population[r2[i]]
            mutants[i] = base + f_values[i] * de_vec + 0.5 * pbest_vec + 0.2 * ring_dir
            
        else:  # ring-guided rand-to-best
            best_idx = np.argmin(fitness)
            base = population[r1[i]]
            best_vec = population[best_idx] - population[r1[i]]
            de_vec = population[r2[i]] - population[r3[i]]
            mutants[i] = base + f_values[i] * de_vec + 0.4 * best_vec + 0.3 * ring_dir
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```