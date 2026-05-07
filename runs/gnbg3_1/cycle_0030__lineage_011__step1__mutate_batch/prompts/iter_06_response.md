**Idea: Trigonometric Mutation with Rank-Based Diversity Injection**

All current variants use linear combinations of population vectors (DE-style mutations), which get trapped in deceptive local optima on the worst tasks (17, 16, 11, 19, 23 with errors >40). This operator replaces the linear mutation with a non-linear trigonometric perturbation that exploits rank-order information and injects diversity through sinusoidal exploration — fundamentally different from any linear combination approach.

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using trigonometric mutation with diversity injection."""
    np_pop = len(population)
    dim = self.dim
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Get global statistics
    sorted_idx = np.argsort(fitness)
    best_idx = sorted_idx[0]
    worst_idx = sorted_idx[-1]
    
    # Compute rank-based weights (higher rank = better fitness = higher weight)
    ranks = np.zeros(np_pop)
    ranks[sorted_idx] = np.arange(1, np_pop + 1)
    rank_weights = ranks / ranks.sum()
    
    # Compute weighted centroid (exploits rank information)
    centroid = np.sum(population * rank_weights[:, np.newaxis], axis=0)
    
    # Compute population statistics for adaptive scaling
    pop_std = np.std(population, axis=0) + 1e-10
    pop_mean = np.mean(population, axis=0)
    
    # Sample random indices for base vectors
    all_indices = np.arange(np_pop)
    r1 = np.array([np.random.choice(np.delete(all_indices, i)) for i in range(np_pop)])
    r2 = np.array([np.random.choice(np.delete(all_indices, [i, r1[i]])) for i in range(np_pop)])
    r3 = np.array([np.random.choice(np.delete(all_indices, [i, r1[i], r2[i]])) for i in range(np_pop)])
    
    # Adaptive F based on diversity and fitness spread
    fitness_spread = np.max(fitness) - np.min(fitness) + 1e-10
    diversity_ratio = self._compute_diversity(population) / (self.upper_bound - self.lower_bound + 1e-10)
    f_base = np.clip(0.3 + 0.4 * diversity_ratio, 0.1, 1.5)
    f_values = f_base * (0.8 + 0.4 * np.random.rand(np_pop))
    
    # Get base vectors
    p1 = population[r1]
    p2 = population[r2]
    p3 = population[r3]
    
    # Use different strategies based on selected_operator
    if selected_operator == 0:  # Trigonometric mutation (non-linear, non-convex)
        # Trigonometric mutation: exploits sin/cos periodicity for diverse exploration
        pbest = population[sorted_idx[:max(1, int(np_pop * 0.05))]]  # Top 5%
        pbest_sample = pbest[np.random.randint(0, len(pbest), np_pop)]
        
        # Compute trigonometric perturbation
        sin_term = np.sin(np.pi * (pbest_sample - p1))
        cos_term = np.cos(np.pi * (p2 - p3))
        
        # Blend with linear term for robustness
        linear_term = f_values[:, np.newaxis] * (p2 - p3)
        trig_term = f_values[:, np.newaxis] * 0.3 * (sin_term + cos_term)
        
        mutants = p1 + linear_term + trig_term
        
    elif selected_operator == 1:  # Centroid-guided with rank weighting
        # Pull toward centroid weighted by fitness ranks
        centroid_offset = centroid - p1
        direction = p2 - p3
        
        mutants = p1 + f_values[:, np.newaxis] * direction + 0.5 * f_values[:, np.newaxis] * centroid_offset
        
    elif selected_operator == 2:  # Adaptive scale mutation (multi-scale exploration)
        # Mix fine-scale and coarse-scale perturbations
        fine_scale = f_values[:, np.newaxis] * (p2 - p3) * 0.5
        coarse_scale = (f_values[:, np.newaxis] * 2.0 * (pop_mean - p1) * 
                        (1.0 + 0.5 * np.random.rand(np_pop, 1)))
        
        # Randomly select scale for each individual
        scale_mask = np.random.rand(np_pop, 1) > 0.5
        mutants = p1 + np.where(scale_mask, fine_scale + coarse_scale, fine_scale)
        
    else:  # Diversity-focused mutation with sinusoidal perturbation
        # Inject diversity using sinusoidal functions of population statistics
        phase = 2 * np.pi * np.random.rand(np_pop, 1)
        sin_perturbation = pop_std * np.sin(phase + 0.1 * np.arange(dim) * np.pi)
        
        # Combine with standard DE mutation
        de_term = f_values[:, np.newaxis] * (p2 - p3)
        diversity_term = 0.3 * sin_perturbation * (1.0 + 0.5 * np.random.randn(np_pop, 1))
        
        mutants = p1 + de_term + diversity_term
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```