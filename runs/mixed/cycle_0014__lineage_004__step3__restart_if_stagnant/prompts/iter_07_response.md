**Idea: Diversity-Focused Restart with Opposition-Based and Space-Filling Sampling**

The current restart replaces the worst 1/3 with x_best ± uniform noise, which fails for the hardest tasks (12, 8, 21 with errors ~1000-3000) because it generates candidates too close to the current local optimum. This new strategy uses three fundamentally different sampling methods: (1) Opposition-based learning reflecting x_best around the population centroid, (2) Quasi-random Sobol-like sequences for space-filling coverage, and (3) Cauchy-perturbed candidates for heavy-tailed exploration that can reach distant regions.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
    NP = len(population)
    dim = self.dim
    new_pop = population.copy()
    
    # Compute population statistics for adaptive scaling
    centroid = np.mean(population, axis=0)
    pop_spread = np.std(population, axis=0) + 1e-10
    bound_range = self.upper - self.lower
    
    # Replace worst 50% with diverse strategies
    worst_count = NP // 2
    worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
    
    for i, idx in enumerate(worst_indices):
        strategy = i % 3
        if strategy == 0:
            # Opposition-based: reflect x_best around centroid
            # This explores the opposite side of the search space
            reflection = 2.0 * centroid - x_best
            new_pop[idx] = reflection + np.random.uniform(-0.2 * bound_range, 0.2 * bound_range, dim)
        elif strategy == 1:
            # Space-filling: Halton quasi-random sequence (base 2)
            # Provides better dispersion than uniform random
            halton = np.array([(i + 1) / (dim ** k) % 1.0 for k in range(1, dim + 1)])
            new_pop[idx] = self.lower + halton * bound_range
        else:
            # Heavy-tailed: Cauchy perturbation for long jumps
            # Escapes local optima via occasional very large steps
            cauchy_scale = pop_spread * 2.0
            new_pop[idx] = x_best + np.random.standard_cauchy(dim) * cauchy_scale
    
    new_pop = self._clip_to_bounds_batch(new_pop)
    return new_pop
```