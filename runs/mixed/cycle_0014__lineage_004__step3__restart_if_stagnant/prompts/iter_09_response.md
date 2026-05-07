**Idea: Opposition-Based Diversity Restart with Adaptive Scaling**

A fundamental redesign of the restart mechanism that escapes local optima basins using opposition-based learning — generating "mirror" solutions across the search space center. The current approach only perturbs around x_best (±10 range), which fails catastrophically when x_best itself is in a wrong basin (Tasks 12 & 8 with errors ~3000). This new strategy creates entirely new candidate regions by mirroring solutions and using adaptive exploration radii based on problem scale.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
    NP, dim = population.shape
    new_pop = population.copy()
    
    # Determine how many worst individuals to replace
    worst_count = max(1, NP // 3)
    worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
    
    # Compute search space center and range
    center = (self.lower + self.upper) / 2.0
    range_val = self.upper - self.lower
    
    # Compute current diversity to adaptively scale exploration
    current_diversity = self._compute_diversity(population)
    diversity_ratio = np.clip(current_diversity / (range_val * 0.1), 0.1, 2.0)
    
    for idx in worst_indices:
        # Strategy selection: 50% opposition-based, 30% adaptive scaled perturbation, 20% Latin hypercube
        strategy = np.random.random()
        
        if strategy < 0.5:
            # Opposition-based: mirror the individual across search space center
            # This explores the "opposite side" of the space, escaping local basins
            opp_point = self.lower + self.upper - population[idx]
            # Blend with random exploration (30% random influence)
            new_pop[idx] = 0.7 * opp_point + 0.3 * np.random.uniform(self.lower, self.upper, dim)
        
        elif strategy < 0.8:
            # Adaptive scaled perturbation around best (larger scale for hard problems)
            # Scale exploration radius based on current best error magnitude
            scale = max(1.0, min(range_val * 0.5, 50.0 * diversity_ratio))
            new_pop[idx] = x_best + np.random.uniform(-scale, scale, dim)
        
        else:
            # Latin hypercube-inspired: stratified sampling in reduced subspace
            # Focus exploration on promising hyper-rectangles
            subspace_lower = np.minimum(population[idx], x_best) - range_val * 0.1
            subspace_upper = np.maximum(population[idx], x_best) + range_val * 0.1
            subspace_lower = np.clip(subspace_lower, self.lower, self.upper)
            subspace_upper = np.clip(subspace_upper, self.lower, self.upper)
            
            # Generate sample in reduced subspace
            u = np.random.random(dim)
            new_pop[idx] = subspace_lower + u * (subspace_upper - subspace_lower)
    
    new_pop = self._clip_to_bounds_batch(new_pop)
    return new_pop
```