**Idea: Population-Center Guided Multi-Direction Jump**

This strategy replaces p_best-centric mutation with population-statistics-based guidance, addressing the catastrophic failures on tasks 12/16/8/21 where p_best gets trapped in poor regions. The population mean acts as a global attractor toward the search basin, while multiple random difference vectors and adaptive scaling provide escape from local optima.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Population-center guided multi-direction jump with adaptive scaling"""
    # Compute population statistics (global guidance instead of p_best)
    pop_mean = np.mean(population, axis=0)
    pop_std = np.std(population, axis=0) + 1e-10
    
    # Direction toward population center
    center_direction = pop_mean - population
    
    # Generate three random index sets for multi-direction exploration
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]
    
    # Compute random difference vectors
    diff_random = population[r1] - population[r2]
    
    # Adaptive F based on stagnation (larger jumps when stuck)
    if self.stagnation_count > 10:
        F_base = min(1.5 + 0.1 * self.stagnation_count, 3.0)
    elif self.stagnation_count > 5:
        F_base = min(1.0 + 0.05 * self.stagnation_count, 2.0)
    else:
        F_base = self.F
    
    F_adapted = np.clip(F_base + np.random.randn(self.NP) * 0.2, 0.3, 3.0)
    
    # Compute fitness-based weight: favor individuals far from mean (exploration)
    dist_to_mean = np.linalg.norm(population - pop_mean, axis=1)
    dist_std = np.std(dist_to_mean) + 1e-10
    exploration_weight = np.exp(-dist_to_mean / dist_std) * 0.3
    exploitation_weight = 1.0 - exploration_weight
    
    # Combine: exploitation toward center + exploration via random differences
    # Exploitation term: move toward population mean
    exploit_term = F_adapted[:, np.newaxis] * exploitation_weight[:, np.newaxis] * center_direction
    
    # Exploration term: random multi-direction perturbation
    explore_term = F_adapted[:, np.newaxis] * (0.5 * diff_random + 0.5 * (population[r2] - pop_mean))
    
    # Small perturbation toward global best if available
    if self.best_solution is not None and np.isfinite(self.best_fitness):
        best_term = 0.1 * self.F * (self.best_solution - population)
    else:
        best_term = 0.0
    
    donors = population + exploit_term + explore_term + best_term
    
    # Add adaptive jump perturbation when highly stagnant
    if self.stagnation_count > 20:
        jump_scale = min(0.5 * np.log1p(self.stagnation_count - 20), 5.0)
        jump = np.random.randn(self.NP, self.dim) * jump_scale
        donors = donors + jump
    
    return np.clip(donors, self.lower, self.upper)
```