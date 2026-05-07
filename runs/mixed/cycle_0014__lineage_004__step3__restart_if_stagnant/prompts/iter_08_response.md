**Idea: Adaptive Error-Scaled Restart with Diversity Injection**

This approach scales the restart perturbation range based on the current best error magnitude, using logarithmic scaling to handle the massive errors (1e+03) on Tasks 12/8. It also injects a portion of completely random solutions and uses adaptive dimension-specific perturbations to escape local optima traps.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
    NP = len(population)
    new_pop = population.copy()
    
    # Adaptive perturbation scale based on error magnitude
    # For huge errors (~1e+03), use large perturbations; for smaller errors (~1e-01), use finer ones
    f_best = np.min(fitness)
    error_scale = max(1.0, np.log10(max(f_best, 1e-10) + 1e-10))
    perturbation_range = np.clip(50.0 * error_scale, 10.0, 500.0)
    
    # Replace worst third with error-scaled perturbations around best
    worst_count = NP // 3
    worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
    
    for idx in worst_indices:
        # Adaptive per-dimension scaling: larger for high-error tasks, smaller for near-solved
        scale_factor = perturbation_range * np.random.uniform(0.5, 1.5, self.dim)
        new_pop[idx] = x_best + np.random.uniform(-scale_factor, scale_factor, self.dim)
    
    # Inject 20% completely random solutions for diversity (critical for escaping traps)
    random_count = max(1, NP // 5)
    random_indices = np.random.choice(NP, random_count, replace=False)
    for idx in random_indices:
        new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
    
    new_pop = self._clip_to_bounds_batch(new_pop)
    return new_pop
```