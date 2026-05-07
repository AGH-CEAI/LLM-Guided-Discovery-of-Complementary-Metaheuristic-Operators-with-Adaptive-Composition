**Idea: Multi-Distribution Ensemble Sampling**

Sample from three parallel distributions with different scales, then recombine the best candidates. This breaks the single-mode attraction that traps the optimizer on the worst tasks (errors 10-100× larger than target), while still exploiting good solutions found so far. The ensemble approach is fundamentally different from all prior variants that sample from a single distribution.

```python
def _sample_population_batch(self):
    """
    Sample from a multi-distribution ensemble: (1) tight around best-so-far,
    (2) standard CMA-ES around current mean, (3) wide exploration.
    Recombine top candidates to escape local optima on the worst tasks.
    """
    pop_size = self.pop_size
    dim = self.dim
    
    # Distribution 1: 30% of budget — tight around global best (exploitation)
    n_elite = max(1, int(0.3 * pop_size))
    elite_pop = np.zeros((n_elite, dim))
    if self.best_x is not None:
        for i in range(n_elite):
            scale = 0.01 * self.sigma * (0.5 ** i)  # shrinking scale for diversity
            elite_pop[i] = self.best_x + scale * np.random.randn(dim)
    else:
        elite_pop = self.mean + 0.01 * self.sigma * np.random.randn(n_elite, dim)
    elite_pop = self._clip_to_bounds_batch(elite_pop)
    
    # Distribution 2: 40% of budget — standard CMA-ES around current mean
    n_standard = max(1, int(0.4 * pop_size))
    z_std = np.random.randn(n_standard, dim)
    scaled_std = z_std * self.D[np.newaxis, :]
    rotated_std = scaled_std @ self.B.T
    standard_pop = self.mean[np.newaxis, :] + self.sigma * rotated_std
    standard_pop = self._clip_to_bounds_batch(standard_pop)
    
    # Distribution 3: 30% of budget — wide uniform exploration (diversity)
    n_wide = pop_size - n_elite - n_standard
    wide_pop = np.random.uniform(self.lb, self.ub, size=(n_wide, dim))
    
    # Combine all candidates
    population = np.vstack([elite_pop, standard_pop, wide_pop])
    all_z = np.vstack([
        np.zeros((n_elite, dim)),  # approximate z for elite
        z_std,
        np.zeros((n_wide, dim))   # approximate z for uniform
    ])
    
    return population, all_z
```