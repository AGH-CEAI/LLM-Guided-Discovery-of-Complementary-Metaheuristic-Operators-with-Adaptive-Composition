**Idea: Heavy-Tailed Exploration Sampling**

The worst unsolved tasks (16: 5e+02, 19: 2.9e+01, 20: 2.2e+01, 23: 1.9e+01, 18: 5.4e+00) are catastrophically stuck at error values 8-10 decades above target. This pattern indicates **premature convergence / entrapment in local optima** — the standard multivariate normal sampling (which all 8 prior variants use) is too focused/exploitative and fails to escape deceptive basins. The algorithm needs **forced diversity injection** to break through.

My mechanism: **Hybrid sampling** where a fraction of the population is randomly re-drawn from the uniform distribution over bounds. This is fundamentally different from all prior variants (which only modified the multivariate normal parameters) because it injects *categorically different* samples from outside the learned covariance ellipsoid. This directly targets the trap-escape problem on the worst tasks.

```python
def _sample_population_batch(self):
    """
    Hybrid sampling: mix CMA-ES multivariate normal samples with
    uniformly distributed random samples for forced diversity.
    """
    pop_size = self.pop_size
    dim = self.dim
    
    # Base CMA-ES sampling
    z = np.random.randn(pop_size, dim)
    scaled = z * self.D[np.newaxis, :]
    rotated = scaled @ self.B.T
    cma_pop = self.mean[np.newaxis, :] + self.sigma * rotated
    
    # Determine mix ratio: use more random samples when stagnating
    if self.stagnation_counter > 10:
        random_fraction = 0.40  # 40% random when stuck
    elif self.stagnation_counter > 5:
        random_fraction = 0.25  # 25% random when slowing
    else:
        random_fraction = 0.15  # 15% random baseline
    
    n_random = int(np.round(pop_size * random_fraction))
    n_random = max(1, min(n_random, pop_size - 1))  # Keep at least 1 CMA sample
    
    # Generate random samples uniform over bounds
    random_pop = np.random.uniform(self.lb, self.ub, size=(n_random, dim))
    
    # Combine: CMA-ES for first (pop_size - n_random), random for last n_random
    combined_pop = np.empty((pop_size, dim), dtype=np.float64)
    combined_pop[:-n_random] = cma_pop[:-n_random] if pop_size > n_random else cma_pop
    combined_pop[-n_random:] = random_pop
    
    # Shuffle to mix CMA and random samples throughout population
    shuffle_idx = np.random.permutation(pop_size)
    population = combined_pop[shuffle_idx]
    
    # For z_vectors: approximate for random samples
    z_combined = np.empty((pop_size, dim), dtype=np.float64)
    z_combined[:-n_random] = z[:-n_random] if pop_size > n_random else z
    z_combined[-n_random:] = np.random.randn(n_random, dim)  # fresh z for random samples
    z_vectors = z_combined[shuffle_idx]
    
    return population, z_vectors
```