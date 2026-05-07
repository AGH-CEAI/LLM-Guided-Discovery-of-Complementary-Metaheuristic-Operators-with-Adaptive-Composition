**Idea: Restart-Triggered Hybrid Sampling**

When stagnation is detected, sample 30% of the population from a restart-based distribution centered on a blended random location to escape local optima, while keeping 70% from the current CMA-ES distribution. This directly targets the worst tasks (16, 19, 20, 23) where errors of 10-500 indicate the optimizer is trapped in poor local minima.

```python
def _sample_population_batch(self):
    """
    Sample a batch of candidates with restart-triggered exploration.
    When stagnated, inject restart-based diversity to escape local optima.
    """
    z = np.random.randn(self.pop_size, self.dim)
    
    # Check stagnation to trigger exploration mode
    stagnated = self.stagnation_counter > self.dim
    
    if stagnated:
        # Restart-based sampling: inject fresh diversity to escape local optima
        n_restart = max(1, int(0.3 * self.pop_size))
        
        # Initialize restart mean within bounds, blended with current mean
        restart_mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=self.dim)
        blend = np.random.uniform(0.3, 0.7)
        self.mean = blend * restart_mean + (1.0 - blend) * self.mean
        self.mean = np.clip(self.mean, self.lb, self.ub)
        
        # Sample restart portion from new mean
        restart_pop = self.mean[np.newaxis, :] + self.sigma * np.random.randn(n_restart, self.dim)
        restart_z = (restart_pop - self.mean[np.newaxis, :]) / (self.sigma + 1e-20)
        
        # Remaining from standard CMA-ES
        z_rest = np.random.randn(self.pop_size - n_restart, self.dim)
        scaled_rest = z_rest * self.D[np.newaxis, :]
        rotated_rest = scaled_rest @ self.B.T
        normal_pop = self.mean[np.newaxis, :] + self.sigma * rotated_rest
        
        # Combine and concatenate z vectors
        population = np.vstack([restart_pop, normal_pop])
        z = np.vstack([restart_z, z_rest])
    else:
        # Standard CMA-ES sampling
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
    
    population = self._clip_to_bounds_batch(population)
    return population, z
```