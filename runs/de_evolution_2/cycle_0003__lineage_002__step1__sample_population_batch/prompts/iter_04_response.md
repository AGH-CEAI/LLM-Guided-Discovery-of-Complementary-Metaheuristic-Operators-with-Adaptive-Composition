Looking at the priority targets, the worst unsolved tasks (16, 19, 20, 23) have errors stuck at 10-500, suggesting the optimizer is getting trapped in local optima. The current sampling uses pure CMA-ES without mechanisms to escape stagnation. I need a fundamentally different approach.

**Idea: Stagnation-Triggered Elite Memory Exploration**

The key failure mode for the worst tasks is premature convergence — the covariance matrix collapses around a local optimum and CMA-ES can't escape. This approach maintains an elite memory buffer, detects stagnation using generational tracking, and when stuck, injects geometrically-informed exploration noise scaled by the current step size and elite solution variance. This is fundamentally different from prior variants because it adds a completely new "escape mode" triggered by stagnation, using the covariance diagonal for geometric awareness and elite variance for adaptive noise scaling.

```python
def _sample_population_batch(self):
    z = np.random.randn(self.pop_size, self.dim)
    scaled = z * self.D[np.newaxis, :]
    rotated = scaled @ self.B.T
    population = self.mean[np.newaxis, :] + self.sigma * rotated

    # Track elite memory for exploration
    if not hasattr(self, '_elite_memory'):
        self._elite_memory = []
    if self.best_x is not None:
        self._elite_memory.append(self.best_x.copy())
        max_memory = 5 * self.mu
        if len(self._elite_memory) > max_memory:
            self._elite_memory = self._elite_memory[-max_memory:]

    # Compute stagnation-based exploration factor
    stagnation_threshold = max(3, int(0.1 * self.dim))
    if self.generation > 5 and self.generation - self.eigen_decomp_gen > stagnation_threshold:
        stagnation_gen = self.generation - self.eigen_decomp_gen - stagnation_threshold
        exploration_factor = min(0.5 * (1.0 - np.exp(-stagnation_gen / (0.3 * self.dim))), 0.4)
    else:
        exploration_factor = 0.0

    # Inject exploration noise when stagnant
    if exploration_factor > 0.01 and len(self._elite_memory) > 0:
        elite_arr = np.array(self._elite_memory)
        elite_std = np.std(elite_arr, axis=0)
        noise_scale = self.sigma * exploration_factor * np.maximum(elite_std, 1e-8)
        exploration_noise = np.random.randn(self.pop_size, self.dim) * noise_scale[np.newaxis, :]
        population = population + exploration_noise

    return population, z
```