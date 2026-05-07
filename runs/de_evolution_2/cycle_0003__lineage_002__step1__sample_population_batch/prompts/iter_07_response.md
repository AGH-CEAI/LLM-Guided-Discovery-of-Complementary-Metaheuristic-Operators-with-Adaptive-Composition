**Idea: Adaptive Hybrid Sampling with Exploration-Exploitation Scheduling**

A fundamentally different strategy that combines multiple sampling modes with a generation-dependent schedule: (1) pure random search for broad exploration, (2) mean-only perturbation with adaptive step scaling, and (3) standard covariance sampling. The exploration weight decreases linearly from 0.5 to 0.1 over iterations, shifting from global search to local refinement. This targets the worst tasks (16, 19, 20, 23) which appear stuck in poor local optima — the random component provides diversity injection that the current mean-centered sampling lacks, while the adaptive schedule ensures eventual convergence precision.

```python
def _sample_population_batch(self):
    pop_size = self.pop_size
    dim = self.dim
    max_gens = 1000
    exploration_weight = max(0.05, 0.5 * (1.0 - self.generation / max_gens))
    n_random = max(1, int(pop_size * exploration_weight))
    n_mean_pert = max(1, int(pop_size * (0.4 - 0.2 * self.generation / max_gens)))
    n_cov = pop_size - n_random - n_mean_pert
    z = np.random.randn(self.pop_size, self.dim)
    scaled = z * self.D[np.newaxis, :]
    rotated = scaled @ self.B.T
    cov_samples = self.mean[np.newaxis, :] + self.sigma * rotated
    mean_pert_scale = self.sigma * max(0.5, 3.0 * self.sigma / (1.0 + self.generation))
    mean_pert_z = np.random.randn(n_mean_pert, dim)
    mean_pert_samples = self.mean[np.newaxis, :] + mean_pert_scale * mean_pert_z
    random_scale = max(abs(self.ub - self.lb), 50.0)
    random_samples = np.random.uniform(
        self.mean - random_scale, self.mean + random_scale, size=(n_random, dim)
    )
    population = np.vstack([cov_samples[:n_cov], mean_pert_samples, random_samples])
    if population.shape[0] > pop_size:
        population = population[:pop_size]
    elif population.shape[0] < pop_size:
        extra = np.random.randn(pop_size - population.shape[0], dim)
        extra = self.mean[np.newaxis, :] + self.sigma * (extra * self.D[np.newaxis, :]) @ self.B.T
        population = np.vstack([population, extra])
    return population, z
```