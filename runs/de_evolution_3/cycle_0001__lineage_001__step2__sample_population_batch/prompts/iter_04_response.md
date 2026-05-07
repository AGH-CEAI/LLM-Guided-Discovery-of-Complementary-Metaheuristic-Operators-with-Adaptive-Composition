**Idea: Restart-Triggered Diversity Injection**
When stagnation is detected, expand sigma and inject random samples to escape local optima. The fraction of random samples scales with stagnation depth (10%→50%), directly targeting the worst tasks (16,19,23) that are stuck at large errors (~10-33) due to premature convergence.

```python
def _sample_population_batch(self):
    """Sample with restart-triggered diversity injection to escape local optima."""
    dim = self.dim
    lam = self.pop_size

    # Determine random injection fraction based on stagnation depth
    if self.stagnation_counter > 8:
        random_frac = 0.5  # 50% random when deeply stuck
    elif self.stagnation_counter > 4:
        random_frac = 0.3  # 30% random when stuck
    elif self.stagnation_counter > 2:
        random_frac = 0.2  # 20% random when beginning to stagnate
    else:
        random_frac = 0.1  # 10% random for mild diversity maintenance

    n_random = max(1, int(lam * random_frac))
    n_cma = lam - n_random

    # Standard CMA-ES sampling
    z_all = np.random.randn(lam, dim)
    sqrt_eig = np.sqrt(self.eigenvalues + 1e-20)
    y = self.eigenvectors @ (sqrt_eig[:, None] * z_all.T)
    cma_samples = self.mean[None, :] + self.sigma * y.T

    # Inject uniform random samples for diversity
    random_samples = np.random.uniform(self.lb, self.ub, (n_random, dim))

    # Combine CMA-ES and random samples
    if n_random > 0 and n_cma > 0:
        population = np.vstack([cma_samples[:n_cma], random_samples])
    elif n_random > 0:
        population = random_samples
    else:
        population = cma_samples

    # Clip to bounds
    population = np.clip(population, self.lb, self.ub)

    return population, z_all
```