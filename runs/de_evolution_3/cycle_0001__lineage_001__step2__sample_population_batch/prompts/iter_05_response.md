**Idea: Restart-Driven Diversity Injection**

The worst unsolved tasks (errors stuck at >1.0) show classic convergence trap symptoms: CMA-ES gets locked into a local basin and the covariance matrix only explores around that basin. The current sampling samples *entirely* from `N(mean, σ²C)`, so if trapped, the algorithm can never escape. This variant injects aggressive random restart samples to break out of deceptive local optima.

```python
def _sample_population_batch(self):
    """Sample a new population with restart-driven diversity injection."""
    dim = self.dim
    lam = self.pop_size

    # Sample standard normal
    z = np.random.randn(lam, dim)

    # Transform: x = mean + sigma * B * D * z
    sqrt_eig = np.sqrt(self.eigenvalues)
    # y = B * D * z^T => shape (dim, lam)
    y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
    # population = mean + sigma * y^T
    population = self.mean[None, :] + self.sigma * y.T

    # Inject restart-based diversity: replace ~50% of population with fresh samples
    num_restart = max(1, int(0.5 * lam))
    restart_indices = np.random.choice(lam, num_restart, replace=False)
    
    for idx in restart_indices:
        # Sample uniformly from the full search space (escape local basin)
        restart_point = np.random.uniform(self.lb, self.ub, dim)
        population[idx] = restart_point
        # Approximate z for covariance update (map restart point back to z-space)
        diff = restart_point - self.mean
        z[idx] = (self.eigenvectors.T @ diff) / (self.sigma * sqrt_eig + 1e-20)

    return population, z
```