**Idea: Hybrid CMA-ES with Best-Solution Centered Sampling**

When CMA-ES gets stuck in local optima, the population mean drifts away from the global optimum while the best solution remains in the correct basin. This variant samples from a hybrid center blending the current mean with the global best solution, using the learned covariance to generate anisotropic mutations that pull offspring back toward promising regions without losing diversity.

```python
def _sample_population_batch(self):
    """Sample population using hybrid center = blend of mean and best solution."""
    dim = self.dim
    lam = self.pop_size

    # Sample standard normal
    z = np.random.randn(lam, dim)

    # Hybrid center: if best solution exists and has improved, bias toward it
    if self.best_x is not None and np.isfinite(self.best_f):
        # Adaptive weight: more pull toward best when mean has drifted
        diff_norm = np.linalg.norm(self.mean - self.best_x)
        drift_scale = min(diff_norm / (self.sigma + 1e-20), 1.0)
        w_best = 0.3 * drift_scale
        center = (1.0 - w_best) * self.mean + w_best * self.best_x
    else:
        center = self.mean

    # Transform: x = center + sigma * B * D * z
    sqrt_eig = np.sqrt(np.maximum(self.eigenvalues, 1e-20))
    y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
    population = center[None, :] + self.sigma * y.T

    return population, z
```