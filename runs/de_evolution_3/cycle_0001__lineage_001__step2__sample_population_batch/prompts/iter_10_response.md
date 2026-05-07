Looking at the error pattern, the worst tasks (16, 19, 23, 21, 17, 18, 22, 20) are all stuck at O(1) to O(30) error, which strongly suggests the algorithm is converging to **local optima** and lacks the exploration power to escape. All 9 previous variants use standard Gaussian sampling (possibly with parameter tweaks), which has light tails and rarely generates the extreme mutations needed to jump between basins.

**Idea: Heavy-tailed t-Distribution Sampling**
Replace Gaussian sampling with a multivariate t-distribution (df=4), which has heavier tails and will occasionally generate extreme offspring that can escape local basins and discover global optima.

```python
def _sample_population_batch(self):
    """Sample a new population from a heavy-tailed t-distribution for escape from local optima."""
    dim = self.dim
    lam = self.pop_size

    # Sample standard normal base
    z = np.random.randn(lam, dim)

    # Sample chi-squared for t-distribution degrees of freedom
    # df=4 gives heavy tails (kurtosis=6) without extreme instability
    df = 4.0
    chi2_samples = np.random.gamma(df / 2.0, 2.0 / df, size=lam)

    # t-scale factor: z_t = z * sqrt(df / chi2)
    t_scale = np.sqrt(df / np.maximum(chi2_samples, 1e-10))[:, None]
    z_t = z * t_scale  # shape (lam, dim)

    # Transform: x = mean + sigma * B * D * z_t
    sqrt_eig = np.sqrt(np.maximum(self.eigenvalues, 1e-20))
    y = self.eigenvectors @ (sqrt_eig[:, None] * z_t.T)
    population = self.mean[None, :] + self.sigma * y.T

    return population, z_t
```