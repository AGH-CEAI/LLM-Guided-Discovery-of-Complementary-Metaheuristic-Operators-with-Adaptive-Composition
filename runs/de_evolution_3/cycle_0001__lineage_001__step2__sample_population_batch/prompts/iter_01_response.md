**Idea: Antithetic Variate Sampling**

Use antithetic variates to generate negatively correlated sample pairs, reducing sampling variance and improving convergence speed across diverse problem types.

```python
def _sample_population_batch(self):
    """Sample a new population using antithetic variates for variance reduction."""
    dim = self.dim
    lam = self.pop_size

    # Determine number of half-samples (must be even for pairs)
    half_lam = (lam // 2) * 1  # integer division
    if lam % 2 == 0:
        n_pairs = lam // 2
    else:
        n_pairs = (lam - 1) // 2  # one extra sample will be added

    # Sample n_pairs from standard normal
    z_pairs = np.random.randn(n_pairs, dim)

    # Create antithetic pairs: stack original and negated samples
    z = np.vstack([z_pairs, -z_pairs])

    # If odd population size, generate one additional independent sample
    if lam % 2 == 1:
        z_extra = np.random.randn(1, dim)
        z = np.vstack([z, z_extra])

    # Transform: x = mean + sigma * B * D * z
    sqrt_eig = np.sqrt(self.eigenvalues)
    y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
    population = self.mean[None, :] + self.sigma * y.T

    return population, z
```