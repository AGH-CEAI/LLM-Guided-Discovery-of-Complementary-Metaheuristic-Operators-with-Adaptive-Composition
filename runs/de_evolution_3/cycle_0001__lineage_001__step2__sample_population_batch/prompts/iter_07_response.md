**Idea: Heavy-Tailed t-Distribution Sampling**

All 6 prior variants use Gaussian sampling, which has light tails and cannot generate the large jumps needed to escape the deep local optima trapping tasks 16-23 (errors 1-30). Task 17 is particularly telling: ALL variants converge to the exact same error 3.1201e+00, proving Gaussian sampling has collapsed into a basin. This variant replaces the normal distribution with a Student's t-distribution (df=3), which has heavy tails and can occasionally produce mutations 10-100× larger than σ, breaking through to unexplored regions.

```python
def _sample_population_batch(self):
    """Sample from heavy-tailed t-distribution for escape from local optima."""
    dim = self.dim
    lam = self.pop_size

    # Sample standard normal vectors
    z = np.random.randn(lam, dim)
    
    # Construct multivariate t-distribution via normal/chi-square ratio
    df = 3.0  # 3 degrees of freedom gives heavy tails (E[t^4] = infinite)
    chi2 = np.random.chisquare(df, size=lam)
    chi2 = np.maximum(chi2, 1e-10)  # safeguard against division by zero
    
    # t_z = z * sqrt(df / chi2)  -- each row scaled independently
    t_z = z * np.sqrt(df / chi2)[:, None]
    
    # Transform through eigendecomposition: x = mean + sigma * B * D * t_z
    sqrt_eig = np.sqrt(self.eigenvalues)
    y = self.eigenvectors @ (sqrt_eig[:, None] * t_z.T)
    population = self.mean[None, :] + self.sigma * y.T

    # Numerical safety: replace any NaN/Inf with uniform boundary samples
    valid = np.isfinite(population)
    if not np.all(valid):
        n_bad = lam - np.sum(valid.any(axis=1))
        if n_bad > 0:
            bad_rows = np.where(~valid.any(axis=1))[0]
            population[bad_rows] = np.random.uniform(
                self.lb, self.ub, size=(n_bad, dim)
            )

    return population, t_z
```