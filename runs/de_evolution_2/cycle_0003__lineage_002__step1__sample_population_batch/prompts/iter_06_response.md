**Idea: Heavy-Tailed t-Distribution Sampling**

Replace Gaussian sampling with Student's t-distribution sampling to enable heavy-tailed exploration. Tasks 16, 19, 20, 23 have errors of 10–500 (stuck ~10 decades above target), meaning the optimizer is trapped in completely wrong regions. Heavy tails allow occasional extreme jumps that can escape local optima basins and discover new search regions.

```python
def _sample_population_batch(self):
        """
        Sample a batch of candidates from the multivariate Student's t-distribution
        T(mean, sigma^2 * C, df) for heavy-tailed exploration. Returns array of shape
        (pop_size, dim).
        """
        # Degrees of freedom - lower = heavier tails for escaping local optima
        df = max(3.001, self.dim / 5.0)
        
        # Sample standard normals
        z = np.random.randn(self.pop_size, self.dim)
        
        # Sample chi-squared for t-distribution scaling
        chi2_samples = np.random.chisquare(df, self.pop_size)
        
        # Ensure numerical stability (avoid division by zero)
        chi2_samples = np.maximum(chi2_samples, 1e-6)
        
        # Compute scaling factor sqrt(df / chi2) for each sample
        scaling = np.sqrt(df / chi2_samples)
        
        # Apply t-distribution scaling to z-vectors
        t_z = z * scaling[:, np.newaxis]
        
        # Transform: x = mean + sigma * B * D * t_z
        scaled = t_z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        
        return population, t_z
```