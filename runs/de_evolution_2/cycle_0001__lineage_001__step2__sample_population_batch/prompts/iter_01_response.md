**Idea: Cholesky-Enabled Sampling**

Replace eigendecomposition-based sampling with Cholesky decomposition, which provides a fundamentally different transformation pathway for generating the multivariate normal samples. This approach is computationally distinct (triangular system solves vs. orthogonal rotation) and often exhibits different convergence characteristics.

```python
def _sample_population_batch(self):
        """
        Sample a batch of candidates using Cholesky decomposition of the
        covariance matrix: x = mean + sigma * L @ z where C ≈ L @ L.T.
        Returns array of shape (pop_size, dim).
        """
        # Sample standard normals
        z = np.random.randn(self.pop_size, self.dim)
        
        # Compute Cholesky factor with small regularization for numerical stability
        # This handles near-singular covariance matrices gracefully
        regularized_C = self.C + 1e-12 * np.eye(self.dim)
        try:
            L = np.linalg.cholesky(regularized_C)
        except np.linalg.LinAlgError:
            # Fallback: add more regularization and try again
            L = np.linalg.cholesky(regularized_C + 1e-8 * np.eye(self.dim))
        
        # Transform: x = mean + sigma * L @ z
        # L @ z computed efficiently using matrix multiplication
        noise = z @ L.T  # (pop_size, dim)
        population = self.mean[np.newaxis, :] + self.sigma * noise
        
        return population, z
```