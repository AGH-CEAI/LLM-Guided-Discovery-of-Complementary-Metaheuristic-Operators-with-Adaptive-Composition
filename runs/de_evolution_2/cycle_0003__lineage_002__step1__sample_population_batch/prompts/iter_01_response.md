**Idea: Cholesky-Based Sampling**
Replace eigendecomposition-based sampling with Cholesky decomposition for O(dim²) sampling instead of O(dim³), avoiding periodic B/D recomputation and providing fundamentally different sampling dynamics.

```python
def _sample_population_batch(self):
        """
        Sample a batch using Cholesky decomposition: x = mean + sigma * L @ z
        where C = L @ L.T. More efficient than eigendecomposition for
        large dimensions and avoids periodic B/D updates.
        """
        z = np.random.randn(self.pop_size, self.dim)
        
        # Ensure Cholesky factor L is valid (C = L @ L.T)
        if not hasattr(self, '_L') or self._L_stale:
            try:
                # Symmetrize C to handle numerical asymmetries
                C_sym = np.triu(self.C) + np.triu(self.C, 1).T
                self._L = np.linalg.cholesky(C_sym)
                self._L_stale = False
            except np.linalg.LinAlgError:
                # C not positive definite - use eigendecomposition fallback
                self._update_eigen_decomposition()
                # Reconstruct from eigendecomposition: C = B @ diag(D²) @ B.T
                self._L = self.B @ np.diag(self.D) @ self.B.T
                self._L_stale = False
        
        # Sample: x = mean + sigma * L @ z
        # (pop_size, dim) = (dim,) + scalar * ((pop_size, dim) @ (dim, dim).T)
        population = self.mean[np.newaxis, :] + self.sigma * (z @ self._L.T)
        
        # Clip to bounds for numerical robustness
        population = np.clip(population, self.lb, self.ub)
        
        return population, z
```