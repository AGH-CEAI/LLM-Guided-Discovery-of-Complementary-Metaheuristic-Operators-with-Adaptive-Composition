**Idea: Cholesky-Enabled Sampling**
Use Cholesky decomposition instead of eigendecomposition for sampling, which is computationally more efficient (O(n²) vs O(n³)) and handles covariance updates more gracefully.
```python
def _sample_trials_batch(self):
    """Sample new trial population using Cholesky decomposition."""
    # Ensure positive definiteness before Cholesky
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-8:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
    # Compute Cholesky factor L where C = L @ L.T
    # More efficient than eigendecomposition for sampling
    try:
        L = np.linalg.cholesky(self.C)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal approximation
        diag_C = np.diag(self.C)
        diag_C = np.maximum(diag_C, 1e-10)
        L = np.diag(np.sqrt(diag_C))
    
    # Sample from standard normal and transform
    z = np.random.randn(self.NP, self.dim)
    self.trials = self.mean + self.sigma * (z @ L.T)
    
    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)
```