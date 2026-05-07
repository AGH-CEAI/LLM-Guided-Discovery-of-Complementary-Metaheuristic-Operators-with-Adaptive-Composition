**Idea: Eigendecomposition Sampling**
Use eigendecomposition instead of Cholesky for sampling, which is more numerically stable and handles ill-conditioned covariance matrices better.

```python
def _sample_trials_batch(self):
    """Sample new trial population using eigendecomposition."""
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-8:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)

    # Eigendecomposition: C = V @ D @ V.T
    eigvals, eigvecs = np.linalg.eigh(self.C)

    # Robustness: ensure positive eigenvalues and bound condition number
    eigvals = np.maximum(eigvals, 1e-10)
    max_eig = np.max(eigvals)
    if max_eig > 1e6 * np.min(eigvals):
        eigvals = np.clip(eigvals, np.min(eigvals), 1e6 * np.min(eigvals))

    # Sample from standard normal and transform: mean + sigma * V @ sqrt(D) @ z
    z = np.random.randn(self.dim, self.NP)
    D_sqrt = np.sqrt(eigvals)
    self.trials = self.mean[:, np.newaxis] + self.sigma * (eigvecs * D_sqrt) @ z
    self.trials = self.trials.T

    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)
```