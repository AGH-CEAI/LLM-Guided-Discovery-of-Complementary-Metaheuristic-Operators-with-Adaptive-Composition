**Idea: Spectral Clipping with Log-Scale Normalization**

Replaces the minimal diagonal shift with full eigendecomposition that aggressively clips eigenvalues to a relative floor (1e-6 of max eigenvalue) and log-scale compresses extreme eigenvalue ratios. This directly targets the ill-conditioned covariance matrices causing the worst tasks (16, 17, 20, 23 with errors ~15-70) to stagnate far from the target.

```python
def _ensure_positive_definite(self, C):
    """Ensure matrix is symmetric and positive definite via eigendecomposition."""
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    eig_max = np.max(eigvals)
    if eig_max <= 0:
        return 1e-6 * np.eye(self.dim)
    eigvals_clipped = np.clip(eigvals, eig_max * 1e-6, None)
    C = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    C = 0.5 * (C + C.T)
    return C
```