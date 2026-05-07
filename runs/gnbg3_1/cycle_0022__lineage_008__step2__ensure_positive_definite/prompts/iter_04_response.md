**Idea: Adaptive Eigendecomposition Flooring**

Instead of adding a fixed constant times identity, perform full eigendecomposition and clip eigenvalues to a dimension-adaptive minimum threshold. This handles ill-conditioned matrices more gracefully and prevents the large, potentially destabilizing corrections that occur when min_eig is very negative (as likely happens on the worst tasks with errors >10).
```python
def _ensure_positive_definite(self, C):
    """Ensure matrix is symmetric and positive definite via adaptive eigenvalue flooring."""
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)
    if max_eig == 0:
        max_eig = 1.0
    # Adaptive threshold based on max eigenvalue and dimension
    eig_thresh = max(1e-10, 1e-6 * max_eig / self.dim)
    if min_eig < eig_thresh:
        eigvals = np.maximum(eigvals, eig_thresh)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return C
```