**Idea: Eigenvalue Clipping with Condition Number Regularization**

This approach performs full eigendecomposition to clip eigenvalues below a dimension-adaptive threshold and regularize the condition number, ensuring well-conditioned sampling for the worst tasks (16, 17, 20, 23) that suffer from ill-conditioned covariance matrices leading to poor exploration.

```python
def _ensure_positive_definite(self, C):
    """Ensure matrix is symmetric and positive definite via eigenvalue clipping and conditioning."""
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)
    cond = max_eig / max(min_eig, 1e-15)
    dim_scale = max(1.0, self.dim / 10.0)
    eig_thresh = 1e-10 * dim_scale
    max_cond = 1e6 * dim_scale
    if cond > max_cond:
        target_min = max_eig / max_cond
        eigvals = np.clip(eigvals, target_min, None)
    elif min_eig < eig_thresh:
        eigvals = np.clip(eigvals, eig_thresh, None)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    C = 0.5 * (C + C.T)
    return C
```