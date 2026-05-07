**Idea: Spectral Condition Number Controlled Eigendecomposition**

Instead of just clipping negative eigenvalues, perform full eigendecomposition with active condition number control via logarithmic eigenvalue rescaling. For the hardest tasks (errors 1e+01 to 7e+01), ill-conditioned covariance matrices likely cause premature convergence and poor sampling directions—the algorithm gets stuck because it can't explore effectively.

```python
def _ensure_positive_definite(self, C):
    """Ensure symmetric positive definite with controlled condition number."""
    C = 0.5 * (C + C.T)
    
    # Full eigendecomposition for direct eigenvalue control
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # Enforce minimum eigenvalue threshold
    eigvals = np.maximum(eigvals, 1e-10)
    
    # Control condition number via logarithmic rescaling
    cond = eigvals[-1] / eigvals[0]
    max_cond = 1e6
    if cond > max_cond:
        log_eigvals = np.log(eigvals)
        log_min = log_eigvals[0]
        log_max = log_eigvals[-1]
        target_range = np.log(max_cond * eigvals[0]) - log_min
        current_range = log_max - log_min
        if current_range > 1e-10:
            scale = target_range / current_range
            log_eigvals[1:] = log_min + scale * (log_eigvals[1:] - log_min)
        eigvals = np.exp(np.clip(log_eigvals, -50, 50))
    
    # Reconstruct with controlled spectrum
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Final safety clip
    C = 0.5 * (C + C.T)
    return C
```