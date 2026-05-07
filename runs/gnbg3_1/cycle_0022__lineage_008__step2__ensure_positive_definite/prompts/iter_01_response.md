**Idea: Eigendecomposition-based Eigenvalue Clipping**
Use full eigendecomposition to clip all eigenvalues to a minimum threshold, ensuring positive definiteness more robustly than diagonal perturbation.

```python
def _ensure_positive_definite(self, C):
    """Ensure matrix is symmetric and positive definite via eigenvalue clipping."""
    # Symmetrize the matrix
    C = 0.5 * (C + C.T)
    
    # Perform eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # Clip all eigenvalues to minimum threshold
    eigvals = np.maximum(eigvals, 1e-10)
    
    # Reconstruct the matrix
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return C
```