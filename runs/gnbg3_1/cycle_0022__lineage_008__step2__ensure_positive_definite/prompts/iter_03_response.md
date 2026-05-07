Looking at the worst unsolved tasks (errors ~1e+1 to ~7e+1), the fundamental issue is that the optimizer is getting stuck in severely suboptimal regions. The `_ensure_positive_definite` function is critical because it controls the sampling covariance - if it fails to maintain a well-conditioned matrix, the search quality collapses.

**Analysis of the problem:**
- Current approach: Only adds a tiny fixed shift (1e-7) to the diagonal when min_eig < 1e-10
- This is insufficient for ill-conditioned matrices that arise during stagnation
- The worst tasks (16, 17, 23, etc.) have errors ~10-70, indicating the covariance adaptation is failing entirely
- Simply shifting the diagonal doesn't fix matrices with severely negative eigenvalues or bad eigenvalue spread

**Key insight:** The current approach is a "band-aid" fix. A fundamentally different strategy is to **perform actual eigendecomposition and directly clip/fix eigenvalues**, then reconstruct the matrix. This is fundamentally different because:
1. It directly addresses the eigenvalue problem rather than patching the diagonal
2. It can handle arbitrarily negative eigenvalues
3. It maintains better spectral properties

**Idea: Eigenvalue Clipping with Spectral Control**

```python
def _ensure_positive_definite(self, C):
    """Ensure matrix is symmetric and positive definite via eigendecomposition."""
    C = 0.5 * (C + C.T)
    
    # Eigendecomposition - directly handle eigenvalue issues
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Clip eigenvalues to ensure positive definiteness (key difference from diagonal shift)
    eig_min = np.min(eigenvalues)
    if eig_min < 1e-12:
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Enforce maximum condition number to prevent ill-conditioning
    eig_max = np.max(eigenvalues)
    cond_threshold = 1e6
    if eig_max / max(eig_min, 1e-12) > cond_threshold:
        eigenvalues = np.clip(eigenvalues, eig_max / cond_threshold, None)
        C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return C
```