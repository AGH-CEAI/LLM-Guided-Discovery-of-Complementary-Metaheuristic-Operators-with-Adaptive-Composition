**Idea: Spectral Clipping with Random Perturbation**

Instead of only adding a constant when eigenvalues are negative, this approach performs full eigendecomposition and clips all eigenvalues to a minimum threshold while adding adaptive noise to the smallest eigenvalues. This fundamentally differs from the current reactive approach by proactively managing the eigenvalue spectrum to maintain exploration ability, which is critical for the worst-performing tasks that appear stuck in local optima.

```python
def _ensure_positive_definite(self, C):
    """Ensure matrix is symmetric and positive definite via eigendecomposition with adaptive spectral management."""
    # Step 1: Symmetrize
    C = 0.5 * (C + C.T)
    
    # Step 2: Eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        # Fallback for non-convergent cases
        return 0.5 * (C + np.eye(self.dim) * 1e-6)
    
    # Step 3: Adaptive minimum eigenvalue based on matrix size and condition
    dim = eigvals.shape[0]
    base_min = 1e-10
    # Scale minimum eigenvalue based on the largest eigenvalue to maintain relative conditioning
    max_eig = np.max(eigvals)
    adaptive_min = base_min * max(1.0, max_eig)
    adaptive_min = np.clip(adaptive_min, 1e-12, 1e-4)
    
    # Step 4: Clip eigenvalues to minimum threshold
    eigvals_clipped = np.maximum(eigvals, adaptive_min)
    
    # Step 5: Add adaptive noise to smallest eigenvalues to maintain exploration
    # The noise scales with the variance of the top eigenvalues
    eigvals_sorted = np.sort(eigvals_clipped)
    if len(eigvals_sorted) > 1:
        # Variance-based noise for diversity
        noise_scale = np.var(eigvals_sorted[-max(1, dim//4):]) * 1e-4
        noise_scale = np.clip(noise_scale, 1e-15, 1e-3)
        # Add noise only to the smallest 20% of eigenvalues
        n_small = max(1, int(0.2 * dim))
        small_eigvals = eigvals_clipped[:n_small]
        noise = np.random.uniform(0, noise_scale, size=n_small)
        eigvals_clipped[:n_small] = small_eigvals + noise
    
    # Step 6: Reconstruct matrix
    C_fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    # Step 7: Final symmetrization and clipping
    C_fixed = 0.5 * (C_fixed + C_fixed.T)
    
    # Ensure diagonal dominance for numerical stability
    diag = np.diag(C_fixed)
    off_diag_max = np.max(np.abs(C_fixed - np.diag(diag)))
    if np.any(diag < off_diag_max):
        diag = np.maximum(diag, off_diag_max * 1.01)
        C_fixed = np.diag(diag - off_diag_max) + C_fixed
    
    return C_fixed
```