**Idea: Eigenspace-Adaptive Covariance with Explosive Exploration**

A fundamentally different covariance adaptation that performs eigendecomposition-based updates with aggressive eigenvalue shaping. By projecting all updates into eigenspace, it can (1) apply much larger learning rates without losing positive-definiteness, (2) explicitly elongate axes for exploration on ill-conditioned tasks, and (3) use eigenvalue-based condition number control to prevent premature convergence. This contrasts with the current Cholesky approach which is numerically stable but limited to incremental updates.

```python
def _adapt_covariance_variant_01(self):
    """Eigenspace-adaptive covariance with explosive exploration."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Eigendecomposition of current covariance
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)
    except np.linalg.LinAlgError:
        self.C = 1e-6 * np.eye(self.dim)
        return
    
    # Project mean change into eigenspace
    z_1 = eigvecs.T @ y_mean
    z_1_norm = np.linalg.norm(z_1)
    if z_1_norm > 1e-10:
        z_1 = z_1 / z_1_norm
    
    # Adaptive learning rates based on eigenvalue spread
    cond_C = np.max(eigvals) / np.min(eigvals)
    cond_log = np.log1p(cond_C)
    
    # More aggressive rates for ill-conditioned problems (priority tasks)
    ccov_1 = min(2.0 / (self.dim + 2.0), 0.3)
    ccov_mu = min(self.mueff / (self.dim + 2.0) / (self.dim + 4.0), 0.2)
    
    # Rank-one update in eigenspace
    rank_one = z_1 ** 2
    
    # Rank-mu update in eigenspace
    rank_mu = np.zeros(self.dim)
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        z_i = eigvecs.T @ diff
        rank_mu += self.weights[i] * (z_i ** 2)
    
    # Update eigenvalues directly
    eigvals_new = (1.0 - ccov_1 - ccov_mu) * eigvals
    eigvals_new += ccov_1 * z_1_norm * z_1_norm * eigvals * rank_one
    eigvals_new += ccov_mu * eigvals * rank_mu
    
    # Explosive exploration: elongate axes for ill-conditioned tasks
    if cond_log > 2.0:
        elongation = min(cond_log * 0.3, 2.0)
        max_idx = np.argmax(eigvals)
        eigvals_new[max_idx] *= (1.0 + elongation)
    
    # Clamp eigenvalues for numerical stability
    eigvals_new = np.clip(eigvals_new, 1e-14, 1e+10)
    
    # Reconstruct covariance matrix
    self.C = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
    self.C = 0.5 * (self.C + self.C.T)
    
    # Condition number control
    new_cond = np.max(eigvals_new) / np.min(eigvals_new)
    if new_cond > 1e+8:
        min_eig = np.min(eigvals_new)
        max_eig = min_eig * 1e+8
        eigvals_new = np.clip(eigvals_new, None, max_eig)
        self.C = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
```