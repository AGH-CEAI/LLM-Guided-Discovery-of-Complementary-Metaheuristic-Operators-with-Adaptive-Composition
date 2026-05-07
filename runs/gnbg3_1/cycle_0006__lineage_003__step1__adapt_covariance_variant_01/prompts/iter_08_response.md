**Idea: Eigendecomposition-Directed Covariance with Adaptive Eigenvalue Control**

This variant directly modifies covariance eigenvalues using eigendecomposition, boosting small eigenvalues to escape local optima and shrinking large ones for fine-grained convergence. It also introduces restart-triggered diversity pulses and orthogonal random direction injection when stagnation is detected.

```python
def _adapt_covariance_variant_01(self):
    """Eigendecomposition-directed covariance with adaptive eigenvalue control."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute population diversity to detect premature convergence
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Compute rank-mu update matrix
    rank_mu = np.zeros((self.dim, self.dim))
    total_weight = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_weight += abs(w)
    
    if total_weight > 1e-30:
        rank_mu /= total_weight
    
    # Build target covariance as weighted combination
    ccov_1 = 1.0 / (self.dim + 2.0)
    ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    
    rank_one = np.outer(y_mean, y_mean) / max(np.dot(y_mean, y_mean), 1e-30)
    
    C_target = (1.0 - ccov_1 - ccov_mu) * self.C
    C_target += ccov_1 * rank_one
    C_target += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])
    
    # Ensure symmetry and positive definiteness
    C_target = 0.5 * (C_target + C_target.T)
    min_eig = np.min(np.linalg.eigvalsh(C_target))
    if min_eig < 1e-12:
        C_target += (1e-11 - min_eig) * np.eye(self.dim)
    
    # Perform eigendecomposition for direct eigenvalue control
    try:
        eigvals, eigvecs = np.linalg.eigh(C_target)
    except np.linalg.LinAlgError:
        C_target = np.eye(self.dim) * np.mean(np.diag(self.C))
        eigvals, eigvecs = np.linalg.eigh(C_target)
    
    # Stability check
    eigvals = np.maximum(eigvals, 1e-15)
    
    # Adaptive eigenvalue modification based on convergence state
    cond = np.max(eigvals) / (np.min(eigvals) + 1e-15)
    
    if cond > 1e6 or diversity < 0.05:
        # Escape mode: boost smallest eigenvalues significantly
        eigvals = eigvals ** 0.5
        min_eig_val = np.min(eigvals)
        boost_factor = 2.0 + 3.0 * (1.0 - diversity)
        eigvals = eigvals + boost_factor * min_eig_val
        eigvals = np.clip(eigvals, 1e-14, None)
    elif cond < 1e2 and diversity > 0.2:
        # Refinement mode: shrink largest eigenvalues for precision
        max_eig_val = np.max(eigvals)
        shrink_factor = 0.7
        eigvals = np.where(eigvals > 0.5 * max_eig_val, eigvals * shrink_factor, eigvals)
        eigvals = np.maximum(eigvals, 1e-15)
    else:
        # Balanced mode: moderate regularization
        eigvals = eigvals * 0.95 + 0.05 * np.mean(eigvals)
    
    # Reconstruct covariance from modified eigenvalues
    self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    self.C = 0.5 * (self.C + self.C.T)
    
    # Inject orthogonal random exploration when diversity is critically low
    if diversity < 0.02:
        for _ in range(3):
            z = np.random.randn(self.dim)
            z = z / max(np.linalg.norm(z), 1e-15)
            random_dir = np.outer(z, z)
            self.C += 0.1 * np.mean(eigvals) * random_dir
    
    # Final safety check
    self.C = self._ensure_positive_definite(self.C)
    
    # Update Cholesky factor for sampling
    try:
        self.L = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.dim))
    except np.linalg.LinAlgError:
        diag_C = np.diag(self.C)
        diag_C = np.maximum(diag_C, 1e-10)
        self.L = np.diag(np.sqrt(diag_C))
```