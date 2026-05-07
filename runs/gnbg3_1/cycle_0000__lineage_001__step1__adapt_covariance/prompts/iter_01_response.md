**Idea: Cholesky-Based Covariance Adaptation**
Directly adapt the Cholesky factor L of the covariance matrix, ensuring positive definiteness naturally and improving numerical stability over evolution-path updates.

```python
def _adapt_covariance(self):
    """
    Adapt covariance via Cholesky factor update.
    Fundamentally different from evolution-path approach - maintains C = LL^T
    structure directly without eigenvalue decomposition.
    """
    # Initialize L if needed
    if not hasattr(self, 'L') or self.L is None:
        self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
    
    # Compute weighted mean shift in z-space (normalized coordinates)
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Rank-1 update: use the normalized mean shift directly
    z_1 = y_mean / np.maximum(np.linalg.norm(y_mean), 1e-10)
    
    # Rank-mu update: weighted sum of squared normalized vectors
    rank_mu = np.zeros((self.dim, self.dim))
    total_weight = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_weight += w
    
    # Normalize rank_mu weights
    if total_weight > 0:
        rank_mu /= total_weight
    
    # Effective learning rate combining rank-1 and rank-mu
    ccov_1 = 1.0 / (self.dim + 2.0)  # Rank-1 rate
    ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)  # Rank-mu rate
    
    # Update covariance directly then refactorize
    C_new = (1.0 - ccov_1 - ccov_mu) * self.C
    C_new += ccov_1 * np.outer(z_1, z_1)
    C_new += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])
    
    # Ensure symmetry and positive definiteness
    C_new = 0.5 * (C_new + C_new.T)
    
    # Add minimal regularization for numerical stability
    min_eig = np.min(np.linalg.eigvalsh(C_new))
    if min_eig < 1e-10:
        C_new += (1e-8 - min_eig) * np.eye(self.dim)
    
    self.C = C_new
    
    # Update Cholesky factor using rank-1 update formula
    try:
        # Use O(d^2) rank-1 Cholesky update
        v = np.linalg.solve(self.L.T, z_1)
        alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
        self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
    except np.linalg.LinAlgError:
        # Fallback: recompute Cholesky from scratch
        self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
```