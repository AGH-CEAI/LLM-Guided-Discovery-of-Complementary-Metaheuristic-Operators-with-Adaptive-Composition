Looking at the benchmark results, the worst unsolved tasks (16, 17, 23, 20, 11, 18, 19, 12, 10) have errors in the 10-80 range, indicating the optimizer is collapsing into deceptive local optima and failing to escape. The existing variants use archive guidance, diversity scaling, active CMA, or temperature-based approaches, but none force systematic exploration of unexplored covariance directions.

**Idea: Eigenvalue Floor with Orthogonal Direction Perturbation**
This approach decomposes the covariance matrix, identifies the most underexplored directions (smallest eigenvalues), and injects structured perturbations along orthogonal axes. It differs fundamentally from prior variants by explicitly forcing the covariance to expand in underexplored eigenspace rather than reacting to stagnation or diversity metrics after the fact.

```python
def _adapt_covariance(self):
    """Eigenvalue floor with orthogonal perturbation for escaping deceptive local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # Force exploration along underexplored eigendirections
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
    except np.linalg.LinAlgError:
        eigvals = np.full(self.dim, np.mean(np.diag(self.C)))
        eigvecs = np.eye(self.dim)
    
    # Enforce eigenvalue floor to prevent spectrum collapse
    min_eig_target = max(1e-8, np.median(eigvals) * 1e-6)
    eigvals_clipped = np.maximum(eigvals, min_eig_target)
    
    # Detect under-explored eigendirections (smallest eigenvalues)
    median_eig = np.median(eigvals_clipped)
    small_mask = eigvals_clipped < (median_eig * 0.1)
    
    if np.any(small_mask):
        # Add structured perturbation along underexplored orthogonal directions
        small_indices = np.where(small_mask)[0]
        for idx in small_indices:
            # Perturb along this eigenvector direction
            perturbation = min_eig_target * 10.0 * np.outer(eigvecs[:, idx], eigvecs[:, idx])
            self.C = self.C + perturbation
    
    # Also add small isotropic perturbation to prevent full collapse
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = pop_variance / (expected_var + 1e-10)
    
    if diversity < 0.05:
        # Severe diversity loss - force exploration
        self.C = self.C + 0.01 * np.mean(np.diag(self.C)) * np.eye(self.dim)
    
    # If covariance is extremely ill-conditioned, reset along random orthogonal basis
    cond = np.max(eigvals_clipped) / (np.min(eigvals_clipped) + 1e-10)
    if cond > 1e7 or np.any(np.isnan(self.C)):
        # Reset to diagonal with floor
        diag_val = max(np.mean(np.diag(self.C)), min_eig_target)
        self.C = np.eye(self.dim) * diag_val
        self.pc = np.zeros(self.dim)
        self.L = None
    
    self.C = self._ensure_positive_definite(self.C)
```