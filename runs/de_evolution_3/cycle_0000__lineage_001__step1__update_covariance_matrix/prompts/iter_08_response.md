**Idea: Adaptive Eigenvalue-Bounded Covariance Update**

The worst unsolved tasks (16, 19, 23, 20) have errors of 3.9 to 40, indicating severe ill-conditioning or multi-modality where the standard covariance update fails to adapt quickly enough. This approach uses: (1) adaptive learning rates scaled by the covariance condition number to accelerate learning on hard problems, (2) eigenvalue bounding via eigendecomposition to ensure positive-definiteness and prevent numerical explosion, and (3) incorporating diagonal variance from selected solutions for more robust rank-mu updates.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update covariance with adaptive learning rates and eigenvalue bounding."""
    dim = self.dim
    selected = sorted_population[:self.mu]
    
    # Rank-1 update component
    rank1 = np.outer(self.p_c, self.p_c)
    
    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
    
    # Rank-mu update component with variance stabilization
    diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
    rank_mu = np.zeros((dim, dim))
    for i in range(self.mu):
        rank_mu += self.weights[i] * np.outer(diff[i], diff[i])
    
    # Add diagonal variance from selected solutions for stability
    selected_var = np.var(selected, axis=0) / (self.sigma ** 2 + 1e-20)
    diag_contribution = np.sum(self.weights) * np.mean(self.weights) * np.diag(selected_var)
    
    # Compute condition number for adaptive learning
    eigenvalues = self.eigenvalues
    cond = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-30)
    cond = min(cond, 1e14)
    
    # Adaptive learning rates: increase for ill-conditioned problems
    adapt_factor = 1.0 + 2.0 * np.log1p(cond) / np.log1p(1e14)
    c_1_adapt = min(self.c_1 * adapt_factor, 0.5)
    c_mu_adapt = min(self.c_mu_cov * adapt_factor, 0.5)
    
    # Combined update with adaptive rates and diagonal contribution
    self.C = ((1.0 - c_1_adapt - c_mu_adapt + delta_h * c_1_adapt) * self.C +
               c_1_adapt * rank1 +
               c_mu_adapt * (rank_mu + 0.1 * diag_contribution))
    
    # Symmetrize
    self.C = 0.5 * (self.C + self.C.T)
    
    # Ensure positive definiteness via eigenvalue bounding
    eigenvalues, eigenvectors = np.linalg.eigh(self.C)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    self.C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    self.C = 0.5 * (self.C + self.C.T)
```