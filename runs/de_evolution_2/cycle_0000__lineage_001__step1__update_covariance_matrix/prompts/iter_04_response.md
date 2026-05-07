**Idea: Adaptive Scaled Rank-Mu with Mean Shift Injection**
Uses adaptive learning rates that increase with population diversity and injects mean shift information into the covariance update to break through stagnation on multimodal/ill-conditioned tasks.
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """
    Update covariance matrix with adaptive scaling based on population diversity
    and mean shift injection to improve exploration on difficult tasks.
    """
    # Compute population diversity (variance of selected individuals)
    selected = sorted_population[:self.mu]
    pop_variance = np.var(selected, axis=0)
    avg_variance = np.mean(pop_variance)
    eigvals = np.linalg.eigvalsh(self.C)
    cond_num = np.max(eigvals) / max(np.min(eigvals), 1e-30)
    diversity_factor = np.clip(avg_variance / (np.mean(np.diag(self.C)) + 1e-30), 0.1, 10.0)
    
    # Adaptive learning rates: higher when diverse, lower when converged
    c_1_adapt = self.c_1 * (1.0 + 0.5 * np.log(1.0 + cond_num) / np.log(1e10))
    c_mu_adapt = self.c_mu_cov * diversity_factor
    
    # Rank-1 update with adaptive scaling
    rank_one = np.outer(self.p_c, self.p_c)
    
    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
    
    # Rank-mu update with adaptive weights
    diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
    weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
    rank_mu = diffs.T @ weighted_diffs  # (dim, dim)
    
    # Mean shift injection: capture direction of mean movement
    mean_shift = (self.mean - old_mean) / max(self.sigma, 1e-30)
    mean_shift_outer = np.outer(mean_shift, mean_shift)
    
    # Combine all components with adaptive rates
    base_decay = 1.0 - c_1_adapt - c_mu_adapt
    self.C = (base_decay + delta_h * c_1_adapt) * self.C + \
             c_1_adapt * rank_one + \
             c_mu_adapt * rank_mu + \
             0.1 * c_mu_adapt * mean_shift_outer
    
    # Ensure symmetry and positive definiteness
    self.C = np.triu(self.C) + np.triu(self.C, 1).T
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-15:
        self.C += (1e-14 - min_eig) * np.eye(self.dim)
```