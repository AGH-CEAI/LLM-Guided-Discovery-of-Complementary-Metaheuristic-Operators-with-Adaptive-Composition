**Idea: Exponential Fitness-Weighted Rank-Mu Covariance Update**
Replace the log-linear rank weights with exponential fitness-based weights for the covariance update, emphasizing the best solutions much more strongly to break through deceptive local optima on the worst tasks.
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update the covariance matrix using exponential fitness-weighted rank-mu update."""
    dim = self.dim
    selected = sorted_population[:self.mu]
    
    # Rank-1 update component (unchanged)
    rank1 = np.outer(self.p_c, self.p_c)
    
    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
    
    # Exponential fitness-based weights for covariance: emphasize best solutions
    # This is fundamentally different from log-linear weights used elsewhere
    inv_ranks = 1.0 / np.arange(1, self.mu + 1)
    exp_weights = np.exp(-1.5 * inv_ranks)  # exponential decay over rank
    exp_weights = exp_weights / np.sum(exp_weights)
    mu_eff_exp = 1.0 / np.sum(exp_weights ** 2)
    
    # Adaptive learning rate based on effective weight concentration
    c_mu_adaptive = min(0.9, self.c_mu_cov * (1.0 + 0.5 * (mu_eff_exp - 1.0)))
    
    # Rank-mu update with exponential weights (more aggressive selection)
    diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
    rank_mu = np.zeros((dim, dim))
    for i in range(self.mu):
        rank_mu += exp_weights[i] * np.outer(diff[i], diff[i])
    
    # Combined update with adaptive learning rate
    self.C = ((1.0 - self.c_1 - c_mu_adaptive + delta_h * self.c_1) * self.C +
               self.c_1 * rank1 +
               c_mu_adaptive * rank_mu)
    
    # Enforce symmetry and add slight diagonal push for numerical stability
    self.C = 0.5 * (self.C + self.C.T)
    self.C.flat[::dim + 1] += 1e-10 * np.ones(dim)
```