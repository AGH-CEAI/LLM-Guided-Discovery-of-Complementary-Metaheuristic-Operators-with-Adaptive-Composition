**Idea: Adaptive Momentum with Current-Mean Rank-Mu**

The current implementation uses `old_mean` for rank-mu (losing directional information from the mean update) and fixed learning rates. The worst unsolved tasks (16, 19, 20, 23) have errors > 10, suggesting the covariance adaptation is too conservative. This variant adds adaptive momentum that scales learning rates by mean-shift quality and uses the current mean for rank-mu to capture the latest directional information.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using adaptive rank-1 and rank-mu updates.
        Uses current mean for rank-mu computation and momentum-based adaptation.
        """
        # Compute mean shift quality for adaptive learning
        mean_shift = self.mean - old_mean
        mean_shift_norm = np.linalg.norm(mean_shift)
        
        # Adaptive momentum: scale by how much the mean moved (relative to sigma)
        # Larger shifts → more aggressive adaptation
        shift_ratio = mean_shift_norm / (self.sigma + 1e-20)
        adaptive_momentum = np.clip(0.9 ** (1.0 / (1.0 + shift_ratio)), 0.85, 0.99)
        
        # Rank-1 update component
        rank_one = np.outer(self.p_c, self.p_c)
        
        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        
        # Adaptive learning rates based on mean shift quality
        # When making good progress, increase learning rates
        progress_factor = np.clip(shift_ratio / 10.0, 0.5, 2.0)
        adaptive_c_1 = self.c_1 * progress_factor
        adaptive_c_mu = self.c_mu_cov * progress_factor
        
        # Rank-mu update: use CURRENT mean (not old_mean) to capture latest direction
        selected = sorted_population[:self.mu]
        diffs = (selected - self.mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu = diffs.T @ weighted_diffs  # (dim, dim)
        
        # Combined update with adaptive momentum
        self.C = (adaptive_momentum * (1.0 - adaptive_c_1 - adaptive_c_mu) + 
                  (1.0 - adaptive_momentum)) * self.C + \
                 adaptive_c_1 * rank_one + \
                 adaptive_c_mu * rank_mu + \
                 delta_h * self.c_1 * np.eye(self.dim)
        
        # Ensure symmetry and positive definiteness
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigvals = np.linalg.eigvalsh(self.C)
        min_eig = np.min(eigvals)
        if min_eig < 1e-12:
            self.C += (1e-11 - min_eig) * np.eye(self.dim)
```