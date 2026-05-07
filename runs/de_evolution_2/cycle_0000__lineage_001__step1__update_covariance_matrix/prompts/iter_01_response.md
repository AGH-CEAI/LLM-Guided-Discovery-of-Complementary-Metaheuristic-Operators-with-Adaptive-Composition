**Idea: Active CMA-ES Covariance Update**
Uses negative weights on worst individuals to actively decrease variance in unfavorable search directions, with improved condition number management via eigenvalue clamping.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using active CMA-ES strategy.
        Uses both best (positive) and worst (negative) weighted individuals
        for more robust covariance adaptation.
        """
        dim = self.dim
        
        # Rank-1 update component (evolution path)
        rank_one = np.outer(self.p_c, self.p_c)
        
        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        
        # Get selected individuals
        selected_best = sorted_population[:self.mu]
        
        # Positive rank-mu update from best individuals
        diffs_pos = (selected_best - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs_pos = self.weights[:, np.newaxis] * diffs_pos
        rank_mu_pos = diffs_pos.T @ weighted_diffs_pos
        
        # Active CMA-ES: negative rank-mu update from worst individuals
        # Use negative weights on worst individuals to decrease variance
        n_neg = min(self.mu, len(sorted_population) - self.mu)
        selected_worst = sorted_population[-n_neg:]
        
        # Compute negative weights (reverse order of worst individuals)
        neg_raw_weights = np.log(n_neg + 0.5) - np.log(np.arange(1, n_neg + 1))
        neg_weights = -neg_raw_weights / np.sum(neg_raw_weights) * 0.5  # Scaled negative weights
        
        # Differences for worst individuals
        diffs_neg = (selected_worst - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs_neg = neg_weights[:, np.newaxis] * diffs_neg
        rank_mu_neg = diffs_neg.T @ weighted_diffs_neg
        
        # Combined rank-mu (positive - negative)
        rank_mu = rank_mu_pos - rank_mu_neg
        
        # Learning rate adjustment for active CMA-ES
        c_1_eff = self.c_1
        c_mu_eff = self.c_mu_cov
        
        # Combined update
        self.C = (1.0 - c_1_eff - c_mu_eff + delta_h * c_1_eff) * self.C + \
                 c_1_eff * rank_one + \
                 c_mu_eff * rank_mu
        
        # Ensure symmetry (numerical robustness)
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        
        # Eigenvalue clamping for condition number control
        eigenvals = np.linalg.eigvalsh(self.C)
        eigenvals = np.maximum(eigenvals, 1e-20)
        
        # Clamp condition number to prevent extreme elongation
        max_eigen = np.max(eigenvals)
        min_eigen = np.min(eigenvals)
        cond_limit = 1e12
        
        if max_eigen / min_eigen > cond_limit:
            scale_factor = np.sqrt(max_eigen / (min_eigen * cond_limit))
            self.C *= scale_factor
        
        # Ensure positive definiteness
        min_eigen_check = np.min(np.linalg.eigvalsh(self.C))
        if min_eigen_check < 1e-15:
            self.C += np.eye(dim) * (1e-14 - min_eigen_check)
```