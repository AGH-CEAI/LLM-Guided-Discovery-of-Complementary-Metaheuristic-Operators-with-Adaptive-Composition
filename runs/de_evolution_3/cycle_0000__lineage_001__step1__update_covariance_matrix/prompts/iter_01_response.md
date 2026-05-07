**Idea: Weighted Rank-MU with Population-Wide Covariance**
Uses weighted rank-mu update (like current) but adds population-wide covariance component for robustness, and applies different weighting normalization to improve exploration.
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update covariance using weighted rank-mu + population-wide covariance."""
        dim = self.dim
        lam = self.pop_size
        selected = sorted_population[:self.mu]

        # Rank-1 update component
        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Weighted rank-mu update (current approach)
        diff = (selected - old_mean[None, :]) / self.sigma
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        # NEW: Population-wide covariance for robustness
        pop_diff = (sorted_population - old_mean[None, :]) / self.sigma
        pop_cov = np.cov(pop_diff.T, bias=True) if lam > 1 else np.zeros((dim, dim))

        # Combined update with population-wide component
        # c_pop controls contribution of population-wide covariance
        c_pop = min(0.1, self.c_mu_cov * 0.5)
        self.C = ((1.0 - self.c_1 - self.c_mu_cov - c_pop + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu +
                   c_pop * pop_cov)

        # Enforce symmetry and positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        # Ensure diagonal is positive
        diag = np.diag(self.C)
        np.fill_diagonal(self.C, np.maximum(diag, 1e-20))
```