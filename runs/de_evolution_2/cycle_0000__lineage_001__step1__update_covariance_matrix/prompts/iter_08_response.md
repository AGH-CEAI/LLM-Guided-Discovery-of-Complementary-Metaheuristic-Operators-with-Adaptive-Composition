**Idea: Negative-Weight Active Covariance Update**

The worst unsolved tasks (16, 19, 20, 23) have errors stuck at 1e+0 to 3e+2, indicating the algorithm converges to poor local optima and fails to escape. The current covariance update only uses positive weights on the top μ individuals, so it never "repels" the search from bad regions. This variant introduces **negative recombination weights** for the worst individuals in the covariance update, actively steering the distribution away from poorly-performing regions — a fundamentally different adaptation mechanism that can break out of deceptive local optima.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using rank-1, rank-mu with positive weights,
        and active covariance update with negative weights (CMA-ES active update).
        """
        # Rank-1 update component
        rank_one = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Standard rank-mu update with positive weights
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu_pos = diffs.T @ weighted_diffs  # (dim, dim)

        # Active covariance update: use worst individuals with negative weights
        n_neg = max(1, self.pop_size // 4)
        neg_start = max(self.mu, self.pop_size - n_neg)
        worst = sorted_population[neg_start:]  # bottom performers
        if len(worst) >= 2:
            neg_diffs = (worst - old_mean[np.newaxis, :]) / self.sigma  # (n_neg, dim)
            # Negative weights: proportional to 1/rank, normalized to sum to -0.5
            neg_raw_weights = np.log(self.pop_size + 0.5) - np.log(np.arange(neg_start + 1, self.pop_size + 1))
            neg_weights = neg_raw_weights / np.sum(np.abs(neg_raw_weights)) * (-0.5)
            neg_weighted_diffs = neg_weights[:, np.newaxis] * neg_diffs  # (n_neg, dim)
            rank_mu_neg = neg_diffs.T @ neg_weighted_diffs  # (dim, dim)
        else:
            rank_mu_neg = np.zeros((self.dim, self.dim))

        # Combined update with active component
        c_pos = self.c_mu_cov
        c_neg = min(0.5, c_pos * 0.5)  # scale negative component
        self.C = (1.0 - self.c_1 - c_pos - c_neg + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 c_pos * rank_mu_pos + \
                 c_neg * rank_mu_neg

        # Ensure symmetry and positive definiteness
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        # Add small identity for numerical stability
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-12:
            self.C += (1e-11 - min_eig) * np.eye(self.dim)
```