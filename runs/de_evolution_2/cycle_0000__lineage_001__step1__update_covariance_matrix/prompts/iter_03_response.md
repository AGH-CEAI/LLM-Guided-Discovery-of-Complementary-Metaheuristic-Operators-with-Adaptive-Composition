**Idea: Full-Population Rank-Mu with Dominant Rank-1**

The worst unsolved tasks (16, 19, 20, 23 with errors 10^1–10^2) are stuck at extremely poor fitness, suggesting the CMA-ES adaptation is too slow and conservative. The current rank-mu update only uses top μ individuals, which provides sparse information about the fitness landscape. This variant uses ALL individuals for rank-mu estimation (with exponential decay weights) and DRAMATICALLY increases rank-1 influence (c_1 → 0.3, c_mu_cov → 0.6) to make the covariance matrix highly responsive to the evolution path. This should escape local optima on the hardest tasks.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using full-population rank-mu update
        with dominant rank-1 component for aggressive adaptation.
        """
        dim = self.dim
        pop_size = self.pop_size

        # Full-population rank-mu update using ALL individuals
        all_selected = sorted_population[:pop_size]
        diffs = (all_selected - old_mean[np.newaxis, :]) / self.sigma  # (pop_size, dim)

        # Exponential decay weights for ALL individuals (not just top mu)
        raw_weights_full = np.exp(-np.arange(pop_size) / (0.3 * pop_size))
        weights_full = raw_weights_full / np.sum(raw_weights_full)

        # Rank-mu matrix from full population
        weighted_diffs_full = weights_full[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs_full

        # Dominant rank-1 update (much higher c_1)
        rank_one = np.outer(self.p_c, self.p_c)
        c_1_aggressive = 0.3
        c_mu_aggressive = 0.6

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Combined update with aggressive learning rates
        self.C = (1.0 - c_1_aggressive - c_mu_aggressive + delta_h * c_1_aggressive) * self.C + \
                 c_1_aggressive * rank_one + \
                 c_mu_aggressive * rank_mu

        # Ensure numerical stability
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        min_eigen = 1e-20
        if np.any(np.linalg.eigvalsh(self.C) < min_eigen):
            self.C += (min_eigen - np.min(np.linalg.eigvalsh(self.C))) * np.eye(dim)
```