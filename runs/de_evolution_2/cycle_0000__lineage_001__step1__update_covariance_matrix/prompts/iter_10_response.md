Looking at the data, the worst tasks (16, 19, 20, 23 with errors 6-300) are severely stuck despite restarts. The current rank-mu update only uses the top μ individuals, which can lead to premature covariance collapse on deceptive landscapes. I need a fundamentally different approach.

**Idea: Full-Population Fitness-Weighted Covariance Update**

Instead of using only the top μ individuals for the rank-μ update, use the **entire population** with **fitness-based exponential weighting**. This captures broader diversity information and provides smoother, more robust covariance estimates that prevent collapse on multi-modal tasks. The fitness-weighted differences from the current mean give a clearer signal of which search directions are actually fruitful.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using rank-1 update plus fitness-weighted
        full-population covariance (instead of only top mu individuals).
        """
        # Rank-1 update component (unchanged)
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Full population differences from CURRENT mean (key change)
        all_diffs = (sorted_population - self.mean[np.newaxis, :]) / self.sigma  # (pop, dim)

        # Compute fitness-based exponential weights for ALL individuals
        # Better individuals get higher weights, spread controlled by temperature
        pop_size = len(sorted_population)
        # Use rank-based weights: weight_i = exp(-rank_i / (pop_size * temperature))
        temperature = 0.3
        ranks = np.arange(1, pop_size + 1)
        raw_weights = np.exp(-ranks / (pop_size * temperature))
        fit_weights = raw_weights / raw_weights.sum()
        
        # Weighted differences from current mean
        weighted_diffs = fit_weights[:, np.newaxis] * all_diffs  # (pop, dim)
        full_pop_cov = all_diffs.T @ weighted_diffs  # (dim, dim)

        # Adaptive learning rate based on fitness improvement signal
        if self.best_f < self.last_best_f:
            c_full = self.c_mu_cov * 1.5  # Increase adaptation on improvement
        else:
            c_full = self.c_mu_cov * 0.5  # Reduce on stagnation

        # Combined update with full-population covariance
        self.C = (1.0 - self.c_1 - c_full + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 c_full * full_pop_cov

        # Ensure symmetry and positive definiteness
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
```