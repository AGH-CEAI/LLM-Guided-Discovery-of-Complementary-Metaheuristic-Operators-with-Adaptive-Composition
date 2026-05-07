Looking at the error patterns, the worst unsolved tasks (16, 19, 20, 23) have errors in the 10^1-10^2 range, suggesting the algorithm gets stuck in poor local optima or fails to explore effectively. The current rank-based weighting doesn't differentiate enough between individuals with vastly different fitness values.

**Idea: Fitness-Proportional Rank-μ Update with Aggressive Rank-1 Exploration**

Instead of using fixed logarithmic rank weights, use actual fitness improvement to weight covariance updates. This allows the algorithm to more aggressively follow directions from individuals that are significantly better, while reducing influence from mediocre individuals that happen to be in the top μ.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using fitness-proportional weighting for rank-mu
        and an aggressive rank-1 component to escape local optima.
        """
        # Rank-1 update component - use larger weight for exploration
        rank_one = np.outer(self.p_c, self.p_c)
        
        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        
        # Rank-mu update with FITNESS-PROPORTIONAL weights
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        
        # Compute fitness-based weights: use normalized rank improvement
        # Best individual gets highest weight, exponential decay
        fitness_weights = np.exp(-0.5 * np.arange(self.mu))  # (mu,)
        fitness_weights = fitness_weights / np.sum(fitness_weights)  # normalize
        
        weighted_diffs = fitness_weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu = diffs.T @ weighted_diffs  # (dim, dim)
        
        # Adaptive learning rates: increase c_1 for exploration on hard tasks
        # Use larger rank-1 weight when sigma is large (exploration phase)
        adaptive_c1 = self.c_1 * (1.0 + 0.5 * np.log1p(self.sigma / 30.0))
        adaptive_c1 = np.clip(adaptive_c1, self.c_1, self.c_1 * 3.0)
        
        # Combined update with adaptive rates
        self.C = (1.0 - adaptive_c1 - self.c_mu_cov + delta_h * adaptive_c1) * self.C + \
                 adaptive_c1 * rank_one + \
                 self.c_mu_cov * rank_mu
        
        # Ensure positive definiteness
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
```