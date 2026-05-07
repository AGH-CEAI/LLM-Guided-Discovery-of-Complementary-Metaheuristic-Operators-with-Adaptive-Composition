**Idea: Diversity-Modulated Adaptive Learning Rate**

The worst unsolved tasks (16, 20, 19, 23 with errors 1e+1 to 3e+2) are stuck in a regime where the covariance matrix isn't adapting fast enough. The key insight is that ALL previous variants use static learning rates (`c_1`, `c_mu_cov`) that never adapt to the problem's conditioning. This variant dynamically scales the learning rates based on population diversity and the magnitude of the evolution path, allowing the covariance to explode toward informative directions on ill-conditioned tasks while staying conservative on well-conditioned ones.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Diversity-modulated adaptive covariance update.
        Learning rates scale with population spread and evolution path length,
        enabling aggressive adaptation on ill-conditioned tasks.
        """
        # Rank-1 component
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        
        # Rank-mu component
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs
        
        # Adaptive diversity-based scaling
        # Measure population spread in the current basis
        pop_spread = np.mean(np.std(selected, axis=0) ** 2)
        spread_ratio = pop_spread / (np.trace(self.C) / max(self.dim, 1) + 1e-20)
        
        # Evolution path magnitude as condition indicator
        path_norm = np.linalg.norm(self.p_c) + 1e-20
        path_adaptive = min(path_norm / np.sqrt(self.dim), 3.0)
        
        # Diversity modulation: boost learning when population is diverse
        diversity_boost = 1.0 + 2.0 * np.tanh(spread_ratio - 0.5)
        
        # Condition-adaptive scaling: amplify on ill-conditioned landscapes
        condition_factor = 1.0 + path_adaptive
        
        # Compute adaptive learning rates
        c_1_adaptive = self.c_1 * diversity_boost * condition_factor
        c_mu_adaptive = self.c_mu_cov * diversity_boost * condition_factor
        
        # Cap to maintain positive semidefiniteness
        c_1_adaptive = min(c_1_adaptive, 0.5)
        c_mu_adaptive = min(c_mu_adaptive, 0.9 - c_1_adaptive)
        
        # Combined update with adaptive rates
        self.C = (1.0 - c_1_adaptive - c_mu_adaptive + delta_h * c_1_adaptive) * self.C + \
                 c_1_adaptive * rank_one + \
                 c_mu_adaptive * rank_mu
        
        # Ensure symmetry and numerical health
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
```