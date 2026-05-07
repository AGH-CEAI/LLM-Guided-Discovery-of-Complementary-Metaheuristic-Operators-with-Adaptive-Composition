Looking at the unsolved tasks (errors 1e+1 to 4e+2), these are clearly hard optimization problems where the current CMA-ES adaptation is failing to converge. The key issue: **the rank-mu update uses `old_mean` as reference, which becomes stale when the mean shifts significantly across basins in multimodal landscapes**. This causes the covariance to "collapse" toward the direction of improvement rather than preserving diversity.

**Idea: Best-Anchored Rank-Mu with Diversity Preservation**

Instead of anchoring rank-mu to `old_mean` (which loses inter-basin diversity), anchor it to the **current best solution** and compute weighted covariance across top performers. This preserves exploration history and prevents premature concentration into a single basin.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Best-anchored rank-mu update for diversity preservation.
        Key insight: anchoring to old_mean loses inter-basin diversity.
        Anchoring to current best preserves exploration across regions.
        """
        dim = self.dim
        
        # Rank-1 update (evolution path - unchanged)
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        
        # Use current best as anchor (not old_mean!)
        # This preserves diversity across different basins found
        best_solution = sorted_population[0]
        
        # Top performers centered on best (not old_mean)
        selected = sorted_population[:self.mu]
        diffs = (selected - best_solution[np.newaxis, :]) / max(self.sigma, 1e-30)
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs
        
        # Adaptive dampening based on condition number
        cond = self.D[-1] / max(self.D[0], 1e-30)
        if cond > 1e7:
            damp = 0.2
        elif cond > 1e5:
            damp = 0.5
        else:
            damp = 1.0
        
        # Stagnation detection for exploration boost
        stagnation = (self.stagnation_counter > self.dim)
        
        # Effective rank-mu learning rate
        c_mu_eff = self.c_mu_cov * damp
        if stagnation:
            c_mu_eff *= 2.5  # Boost exploration when stuck
        
        # Combined update
        self.C = (1.0 - self.c_1 - c_mu_eff + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 c_mu_eff * rank_mu
        
        # Ensure symmetry and positive definiteness
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigvals = np.linalg.eigvalsh(self.C)
        if np.min(eigvals) < 1e-15:
            self.C += np.eye(dim) * (1e-14 - np.min(eigvals))
```