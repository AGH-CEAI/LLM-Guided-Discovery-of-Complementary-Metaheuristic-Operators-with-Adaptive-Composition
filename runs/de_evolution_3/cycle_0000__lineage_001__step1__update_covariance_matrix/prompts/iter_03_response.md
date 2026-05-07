**Idea: Elite-Weighted Covariance with Adaptive Quality Factor**

A fundamentally different approach: instead of relying solely on the evolution path (p_c) for rank-1 updates, incorporate the **best solution's displacement** scaled by its quality relative to the population. This directly targets the worst tasks (16, 19, 20, 23) which appear stuck at high errors — likely due to the evolution path being too slow/conservative for deceptive landscapes. The key mechanism: if the best solution shows significant improvement, amplify its influence on the covariance matrix to accelerate movement in the right direction.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update covariance using adaptive elite-weighted rank-1 + rank-mu updates."""
    dim = self.dim
    selected = sorted_population[:self.mu]
    best = selected[0]
    worst = selected[-1]

    # Compute quality factor based on best solution improvement
    mean_fitness = np.mean([sorted_population[i][0] if hasattr(sorted_population[i], '__getitem__') else 0 
                            for i in range(min(3, len(sorted_population)))])
    disp_best = (best - old_mean) / (self.sigma + 1e-30)
    disp_best_norm = np.linalg.norm(disp_best)

    # Adaptive weight: stronger when best shows clear direction
    quality_factor = min(1.0, disp_best_norm / (np.sqrt(dim) * 0.5 + 1e-30))
    elite_weight = self.c_1 * (0.5 + 2.0 * quality_factor)

    # Rank-1 update using best solution displacement (not evolution path)
    rank1_elite = np.outer(disp_best, disp_best)

    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

    # Rank-mu update with reweighted contributions (emphasize top performers)
    diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
    rank_mu = np.zeros((dim, dim))
    for i in range(self.mu):
        # Amplify weights for top-2 solutions
        weight_mult = 2.0 if i < 2 else 1.0
        rank_mu += self.weights[i] * weight_mult * np.outer(diff[i], diff[i])
    # Renormalize rank_mu contribution
    rank_mu *= (1.0 / max(1.5, self.mu / 2.0))

    # Combined update
    self.C = ((1.0 - elite_weight - self.c_mu_cov + delta_h * elite_weight) * self.C +
               elite_weight * rank1_elite +
               self.c_mu_cov * rank_mu)

    # Enforce symmetry and positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    # Ensure diagonal dominance for numerical stability
    diag = np.diag(self.C)
    min_diag = 1e-20
    if np.any(diag < min_diag):
        self.C += np.diag(np.maximum(0, min_diag - diag))
```