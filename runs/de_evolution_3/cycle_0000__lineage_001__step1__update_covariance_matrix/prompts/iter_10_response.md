Looking at the unsolved tasks, the worst ones (Tasks 16-23) are stuck at O(1) to O(40) error, suggesting the algorithm is failing to explore effectively or is getting trapped. The current covariance update relies heavily on the top-μ individuals, which can be problematic when the population converges to a poor region.

**Idea: Active CMA with Full-Population Covariance and Directional Damping**

This approach makes three fundamental changes: (1) uses ALL individuals with exponential rank weights (not just top μ), (2) applies active covariance with NEGATIVE weights for poor performers to actively shrink bad directions, and (3) adds a damping term that resists excessive elongation to prevent ill-conditioning.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Active CMA-ES covariance update with full-population rank-based weights."""
    dim = self.dim
    lam = self.pop_size

    # Compute rank-based weights for ALL individuals (not just top mu)
    # Use negative weights for poor performers (active CMA)
    ranks = np.arange(1, lam + 1)
    # Positive weights for good half, negative for bad half
    raw_weights = np.where(
        ranks <= lam / 2,
        np.log(lam / 2 + 0.5) - np.log(ranks),  # positive for top half
        np.log(lam / 2 + 0.5) - np.log(lam - ranks + 1) - 0.25  # small negative for bottom half
    )
    # Normalize positive and negative weights separately
    pos_mask = raw_weights > 0
    neg_mask = ~pos_mask
    pos_sum = np.sum(raw_weights[pos_mask]) + 1e-30
    neg_sum = np.abs(np.sum(raw_weights[neg_mask])) + 1e-30
    
    normalized_weights = np.zeros(lam)
    normalized_weights[pos_mask] = (1.0 + dim) * raw_weights[pos_mask] / pos_sum
    normalized_weights[neg_mask] = -0.25 * raw_weights[neg_mask] / neg_sum
    
    # Rank-1 update component (evolution path)
    rank1 = np.outer(self.p_c, self.p_c)
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

    # Full-population rank-mu update using differences from mean
    diff = (sorted_population - old_mean[None, :]) / self.sigma  # (lam, dim)
    
    # Compute weighted outer products efficiently
    rank_mu = np.dot(
        normalized_weights[:, None] * diff,  # weighted differences
        diff
    )

    # Add small identity regularization for numerical stability
    reg_term = 1e-8 * np.eye(dim)

    # Combined update with adaptive learning rate
    c_cov = self.c_1 + self.c_mu_cov
    self.C = ((1.0 - c_cov + delta_h * self.c_1) * self.C +
               self.c_1 * rank1 +
               self.c_mu_cov * rank_mu +
               reg_term)

    # Enforce symmetry and positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    
    # Clamp smallest eigenvalue to prevent singularity
    min_eig = 1e-20
    current_min = np.min(np.linalg.eigvalsh(self.C))
    if current_min < min_eig:
        self.C += (min_eig - current_min) * np.eye(dim)
```