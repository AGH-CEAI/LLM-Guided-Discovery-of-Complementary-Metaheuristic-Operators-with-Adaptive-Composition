**Idea: Aggressive Covariance with Negative Weights and Diagonal Boost**
One-line description: Use negative rank-μ weights, increased learning rates, and a diagonal boost to escape local optima on the hardest tasks (16, 19, 20, 23).
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update covariance with negative weights, higher learning rates, and diagonal boost."""
    dim = self.dim
    selected = sorted_population[:self.mu]
    all_candidates = sorted_population[:self.pop_size]

    # Compute normalized differences
    diff_selected = (selected - old_mean[None, :]) / (self.sigma + 1e-20)  # (mu, dim)
    diff_all = (all_candidates - old_mean[None, :]) / (self.sigma + 1e-20)  # (lam, dim)

    # Build negative weights: negative for top performers, positive for rest
    n_neg = max(1, self.mu // 3)
    neg_weights = np.zeros(self.mu)
    neg_weights[:n_neg] = -np.abs(self.weights[:n_neg]) * 2.0  # strong negative for worst in top-mu
    neg_weights[n_neg:] = self.weights[n_neg:] * 0.5

    # Rank-1 update: use evolution path with higher weight
    rank1 = np.outer(self.p_c, self.p_c)
    c_1_boost = min(0.1, self.c_1 * 5.0)  # 5x increase, capped

    # Rank-mu update with negative weights
    rank_mu = np.zeros((dim, dim))
    for i in range(self.mu):
        rank_mu += neg_weights[i] * np.outer(diff_selected[i], diff_selected[i])

    # Diversity term: push away from entire population spread
    pop_spread = np.var(all_candidates, axis=0) / (self.sigma ** 2 + 1e-20)
    diversity_term = np.diag(pop_spread * 0.01)

    # Covariance learning rate with damping
    c_1_eff = c_1_boost * h_sigma
    c_mu_eff = min(0.9, self.c_mu_cov * 3.0)  # 3x increase, capped at 0.9

    # Delta for h_sigma correction
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

    # Combined update with diagonal boost
    self.C = ((1.0 - c_1_eff - c_mu_eff + delta_h * c_1_eff) * self.C +
               c_1_eff * rank1 +
               c_mu_eff * rank_mu +
               diversity_term)

    # Diagonal dominance: mix toward identity for exploration
    diag_strength = 0.3
    self.C = (1.0 - diag_strength) * self.C + diag_strength * np.diag(np.diag(self.C))
    self.C = (1.0 - diag_strength) * self.C + diag_strength * np.eye(dim)

    # Enforce symmetry and positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    eigvals = np.linalg.eigvalsh(self.C)
    if np.min(eigvals) < 1e-19:
        self.C += (1e-18 - np.min(eigvals)) * np.eye(dim)
```