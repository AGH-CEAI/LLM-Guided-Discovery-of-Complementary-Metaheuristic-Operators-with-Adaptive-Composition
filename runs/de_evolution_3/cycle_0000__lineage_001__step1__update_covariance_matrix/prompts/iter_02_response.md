**Idea: Active Covariance with Negative Weights**
Replaces rank-mu update with an *active CMA* approach that uses negative weights on poorly-ranked individuals to penalize already-explored regions and improve exploration on deceptive/multimodal tasks.
```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update the covariance matrix with rank-1 and active rank-mu updates."""
    dim = self.dim
    lam = self.pop_size

    # Rank-1 update component
    rank1 = np.outer(self.p_c, self.p_c)

    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

    # Active CMA: use ALL individuals with positive/negative weights
    # Positive weights for top half, negative weights for bottom half
    diff = (sorted_population - old_mean[None, :]) / self.sigma  # (lambda, dim)

    # Compute positive weights (top half)
    mu_pos = self.mu
    raw_pos = np.log(mu_pos + 0.5) - np.log(np.arange(1, mu_pos + 1))
    pos_weights = raw_pos / np.sum(raw_pos)

    # Compute negative weights (bottom half) with smaller magnitude
    mu_neg = lam - self.mu
    raw_neg = np.log(mu_neg + 0.5) - np.log(np.arange(1, mu_neg + 1))
    neg_weights = -raw_neg / np.sum(raw_neg) * 0.4  # scaled negative weights

    # Combine weights for all lambda individuals
    all_weights = np.concatenate([pos_weights, neg_weights])
    all_weights = np.clip(all_weights, -0.5, None)  # clip negative extremes

    # Normalize so sum of abs(weights) = 1 for consistent learning rate
    weight_sum = np.sum(np.abs(all_weights))
    if weight_sum > 1e-30:
        all_weights = all_weights / weight_sum

    # Compute rank-mu update using full population
    rank_mu = np.zeros((dim, dim))
    for i in range(lam):
        w = all_weights[i]
        if abs(w) > 1e-30:
            rank_mu += w * np.outer(diff[i], diff[i])

    # Combined update
    self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
               self.c_1 * rank1 +
               self.c_mu_cov * rank_mu)

    # Enforce symmetry and positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    # Add minimal diagonal if needed for numerical stability
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-15:
        self.C += (1e-14 - min_eig) * np.eye(dim)
```