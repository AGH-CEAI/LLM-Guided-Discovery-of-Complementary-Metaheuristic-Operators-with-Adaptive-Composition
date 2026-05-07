**Idea: Condition-Number-Driven Adaptive Learning Rate**

A fundamentally different covariance update that uses the covariance matrix's condition number as a continuous difficulty signal to dynamically scale the rank-1 learning rate. When the condition number is high (ill-conditioned = harder), the learning rate increases to enable aggressive exploration; when well-conditioned, it remains conservative. This directly targets the stuck-at-high-error behavior on Tasks 16-23 where the optimizer fails to explore out of local basins.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update covariance with condition-number-driven adaptive learning rate."""
    dim = self.dim
    selected = sorted_population[:self.mu]

    # Compute condition number as continuous difficulty signal
    cond = np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-30)
    cond = np.clip(cond, 1.0, 1e15)

    # Map condition number to adaptive learning rate via log-sigmoid
    # Low cond -> low rate (conservative), high cond -> high rate (aggressive)
    log_cond = np.log10(cond + 1e-30)
    alpha = 2.0 / (1.0 + np.exp(-0.5 * (log_cond - 4.0))) - 1.0
    alpha = np.clip(alpha, 0.0, 1.0)

    # Rank-1 update with adaptive learning rate
    rank1 = np.outer(self.p_c, self.p_c)
    c_1_adaptive = self.c_1 * (0.1 + 0.9 * alpha)

    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

    # Rank-mu update with complementary adaptive scaling
    diff = (selected - old_mean[None, :]) / self.sigma
    rank_mu = np.zeros((dim, dim))
    for i in range(self.mu):
        rank_mu += self.weights[i] * np.outer(diff[i], diff[i])
    c_mu_adaptive = self.c_mu_cov * (0.5 + 0.5 * alpha)

    # Combined update with adaptive rates
    self.C = ((1.0 - c_1_adaptive - c_mu_adaptive + delta_h * c_1_adaptive) * self.C +
               c_1_adaptive * rank1 +
               c_mu_adaptive * rank_mu)

    # Enforce symmetry
    self.C = 0.5 * (self.C + self.C.T)
```