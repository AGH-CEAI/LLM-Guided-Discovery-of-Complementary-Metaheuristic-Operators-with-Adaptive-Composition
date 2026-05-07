**Idea: Aggressive Covariance Reset with Step-Size Reduction**

The worst unsolved tasks (16-23) are stuck at ~1e+1 error, suggesting the optimizer is trapped in local optima due to a highly elongated covariance ellipse that can't escape. The key insight is that when the condition number becomes extreme (ill-conditioned landscape), the current approach only *scales down* learning rates, which slows convergence but doesn't actually change the search direction. This variant instead performs an **aggressive covariance matrix reset** combined with **step-size shrinking** — fundamentally reshaping the search distribution to escape the trap.

```python
def _adapt_covariance_variant_01(self):
    """Aggressive covariance reset with step-size reduction for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma

    # Compute condition number and eigenvalue spread
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

    # Check for stagnation indicators
    recent_improved = 0
    for i in range(min(len(self.trial_fitness), len(self.fitness))):
        if self.trial_fitness[i] < self.fitness[i]:
            recent_improved += 1
    success_rate = recent_improved / max(len(self.trial_fitness), 1)

    # Trigger aggressive reset when:
    # 1. Extreme condition number (ill-conditioned landscape)
    # 2. Very small eigenvalue spread (collapsed covariance)
    # 3. Stagnation with low success rate (trapped in local optimum)
    should_reset = (cond > 1e6) or (eig_spread < 1e-6) or \
                   (success_rate < 0.05 and self.stagnation_counter > 20)

    if should_reset:
        # Reset to spherical covariance (isotropic search)
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.L = None

        # Shrink step-size to refine search after reset
        self.sigma = max(self.sigma * 0.3, 1e-10)

        # Use conservative learning rates after reset
        ccov_1 = 0.5 / (self.dim + 2.0)
        ccov_mu = 0.5 * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = 0.3 * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
    else:
        # Normal adaptation when not in trap
        ccov_1 = self.ccov
        ccov_mu = self.ccov
        cc_adapt = self.cc

    # Evolution path update
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)

    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)

    # Combine updates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu)

    self.C = self._ensure_positive_definite(self.C)
```