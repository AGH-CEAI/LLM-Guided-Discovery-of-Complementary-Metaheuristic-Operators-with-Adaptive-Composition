**Idea: Exploration-Driven Covariance with Stagnation-Triggered Identity Injection**

The worst unsolved tasks (16, 19, 20, 23) have errors stuck at 1e+0 to 5e+1 — indicating the algorithm is trapped in local optima rather than slowly converging. The standard rank-1 + rank-mu update is too conservative; when stagnation occurs, the covariance matrix becomes too peaked and sigma collapses, preventing escape. This variant injects an explicit identity component scaled by stagnation level, forcing wider exploration in all directions simultaneously — fundamentally different from relying on accumulated step history.

```python
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
    """Update covariance matrix with stagnation-triggered diversification via identity injection."""
    dim = self.dim
    selected = sorted_population[:self.mu]

    # Rank-1 update component
    rank1 = np.outer(self.p_c, self.p_c)

    # Correction for h_sigma
    delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

    # Rank-mu update component
    diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
    rank_mu = np.zeros((dim, dim))
    for i in range(self.mu):
        rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

    # Compute condition number for adaptive diversification
    eig_min = np.min(self.eigenvalues) + 1e-30
    eig_max = np.max(self.eigenvalues) + 1e-30
    cond_num = eig_max / eig_min

    # Stagnation level drives exploration strength (0 to ~0.3)
    stagnation_ratio = min(self.stagnation_counter / max(1, 10 + int(30 * dim / self.pop_size)), 1.0)

    # Diversification coefficient: identity injection for wider search
    # Higher when stagnant OR when covariance is ill-conditioned
    c_diversify = 0.1 * stagnation_ratio + 0.05 * min(cond_num / 1e10, 1.0)
    c_diversify = min(c_diversify, 0.3)  # cap to avoid instability

    # Combined update with identity injection
    self.C = ((1.0 - self.c_1 - self.c_mu_cov - c_diversify + delta_h * self.c_1) * self.C +
               self.c_1 * rank1 +
               self.c_mu_cov * rank_mu +
               c_diversify * np.eye(dim))

    # Enforce symmetry and positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    # Ensure no negative eigenvalues due to numerical errors
    eigvals = np.linalg.eigvalsh(self.C)
    if np.min(eigvals) < 1e-14:
        self.C += (1e-13 - np.min(eigvals)) * np.eye(dim)
```