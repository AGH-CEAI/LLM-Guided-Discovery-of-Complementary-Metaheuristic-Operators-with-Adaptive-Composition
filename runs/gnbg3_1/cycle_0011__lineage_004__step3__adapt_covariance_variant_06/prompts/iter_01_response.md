**Idea: Active CMA-ES with Negative Weights**
Uses active covariance update that incorporates poorly-performing individuals to decrease variance in unfavorable directions, improving exploration and escape from local optima.

```python
def _adapt_covariance_variant_06(self):
    """Active CMA-ES with negative weights for improved exploration."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update with positive weights (standard)
    rank_mu_pos = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu_pos += self.weights[i] * np.outer(diff, diff)
    
    # Active CMA-ES: Rank-mu update with negative weights (worst individuals)
    # Use bottom mu individuals with negative weights
    neg_mu = max(1, self.mu // 2)
    neg_weights = np.zeros(neg_mu)
    sum_neg = 0.0
    for i in range(neg_mu):
        neg_weights[i] = -(np.log((self.NP + i) / 2.0 + 0.5) - np.log(self.NP / 2.0 + 1.0))
        sum_neg += neg_weights[i]
    
    if sum_neg > 1e-10:
        neg_weights /= sum_neg
    
    rank_mu_neg = np.zeros((self.dim, self.dim))
    for i in range(neg_mu):
        idx = self.NP - 1 - i
        diff = (self.population[idx] - self.old_mean) / self.sigma
        rank_mu_neg += neg_weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rates based on condition number
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    
    # Reduce adaptation if already well-conditioned
    if cond < 1e3:
        ccov_scale = 1.0
    elif cond < 1e5:
        ccov_scale = 0.5
    else:
        ccov_scale = 0.25
    
    ccov_1 = ccov_scale / (self.dim + 2.0)
    ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    ccov_mu_neg = 0.5 * ccov_scale / (self.dim + 2.0) / (self.dim + 4.0)
    
    # Combine all updates
    self.C = ((1.0 - ccov_1 - ccov_mu - ccov_mu_neg) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu_pos +
              ccov_mu_neg * rank_mu_neg)
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
```