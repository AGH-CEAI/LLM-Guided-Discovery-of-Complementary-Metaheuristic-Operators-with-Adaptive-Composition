**Idea: Active CMA-ES with Negative Weights**
Uses both positive and negative ranked weights to actively shrink covariance in directions that produce poor solutions, a well-established improvement over basic CMA-ES.

```python
def _adapt_covariance_variant_09(self):
    """Active CMA-ES: Negative weights shrink bad directions."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Active CMA-ES: Use both positive and negative weights
    n_active = self.mu
    active_weights = np.zeros(self.NP)
    for i in range(self.NP):
        if i < self.mu:
            active_weights[i] = np.log(self.mu + 0.5) - np.log(i + 1)
        elif i < n_active:
            active_weights[i] = -np.log(n_active + 0.5) + np.log(i + 1)
    
    sum_pos = np.sum(active_weights[active_weights > 0])
    sum_neg = np.sum(active_weights[active_weights < 0])
    
    if sum_pos > 0:
        active_weights[active_weights > 0] /= sum_pos
    if sum_neg < 0:
        active_weights[active_weights < 0] /= -sum_neg
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(n_active):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += active_weights[i] * np.outer(diff, diff)
    
    rank_one = np.outer(self.pc, self.pc)
    
    # Adaptive learning rate based on condition number
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    ccov_base = min(self.ccov, 0.3 / (1.0 + np.log1p(cond)))
    
    # Combine rank-1 and rank-μ updates
    self.C = ((1.0 - ccov_base) * self.C + 
              ccov_base * rank_one + 
              ccov_base * (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu)
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
```