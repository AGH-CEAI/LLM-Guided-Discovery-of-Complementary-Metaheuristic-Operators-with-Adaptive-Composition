**Idea: Active CMA-ES with Negative Update**
Active CMA-ES reduces covariance learning along unfavorable eigendirections by separating positive and negative weighted rank-μ updates, preventing premature convergence and maintaining diversity.

```python
def _adapt_covariance_variant_08(self):
    """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Evolution path update (unchanged)
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update with positive/negative decomposition
    rank_mu_pos = np.zeros((self.dim, self.dim))
    rank_mu_neg = np.zeros((self.dim, self.dim))
    
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        weight = self.weights[i]
        outer = np.outer(diff, diff)
        
        if weight > 0:
            rank_mu_pos += weight * outer
        else:
            rank_mu_neg += abs(weight) * outer
    
    # Normalize by sum of positive and negative weights separately
    pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
    neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)
    
    rank_mu_pos /= pos_sum
    rank_mu_neg /= neg_sum
    
    # Combine positive and negative updates (active CMA-ES core)
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * rank_one +
              (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
              self.ccov * 0.4 * rank_mu_neg)
    
    self.C = self._ensure_positive_definite(self.C)
```