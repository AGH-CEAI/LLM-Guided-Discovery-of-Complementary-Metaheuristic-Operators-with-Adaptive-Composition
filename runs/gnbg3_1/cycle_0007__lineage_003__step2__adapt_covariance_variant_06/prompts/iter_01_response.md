**Idea: Active CMA-ES with Negative Weights**
A well-established strategy that actively promotes diversity by incorporating negative weights in the rank-μ update, helping escape local optima and maintain exploration.

```python
def _adapt_covariance_variant_06(self):
    """Active CMA-ES with negative weights for diversity-promoting rank-μ update."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Standard evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Active CMA-ES: compute both positive and negative weights
    n_pos = self.mu // 2
    n_neg = self.mu - n_pos
    
    # Sort population by fitness (ascending - best first)
    sorted_indices = np.argsort(self.fitness)
    sorted_pop = self.population[sorted_indices]
    
    # Positive weights for top mu individuals (standard)
    pos_weights = np.log(n_pos + 0.5) - np.log(np.arange(1, n_pos + 1))
    pos_weights /= np.sum(pos_weights)
    pos_weights = np.maximum(pos_weights, 0.0)
    
    # Negative weights for bottom mu individuals (inverted order)
    neg_weights = np.log(n_neg + 0.5) - np.log(np.arange(1, n_neg + 1))
    neg_weights /= np.sum(neg_weights)
    neg_weights = np.maximum(neg_weights, 0.0)
    
    # Build rank-mu update with both positive and negative contributions
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(n_pos):
        diff = (sorted_pop[i] - self.old_mean) / self.sigma
        rank_mu += pos_weights[i] * np.outer(diff, diff)
    
    for i in range(n_neg):
        idx = self.mu - 1 - i
        diff = (sorted_pop[idx] - self.old_mean) / self.sigma
        rank_mu -= 0.25 * neg_weights[i] * np.outer(diff, diff)
    
    # Combine updates with active covariance learning rate
    c_c = 2.0 / ((self.dim + 1.41) ** 2)
    c_1 = 1.0 / (self.dim + 2.0)
    c_mu = self.mueff / (self.dim + 2.0)
    
    alpha_c = 1.0 + c_1 + c_mu
    c_1_scaled = c_1 / alpha_c
    c_mu_scaled = c_mu / alpha_c
    c_c_scaled = c_c / alpha_c
    
    self.C = ((1.0 - c_c_scaled) * self.C + 
              c_c_scaled * rank_one +
              c_mu_scaled * rank_mu)
    
    # Ensure numerical stability
    self.C = self._ensure_positive_definite(self.C)
```