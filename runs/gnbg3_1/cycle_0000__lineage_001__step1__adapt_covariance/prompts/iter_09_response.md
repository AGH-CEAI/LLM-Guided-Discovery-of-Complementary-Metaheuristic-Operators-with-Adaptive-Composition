**Idea: Adaptive Diversity-Sensitive Covariance Adaptation**

Instead of fixed learning rates, this approach monitors population diversity and stagnation signals to dynamically scale the covariance update. When diversity is low or progress stalls (common in the multimodal Tasks 17, 16, 6), it increases `ccov` and amplifies rank-mu contributions to escape local optima. This fundamentally differs from fixed-rate updates by providing exploration-on-demand.

```python
def _adapt_covariance(self):
    """
    Adapt covariance matrix with diversity-sensitive and stagnation-aware scaling.
    """
    # Track population diversity (scaled variance)
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)
    
    # Track improvement for stagnation detection
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.improvement_ema = 1.0
    improvement = max(1e-10, self.prev_f_opt - self.f_opt)
    self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
    self.prev_f_opt = self.f_opt
    
    stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
    
    # Adaptive ccov: boost when diversity is low or stagnated
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    ccov_adaptive = base_ccov * (1.5 if stagnation else 1.0) * (1.5 if diversity < 0.3 else 1.0)
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
    
    # Compute evolution path
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update with diversity-sensitive scaling
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Scale rank-mu: boost when diversity is low (encourage exploration)
    rank_mu_scale = 2.0 if (stagnation or diversity < 0.3) else 1.0
    
    # Combined update with adaptive rates
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              ccov_adaptive * rank_mu_scale * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
    # Ensure positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-10:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
```