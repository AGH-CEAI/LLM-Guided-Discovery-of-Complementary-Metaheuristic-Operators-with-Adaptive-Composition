**Idea: Mutation-Guided Covariance Adaptation**
A fundamentally different approach that uses successful differential evolution-style mutation vectors to directly inform the covariance matrix, complementing the CMA-ES rank-μ update with problem-specific directional learning from recent improvements.

```python
def _adapt_covariance(self):
    """
    Adapt covariance using hybrid CMA-ES + mutation-guided update.
    Tracks successful mutation directions (mean - old_mean) weighted by 
    improvement magnitude and blends their covariance into the update.
    """
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Mutation-guided covariance: track successful directions
    # Use mean shift as proxy for mutation direction, weighted by improvement
    improvement = max(self.f_opt_prev - self.f_opt, 0.0)
    if improvement > 0:
        self.mutation_hist.append((y_mean.copy(), improvement))
    
    # Keep only recent history
    if len(self.mutation_hist) > 10:
        self.mutation_hist = self.mutation_hist[-10:]
    
    # Compute weighted covariance from successful mutations
    mut_cov = np.zeros((self.dim, self.dim))
    if len(self.mutation_hist) >= 2:
        total_weight = sum(w for _, w in self.mutation_hist)
        if total_weight > 1e-30:
            for vec, w in self.mutation_hist:
                mut_cov += (w / total_weight) * np.outer(vec, vec)
            # Normalize to appropriate scale
            mut_cov = mut_cov * self.dim
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu) +
              0.3 * mut_cov)
    
    # Ensure positive definiteness
    self.C = 0.5 * (self.C + self.C.T)
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-10:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
```