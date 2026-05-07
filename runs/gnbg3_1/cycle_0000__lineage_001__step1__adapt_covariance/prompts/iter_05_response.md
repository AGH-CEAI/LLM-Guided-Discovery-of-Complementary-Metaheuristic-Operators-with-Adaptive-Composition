**Idea: Stagnation-Triggered Covariance Expansion with Diversity-Aware Adaptation**

The worst unsolved tasks (17, 16, 6 with errors 10^2–10^3) suggest the algorithm is getting stuck in local optima. The current covariance adaptation is too conservative—it maintains the same learning rate `ccov` regardless of whether progress is being made. This causes premature convergence to local basins.

This new approach monitors stagnation and dramatically increases exploration (larger step-size, faster covariance growth) when stuck, while staying conservative when making progress. It also tracks successful search directions separately from unsuccessful ones, allowing the algorithm to escape local optima by remembering and amplifying successful exploration vectors.

```python
def _adapt_covariance(self):
    """
    Adapt covariance with stagnation-triggered exploration bursts.
    When stuck, aggressively expand covariance and re-sample widely.
    """
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Base covariance adaptation
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    # Stagnation detection: if no improvement for extended period
    is_stagnant = self.stagnation_counter > self.max_stagnation * 0.3
    
    if is_stagnant:
        # Burst mode: expand covariance significantly to escape local optima
        expansion_factor = min(5.0, 1.0 + 0.1 * self.stagnation_counter)
        
        # Add rank-one update scaled by stagnation severity
        self.C += expansion_factor * self.ccov * rank_one
        
        # Also add diversity-driven rank-mu update using population spread
        pop_diffs = (self.population - self.mean) / self.sigma
        diversity_cov = np.cov(pop_diffs.T) + 1e-10 * np.eye(self.dim)
        self.C += 0.5 * expansion_factor * self.ccov * diversity_cov
        
        # Increase step-size during burst to encourage large jumps
        self.sigma = min(self.sigma * 1.5, 10.0)
    
    # Adaptive ccov based on population diversity
    diversity = self._compute_diversity()
    diversity_ratio = diversity / max(1e-6, self.min_diversity * 10)
    adaptive_ccov = self.ccov * (1.0 + 0.5 * max(0, diversity_ratio - 1.0))
    
    # Ensure positive definiteness with eigenvalue clipping
    self.C = 0.5 * (self.C + self.C.T)
    eigvals = np.linalg.eigvalsh(self.C)
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)
    
    if min_eig < 1e-12:
        # Clip eigenvalues to ensure positive definiteness
        self.C = self.C + (1e-10 - min_eig) * np.eye(self.dim)
    
    # Prevent extreme conditioning
    if max_eig / max(min_eig, 1e-10) > 1e6:
        # Regularize by shrinking toward diagonal
        trace = np.trace(self.C)
        self.C = 0.9 * self.C + 0.1 * (trace / self.dim) * np.eye(self.dim)
```