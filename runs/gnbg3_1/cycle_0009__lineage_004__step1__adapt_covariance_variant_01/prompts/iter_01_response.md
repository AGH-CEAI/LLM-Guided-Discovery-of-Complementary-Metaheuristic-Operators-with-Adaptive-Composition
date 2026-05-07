**Idea: Momentum-Enhanced Covariance Adaptation**
Uses exponential moving average of mean shift direction to stabilize covariance updates and improve convergence on ill-conditioned problems.
```python
def _adapt_covariance_variant_01(self):
    """Momentum-enhanced covariance adaptation with adaptive learning rates."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Momentum term: EMA of mean shift direction for stable adaptation
    beta = 0.5
    if not hasattr(self, 'y_mean_ema'):
        self.y_mean_ema = np.zeros(self.dim)
    self.y_mean_ema = beta * self.y_mean_ema + (1.0 - beta) * y_mean
    
    # Adaptive learning rate based on eigenvalue spread
    eigvals = np.linalg.eigvalsh(self.C)
    eig_ratio = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    cond_factor = min(1.0, 1.0 / np.log1p(eig_ratio + 1.0))
    
    # Momentum-enhanced evolution path
    cc_momentum = self.cc * (0.5 + 0.5 * cond_factor)
    cc_momentum = np.clip(cc_momentum, 0.01, 0.2)
    
    self.pc = (1.0 - cc_momentum) * self.pc + np.sqrt(cc_momentum * (2.0 - cc_momentum)) * self.y_mean_ema
    
    # Rank-one update with momentum
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adaptive covariance learning rate
    ccov_adaptive = self.ccov * (0.5 + 0.5 * cond_factor)
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)
    
    # Combine with momentum-weighted original covariance
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```