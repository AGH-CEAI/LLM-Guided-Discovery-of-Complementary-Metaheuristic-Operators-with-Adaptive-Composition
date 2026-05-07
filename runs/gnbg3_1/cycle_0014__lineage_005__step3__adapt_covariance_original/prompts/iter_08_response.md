Looking at the error patterns, the worst tasks (16: 83.4, 17: 37.3, 23: 18.9) have errors 100× larger than mid-tier tasks, suggesting severe trapping in local optima with ill-conditioned covariance matrices. Current variants adjust learning rates or add perturbations, but none explicitly reset the covariance structure when stuck.

**Idea: Covariance Restart with Momentum Escaping**
This approach explicitly detects stagnation AND ill-conditioning, then performs a controlled restart that injects fresh exploration while preserving step-size momentum. Fundamentally different from all prior variants which only modify learning rates.

```python
def _adapt_covariance_original(self):
    """Covariance adaptation with stagnation-triggered restart for escaping local optima."""
    stagnation_threshold = max(20, self.dim * 2)
    cond_threshold = 1e7
    eigmin_threshold = 1e-8
    
    need_restart = (self.stagnation_counter > stagnation_threshold)
    
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / max(np.min(eigvals), 1e-15)
    need_restart = need_restart or (cond > cond_threshold) or (np.min(eigvals) < eigmin_threshold)
    
    if need_restart:
        self.C = np.eye(self.dim) * (self.sigma ** 2)
        self.pc *= 0.0
        self.stagnation_counter = 0
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    if not hasattr(self, 'momentum_ema'):
        self.momentum_ema = np.zeros(self.dim)
    self.momentum_ema = 0.7 * self.momentum_ema + 0.3 * y_mean
    y_momentum = self.momentum_ema / max(np.linalg.norm(self.momentum_ema), 1e-10) * np.linalg.norm(y_mean)
    
    cc_adapt = self.cc * (1.5 if need_restart else 1.0)
    cc_adapt = np.clip(cc_adapt, 0.001, 0.5)
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_momentum
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    ccov_boost = 3.0 if need_restart else 1.0
    ccov_adapt = min(self.ccov * ccov_boost, 0.4)
    
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * rank_one + 
              (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
    
    if need_restart:
        self.C += 0.1 * np.eye(self.dim) * (self.sigma ** 2)
    
    self.C = self._ensure_positive_definite(self.C)
```