**Idea: Restart-Triggered Aggressive Covariance Injection**

The worst unsolved tasks (errors 1e+01 to 1e+02) indicate the optimizer is catastrophically stuck in local optima or the covariance has collapsed. The current variant_08 uses fixed small learning rates across multiple evolution paths but never fundamentally restructures the covariance when trapped. This variant detects stagnation and condition collapse, then aggressively injects fresh random direction + dramatically boosts learning rates to break out.

```python
def _adapt_covariance_variant_08(self):
    """Restart-triggered aggressive covariance injection for escaping catastrophic stagnation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Check stagnation severity and covariance health
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / np.maximum(np.min(eigvals), 1e-15)
    
    stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
    
    # Determine boost level based on how stuck we are
    if stagnation_ratio > 0.7 or cond > 1e5:
        # Catastrophic stagnation: inject fresh random direction + maximum boost
        inject_dir = np.random.randn(self.dim)
        inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
        random_rank_one = np.outer(inject_dir, inject_dir)
        inject_strength = 0.5
        ccov_boost = 10.0
        cc_boost = 5.0
    elif stagnation_ratio > 0.4 or cond > 1e4:
        # Moderate stagnation: partial injection + strong boost
        inject_dir = np.random.randn(self.dim)
        inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
        random_rank_one = np.outer(inject_dir, inject_dir)
        inject_strength = 0.2
        ccov_boost = 4.0
        cc_boost = 2.5
    elif stagnation_ratio > 0.2:
        # Mild stagnation: small injection + moderate boost
        inject_dir = np.random.randn(self.dim)
        inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
        random_rank_one = np.outer(inject_dir, inject_dir)
        inject_strength = 0.1
        ccov_boost = 2.0
        cc_boost = 1.5
    else:
        # Normal mode: no injection
        random_rank_one = np.zeros((self.dim, self.dim))
        inject_strength = 0.0
        ccov_boost = 1.0
        cc_boost = 1.0
    
    # Apply boosted learning rates
    cc_adapted = min(self.cc * cc_boost, 0.5)
    self.pc = (1.0 - cc_adapted) * self.pc + np.sqrt(cc_adapted * (2.0 - cc_adapted)) * y_mean
    
    # Rank-one update (primary evolution path)
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Apply boosted covariance update with injection
    ccov_adapted = min(self.ccov * ccov_boost, 0.6)
    self.C = ((1.0 - ccov_adapted) * self.C +
              ccov_adapted * rank_one +
              inject_strength * random_rank_one +
              (1.0 - 1.0 / self.mueff) * ccov_adapted * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Full reset on extreme ill-conditioning
    eigvals_new = np.linalg.eigvalsh(self.C)
    cond_new = np.max(eigvals_new) / np.maximum(np.min(eigvals_new), 1e-15)
    if cond_new > 1e8:
        self.C = np.eye(self.dim) * np.mean(eigvals_new)
        self.pc = np.zeros(self.dim)
        self.L = None
```