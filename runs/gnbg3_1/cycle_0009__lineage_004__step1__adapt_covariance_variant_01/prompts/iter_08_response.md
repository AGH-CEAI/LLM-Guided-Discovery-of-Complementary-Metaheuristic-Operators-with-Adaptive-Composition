**Idea: Improvement-Momentum Driven Covariance**
Uses exponential smoothing of fitness improvement rate to dynamically modulate learning rates and inject diversity when stuck, fundamentally different from condition-number-only approaches.

```python
def _adapt_covariance_variant_01(self):
    """Improvement-momentum driven covariance adaptation with stagnation escape."""
    y_mean = (self.mean - self.old_mean) / self.sigma

    # Track improvement momentum using exponential moving average
    if not hasattr(self, 'improvement_ema'):
        self.improvement_ema = 1.0
        self.improvement_count = 0

    # Compute relative improvement rate
    fit_improvement = max(0.0, self.f_opt_prev - self.f_opt)
    rel_improvement = fit_improvement / max(abs(self.f_opt), 1.0)
    
    # Update EMA with momentum
    alpha = 0.3
    self.improvement_ema = (1.0 - alpha) * self.improvement_ema + alpha * max(rel_improvement, 1e-15)
    self.improvement_count += 1

    # Compute condition number for numerical stability
    eigvals = np.linalg.eigvalsh(self.C)
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
    cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

    # Adaptive learning rate based on condition number (for numerical stability)
    if cond > 1e6:
        ccov_scale = 0.05
        cc_scale = 0.05
    elif cond > 1e4:
        ccov_scale = 0.2
        cc_scale = 0.2
    elif cond > 1e2:
        ccov_scale = 0.4
        cc_scale = 0.4
    else:
        ccov_scale = 1.0
        cc_scale = 1.0

    # Scale by improvement momentum (key difference from other variants)
    # Low improvement = reduce learning rate, high improvement = maintain/increase
    improvement_factor = np.log1p(self.improvement_ema * 1e6) / 10.0
    improvement_factor = np.clip(improvement_factor, 0.05, 2.0)
    
    ccov_scale *= improvement_factor
    cc_scale *= improvement_factor

    # Inject diversity on stagnation (different from condition-number restarts)
    stagnation_threshold = 1e-8
    if self.improvement_ema < stagnation_threshold and self.improvement_count > 20:
        # Add exploration noise proportional to stagnation severity
        noise_scale = np.log(1.0 / max(self.improvement_ema, 1e-15)) / 50.0
        noise_scale = np.clip(noise_scale, 0.01, 0.5)
        
        # Inject noise into covariance
        exploration_noise = np.eye(self.dim) * np.mean(eigvals) * noise_scale
        self.C = self.C + exploration_noise
        
        # Reset momentum to allow recovery
        self.improvement_ema = max(self.improvement_ema, 1e-6)
        self.improvement_count = 0

    # Trigger covariance restart on extreme condition number
    if cond > 1e8 or eig_spread < 1e-8:
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.L = None

    # Compute adaptive learning rates
    ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
    ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

    # Evolution path update
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)

    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)

    # Combine updates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
              ccov_1 * rank_one + 
              ccov_mu * rank_mu)

    self.C = self._ensure_positive_definite(self.C)
```