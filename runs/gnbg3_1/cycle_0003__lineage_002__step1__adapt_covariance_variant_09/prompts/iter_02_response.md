**Idea: Exponential Moving Average Covariance with Natural Gradient Damping**

This approach uses exponential moving averages of rank-one and rank-mu updates to smooth covariance evolution, combined with natural gradient-inspired damping based on the inverse covariance matrix (FIM approximation). The key insight is that the worst tasks (errors ~10^2-10^3) likely suffer from erratic covariance updates causing exploration instability—this EMA smoothing will damp oscillations while natural gradient coupling improves conditioning on ill-shaped landscapes.

```python
def _adapt_covariance_variant_09(self):
    """Exponential moving average covariance with natural gradient damping."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute natural gradient direction (C^-1 @ y_mean)
    try:
        y_cov_norm = np.linalg.solve(self.C, y_mean)
    except np.linalg.LinAlgError:
        y_cov_norm = np.linalg.solve(self.C + 1e-6 * np.eye(self.dim), y_mean)
    
    nat_grad_norm = np.linalg.norm(y_cov_norm)
    damping_factor = min(1.0 + nat_grad_norm * 0.1, 2.0)
    
    cc_damped = self.cc / damping_factor
    self.pc = (1.0 - cc_damped) * self.pc + np.sqrt(cc_damped * (2.0 - cc_damped)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Initialize EMA tracking for smooth covariance evolution
    if not hasattr(self, 'ema_rank_one'):
        self.ema_rank_one = np.zeros((self.dim, self.dim))
        self.ema_rank_mu = np.zeros((self.dim, self.dim))
        self.ema_beta = 0.95
    
    self.ema_rank_one = self.ema_beta * self.ema_rank_one + (1.0 - self.ema_beta) * rank_one
    self.ema_rank_mu = self.ema_beta * self.ema_rank_mu + (1.0 - self.ema_beta) * rank_mu
    
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)
    
    # Stagnation detection for adaptive scaling
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.improvement_ema = 1.0
    
    improvement = max(1e-10, self.prev_f_opt - self.f_opt)
    self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
    self.prev_f_opt = self.f_opt
    
    stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
    
    # Adaptive learning rate based on state
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    
    ccov_scale = 1.0
    if stagnation:
        ccov_scale *= 2.0
    if diversity < 0.3:
        ccov_scale *= 1.5
    
    ccov_adaptive = np.clip(base_ccov * ccov_scale, 1e-10, 0.5)
    
    # Blend current and EMA updates for stability
    ema_weight = 0.3 if stagnation else 0.5
    rank_one_blend = ema_weight * self.ema_rank_one + (1.0 - ema_weight) * rank_one
    rank_mu_blend = ema_weight * self.ema_rank_mu + (1.0 - ema_weight) * rank_mu
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one_blend + 
              ccov_adaptive * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu_blend)
    
    # Eigenvalue bounds for conditioning control
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min, eig_max = np.min(eigvals), np.max(eigvals)
    cond = eig_max / (eig_min + 1e-10)
    
    if cond > 1e6:
        shrink = np.sqrt(1e6 / cond)
        self.C *= shrink
    
    self.C = self._ensure_positive_definite(self.C)
```