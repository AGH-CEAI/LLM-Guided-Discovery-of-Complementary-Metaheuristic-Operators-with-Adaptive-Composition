**Idea: EMA-Based Stagnation Detection**
Uses exponential moving average of fitness improvement rates with adaptive thresholds for more robust stagnation detection, avoiding false triggers from noisy plateaus.
```python
def _check_stagnation(self):
    """Track stagnation using EMA-smoothed improvement rate with adaptive thresholds."""
    improvement = self.f_opt_prev - self.f_opt
    
    # Initialize EMA tracking if needed
    if not hasattr(self, 'improvement_ema'):
        self.improvement_ema = 0.0
        self.ema_alpha = 0.3
    
    # Update EMA with improvement (positive = getting better)
    self.improvement_ema = self.ema_alpha * improvement + (1.0 - self.ema_alpha) * self.improvement_ema
    
    # Adaptive threshold based on problem scale and current fitness magnitude
    fit_scale = max(abs(self.f_opt), 1.0)
    threshold = max(1e-10, 1e-8 * fit_scale)
    
    # Also track population-level progress via variance
    if len(self.fitness) >= 2:
        fit_variance = float(np.var(self.fitness))
        var_scale = max(fit_variance, 1e-10)
        # Normalized improvement relative to population variance
        relative_improvement = abs(improvement) / var_scale
    else:
        relative_improvement = 0.0
    
    # Stagnation: EMA near zero AND no significant relative improvement
    is_stagnant = abs(self.improvement_ema) < threshold and relative_improvement < 1e-6
    
    if is_stagnant:
        self.stagnation_counter += 1
    else:
        self.stagnation_counter = 0
    
    # Adaptive EMA smoothing: faster when improving, slower when stagnating
    if self.improvement_ema > threshold:
        self.ema_alpha = min(0.5, self.ema_alpha * 1.1)
    else:
        self.ema_alpha = max(0.1, self.ema_alpha * 0.95)
    
    self.f_opt_prev = self.f_opt
```