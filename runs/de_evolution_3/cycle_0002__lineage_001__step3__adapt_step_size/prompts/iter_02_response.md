**Idea: Momentum-Enhanced Sigma Adaptation with Stagnation Detection**

Replace the standard CSA with a momentum-based step size controller that accumulates a velocity term on sigma, uses fitness improvement history to detect stagnation, and applies corrective pulses when progress stalls. This addresses the root failure mode: the worst tasks (errors 1-4.6) are NOT converging at all, likely due to sigma oscillations or premature settling. The momentum term smooths adaptation and prevents erratic sigma jumps, while the stagnation trigger provides targeted exploration boosts.

```python
def _adapt_step_size(self):
    # Momentum velocity for sigma (accumulates over generations)
    if not hasattr(self, '_sigma_velocity'):
        self._sigma_velocity = 0.0
    if not hasattr(self, '_prev_sigma'):
        self._prev_sigma = self.sigma
    
    # Fitness-based stagnation detection using improvement history
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    if not hasattr(self, '_prev_best_f'):
        self._prev_best_f = self.best_f
    
    current_best = self.best_f
    if np.isfinite(current_best) and np.isfinite(self._prev_best_f):
        improvement = self._prev_best_f - current_best
        self._fitness_history.append(improvement)
        # Keep history bounded
        if len(self._fitness_history) > 20:
            self._fitness_history.pop(0)
    
    self._prev_best_f = current_best
    
    # Compute stagnation: no improvement over recent generations
    is_stagnant = False
    if len(self._fitness_history) >= 10:
        recent_improvements = self._fitness_history[-10:]
        positive_count = sum(1 for x in recent_improvements if x > 1e-14 * max(1.0, abs(self._prev_best_f)))
        if positive_count <= 2:
            is_stagnant = True
    
    # Standard CMA-ES cumulative path ratio
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    target_norm = self.chi_n
    ratio = p_sigma_norm / (target_norm + 1e-20)
    
    # Base update: log-smoothed step size change
    base_update = (self.c_sigma / self.d_sigma) * (ratio - 1.0)
    
    # Momentum term: blend current update with previous velocity
    momentum = 0.7
    self._sigma_velocity = momentum * self._sigma_velocity + (1.0 - momentum) * base_update
    
    # Apply momentum-enhanced update
    log_sigma_change = self._sigma_velocity
    self.sigma *= np.exp(log_sigma_change)
    
    # Stagnation recovery: if stuck, apply exploration pulse
    if is_stagnant:
        # Increase sigma by factor 2-5x to escape local basin
        pulse = np.random.uniform(2.0, 5.0)
        self.sigma *= pulse
        # Reset momentum to allow fresh direction
        self._sigma_velocity *= 0.1
        # Clear fitness history to reset detection
        self._fitness_history = []
    
    # Fine-tuning: if sigma is very small and we're making progress, slow decay
    if self.sigma < 1e-3 and len(self._fitness_history) >= 5:
        recent_improvements = self._fitness_history[-5:]
        if all(x > 0 for x in recent_improvements):
            # Gentle reduction to maintain progress
            self.sigma *= 0.95
    
    # Bound sigma to prevent numerical issues
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```