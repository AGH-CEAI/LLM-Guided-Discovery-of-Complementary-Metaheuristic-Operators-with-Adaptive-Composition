**Idea: Adaptive Relative Improvement Stagnation**
Monitors relative improvement rate vs. fitness scale, using EMA-smoothed improvement and adaptive threshold to detect when progress becomes negligible relative to problem difficulty.
```python
def _check_stagnation(self):
    """Track stagnation using adaptive relative improvement threshold.
    
    Key insight for worst unsolved tasks (errors ~1e+01 to 1e+02):
    - Absolute threshold 1e-12 is meaningless at these scales
    - Need relative improvement rate that adapts to fitness magnitude
    - EMA smoothing filters noise while detecting genuine stagnation
    """
    if self.f_opt < self.f_opt_prev - 1e-12:
        self.stagnation_counter = 0
    else:
        self.stagnation_counter += 1
    
    # Compute improvement (negative = better)
    improvement = self.f_opt_prev - self.f_opt
    improvement = max(0.0, improvement)
    
    # Adaptive threshold scales with fitness magnitude
    # For high-error tasks (~1e+02), larger absolute improvements needed
    # For near-target tasks (~1e-08), tiny improvements matter
    fitness_scale = max(abs(self.f_opt), 1.0)
    relative_threshold = fitness_scale * 1e-4
    
    # EMA-smoothed improvement for noise robustness
    if not hasattr(self, 'improvement_ema_stagnation'):
        self.improvement_ema_stagnation = improvement
    else:
        alpha = 0.3
        self.improvement_ema_stagnation = (
            alpha * improvement + (1 - alpha) * self.improvement_ema_stagnation
        )
    
    # Secondary: check if population diversity is collapsing
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = pop_variance / (expected_var + 1e-10)
    
    # Count as stagnant if improvement is negligible relative to scale
    # OR if diversity has severely collapsed
    is_stagnant = (
        self.improvement_ema_stagnation < relative_threshold and
        diversity_ratio < 0.05
    )
    
    if is_stagnant:
        self.stagnation_counter += 2  # Accelerate restart detection
    
    self.f_opt_prev = self.f_opt
```