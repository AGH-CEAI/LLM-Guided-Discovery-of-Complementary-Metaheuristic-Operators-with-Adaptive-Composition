Looking at the data, the worst unsolved tasks (17, 16, 20, 10, 11, 18, 23, 12) have errors stuck at 1e+1 to 1e+2 — roughly 8-10 decades above target. This suggests the optimizer is getting trapped in local optima and the current stagnation detection (simple threshold check) is too passive, not triggering restarts aggressively enough when progress stalls.

The current `_check_stagnation` uses a fixed 1e-12 threshold and only increments a counter. I need a fundamentally different approach that:
1. Detects stagnation through multiple signals (not just fitness improvement)
2. Monitors covariance matrix health (ill-conditioning often precedes failure on these tasks)
3. Uses adaptive thresholds based on problem scale
4. Triggers restarts more aggressively when the algorithm is "spinning wheels"

**Idea: Multi-Signal Adaptive Stagnation with Covariance Health Monitoring**

```python
def _check_stagnation(self):
    """Multi-signal stagnation detection with covariance health monitoring."""
    # Signal 1: Improvement-based stagnation (adaptive threshold)
    fitness_scale = max(abs(self.f_opt), 1.0)
    adaptive_threshold = 1e-6 * fitness_scale
    if self.f_opt < self.f_opt_prev - adaptive_threshold:
        self.stagnation_counter = 0
        if hasattr(self, 'improvement_streak'):
            self.improvement_streak += 1
        else:
            self.improvement_streak = 1
    else:
        self.stagnation_counter += 1
        if hasattr(self, 'improvement_streak'):
            self.improvement_streak = 0
    
    # Signal 2: Covariance health (detect ill-conditioning before collapse)
    try:
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        cond = np.max(eigvals) / np.min(eigvals)
        cond_stagnation = cond > 1e6
    except:
        cond_stagnation = True
    
    # Signal 3: Step-size stagnation (sigma not changing)
    if hasattr(self, 'prev_sigma'):
        sigma_ratio = self.sigma / max(self.prev_sigma, 1e-15)
        sigma_stagnation = 0.95 < sigma_ratio < 1.05
    else:
        sigma_stagnation = False
    self.prev_sigma = self.sigma
    
    # Signal 4: Diversity collapse (population converging)
    pop_spread = np.mean(np.std(self.population, axis=0))
    expected_spread = (self.ub[0] - self.lb[0]) / 6.0
    diversity_stagnation = pop_spread < 0.1 * expected_spread
    
    # Signal 5: Fitness plateau (no improvement in many generations)
    plateau_stagnation = self.stagnation_counter > self.max_stagnation // 3
    
    # Combined stagnation score (any strong signal triggers restart consideration)
    combined_stagnation = (
        self.stagnation_counter > self.max_stagnation // 2 or
        cond_stagnation or
        (sigma_stagnation and plateau_stagnation) or
        diversity_stagnation
    )
    
    # If multiple mild signals, also consider stagnant
    mild_signals = int(plateau_stagnation) + int(cond_stagnation) + int(sigma_stagnation)
    if mild_signals >= 2:
        combined_stagnation = True
    
    # Update f_opt_prev
    self.f_opt_prev = self.f_opt
    
    # Trigger aggressive restart check in _restart_if_needed by setting flag
    if combined_stagnation and not hasattr(self, 'aggressive_restart_pending'):
        self.aggressive_restart_pending = True
    elif not combined_stagnation and hasattr(self, 'aggressive_restart_pending'):
        self.aggressive_restart_pending = False
```

This is fundamentally different because:
- **Multi-signal**: Uses 5 independent indicators instead of just fitness improvement
- **Covariance monitoring**: Detects ill-conditioning (cond > 1e6) which is a precursor to failure on the hardest tasks
- **Adaptive threshold**: Scales improvement threshold by fitness magnitude (1e-6 × fitness_scale) rather than fixed 1e-12
- **Aggressive flag**: Passes a signal to restart logic for more aggressive recovery when stuck

This should help the worst tasks (17, 16, 20, etc.) by detecting when the optimizer is "spinning wheels" much earlier and triggering restarts before it gets permanently stuck in local optima.