**Idea: EMA Momentum-Aware Parameter Drift Detection**

Use dual EMA smoothing of success rates to detect momentum and regime changes, combined with parameter drift tracking and stagnation-aware adaptation.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using temporal EMA dynamics and parameter drift detection."""
    success_rate = np.mean(improved_mask)
    
    # Initialize EMA state on first call
    if not hasattr(self, 'success_ewma_short'):
        self.success_ewma_short = success_rate
        self.success_ewma_long = success_rate
        self.stagnation_counter = 0
        self.F_history = [self.F]
        self.Cr_history = [self.Cr]
    
    # Update EMAs with different smoothing constants
    alpha_short = 0.3   # Fast EMA for short-term dynamics
    alpha_long = 0.1    # Slow EMA for trend detection
    self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
    self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long
    
    # Track parameter history for drift detection
    self.F_history.append(F_used)
    self.Cr_history.append(F_used)
    if len(self.F_history) > 15:
        self.F_history.pop(0)
        self.Cr_history.pop(0)
    
    # Compute parameter drift from historical mean
    F_mean = np.mean(self.F_history)
    Cr_mean = np.mean(self.Cr_history)
    F_drift = self.F / (F_mean + 1e-10)
    Cr_drift = self.Cr / (Cr_mean + 1e-10)
    
    # Regime detection via EMA divergence
    short_trending_up = self.success_ewma_short > self.success_ewma_long
    short_trending_down = self.success_ewma_short < self.success_ewma_long
    both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
    
    # Compute momentum-adjusted base adjustments
    momentum_factor = np.clip(F_drift, 0.7, 1.4)
    
    if short_trending_up and self.success_ewma_short > 0.25:
        # Exploitation phase: success accelerating, increase convergence pressure
        F_adjustment = 1.05 * momentum_factor
        Cr_adjustment = 1.08
        self.stagnation_counter = max(0, self.stagnation_counter - 1)
    elif short_trending_down and self.success_ewma_short < 0.2:
        # Exploration phase: success declining, inject diversity
        F_adjustment = 0.88 / momentum_factor
        Cr_adjustment = 0.92
        self.stagnation_counter += 1
    elif both_stagnant:
        # Deep stagnation: large perturbation to escape local traps
        F_adjustment = 0.75
        Cr_adjustment = 1.15
        self.stagnation_counter += 2
    else:
        # Neutral regime: gentle correction toward baseline
        delta = self.success_ewma_short - self.success_ewma_long
        F_adjustment = 1.0 + 0.08 * delta
        Cr_adjustment = 1.0 - 0.05 * delta
    
    # Apply adjustments with drift correction
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```