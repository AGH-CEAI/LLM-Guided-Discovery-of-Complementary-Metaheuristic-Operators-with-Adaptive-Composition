**Idea: Autocorrelation-Stabilized Temporal Control**

Category F: Temporal / dynamical — uses autocorrelation of success rate to detect oscillatory vs steady convergence regimes, and tracks momentum in parameter space to prevent runaway drift. This gives fundamentally different regime detection than simple EMA divergence.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using autocorrelation-based regime detection and parameter momentum."""
    success_rate = np.mean(improved_mask)

    # Initialize temporal state on first call
    if not hasattr(self, 'success_history'):
        self.success_history = []
        self.success_ewma_short = success_rate
        self.success_ewma_long = success_rate
        self.F_momentum = 0.0
        self.Cr_momentum = 0.0
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.best_fitness_streak = 0
        self.F_history = [self.F]
        self.Cr_history = [self.Cr]

    # Update EMAs with adaptive smoothing
    alpha_short = 0.4
    alpha_long = 0.15
    self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
    self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

    # Track rolling success history for autocorrelation
    self.success_history.append(success_rate)
    if len(self.success_history) > 12:
        self.success_history.pop(0)

    # Compute autocorrelation at lag-1: detects oscillatory vs steady behavior
    if len(self.success_history) >= 4:
        sh = np.array(self.success_history)
        n = len(sh)
        mean_s = np.mean(sh)
        var_s = np.var(sh) + 1e-10
        autocorr_lag1 = np.mean((sh[:-1] - mean_s) * (sh[1:] - mean_s)) / var_s
    else:
        autocorr_lag1 = 0.5  # Default to no oscillation

    # Compute success rate derivative (acceleration/deceleration)
    success_derivative = success_rate - self.success_ewma_long

    # Track parameter history for momentum and drift
    self.F_history.append(F_used)
    self.Cr_history.append(Cr_used)
    if len(self.F_history) > 15:
        self.F_history.pop(0)
        self.Cr_history.pop(0)

    # Compute parameter momentum (rate of change)
    if len(self.F_history) >= 3:
        F_velocity = (self.F_history[-1] - self.F_history[-3]) / 2.0
        Cr_velocity = (self.Cr_history[-1] - self.Cr_history[-3]) / 2.0
        self.F_momentum = 0.7 * self.F_momentum + 0.3 * F_velocity
        self.Cr_momentum = 0.7 * self.Cr_momentum + 0.3 * Cr_velocity

    # Detect stagnation via best fitness not improving
    current_best = np.min(self.population.fitness) if hasattr(self, 'population') else np.inf
    if current_best < self.prev_best_fitness - 1e-10:
        self.best_fitness_streak += 1
    else:
        self.best_fitness_streak = 0
    self.prev_best_fitness = current_best

    # Regime detection via autocorrelation and derivative
    oscillating = autocorr_lag1 < -0.15  # Alternating success/failure
    steady_converging = autocorr_lag1 > 0.5 and success_rate > 0.2
    accelerating = success_derivative > 0.05
    decelerating = success_derivative < -0.05
    stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
    improving_streak = self.best_fitness_streak > 10

    # Compute momentum factor to damp runaway parameters
    F_hist_std = np.std(self.F_history) + 1e-10
    momentum_damp = np.clip(1.0 - 0.3 * abs(self.F_momentum) / F_hist_std, 0.6, 1.0)

    if stagnant or improving_streak and success_rate < 0.15:
        # Deep stagnation: reset with perturbation, damp momentum
        F_adjustment = 0.7 * momentum_damp
        Cr_adjustment = 1.2
        self.F_momentum *= 0.5
        self.Cr_momentum *= 0.5
        self.stagnation_counter += 2
    elif oscillating:
        # Oscillatory regime: counteract momentum, increase stability
        F_adjustment = 0.85 * momentum_damp
        Cr_adjustment = 0.9 - 0.2 * autocorr_lag1
        self.stagnation_counter += 1
    elif accelerating and steady_converging:
        # Exploitation: success accelerating steadily, increase convergence pressure
        F_adjustment = 1.08 * momentum_damp
        Cr_adjustment = 1.05
        self.stagnation_counter = max(0, self.stagnation_counter - 1)
    elif decelerating and not oscillating:
        # Deceleration without oscillation: moderate diversity injection
        F_adjustment = 0.92
        Cr_adjustment = 0.95
        self.stagnation_counter += 1
    else:
        # Neutral regime: gentle correction toward baseline
        delta = self.success_ewma_short - self.success_ewma_long
        F_adjustment = 1.0 + 0.1 * delta
        Cr_adjustment = 1.0 - 0.05 * delta

    # Apply adjustments with momentum correction
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```