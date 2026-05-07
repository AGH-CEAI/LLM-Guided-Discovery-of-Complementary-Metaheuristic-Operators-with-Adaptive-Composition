**Idea: Autocorrelation-Driven Temporal Regime Detection**
Use autocorrelation of the success rate time series to detect cyclic behavior and rate-of-change signals to distinguish convergence phases from stagnation, adjusting F/Cr accordingly.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using temporal autocorrelation, rate-of-change, and drift."""
    success_rate = np.mean(improved_mask)

    # Initialize temporal state on first call
    if not hasattr(self, 'success_history'):
        self.success_history = []
        self.F_ewma = self.F
        self.Cr_ewma = self.Cr
        self.stagnation_window = []
        self.stagnation_counter = 0
        self.last_best = np.inf

    # Append to rolling time series
    self.success_history.append(success_rate)
    if len(self.success_history) > 20:
        self.success_history.pop(0)

    # Track improvement for stagnation window
    current_best = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.inf
    improved_this_gen = current_best < self.last_best
    self.last_best = current_best

    self.stagnation_window.append(1 if improved_this_gen else 0)
    if len(self.stagnation_window) > 12:
        self.stagnation_window.pop(0)

    # Compute rate-of-change: derivative-like signal from EMA difference
    alpha_fast = 0.4
    alpha_slow = 0.15
    if not hasattr(self, 'ewma_fast'):
        self.ewma_fast = success_rate
        self.ewma_slow = success_rate
    self.ewma_fast = alpha_fast * success_rate + (1 - alpha_fast) * self.ewma_fast
    self.ewma_slow = alpha_slow * success_rate + (1 - alpha_slow) * self.ewma_slow
    rate_of_change = self.ewma_fast - self.ewma_slow  # Positive = improving, negative = declining

    # Compute autocorrelation at lag-1 using time series
    n_hist = len(self.success_history)
    autocorr = 0.0
    if n_hist >= 4:
        series = np.array(self.success_history)
        mean_series = np.mean(series)
        var_series = np.var(series)
        if var_series > 1e-12:
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
            autocorr = np.clip(autocorr, -1.0, 1.0)

    # Compute temporal drift: EMA-weighted parameter history
    self.F_ewma = 0.7 * self.F_ewma + 0.3 * F_used
    self.Cr_ewma = 0.7 * self.Cr_ewma + 0.3 * Cr_used
    F_drift_ratio = self.F / (self.F_ewma + 1e-12)
    Cr_drift_ratio = self.Cr / (self.Cr_ewma + 1e-12)

    # Stagnation severity from time-window
    stagnation_ratio = np.mean(self.stagnation_window)  # Fraction of generations with improvement

    # Regime detection via temporal signals
    is_oscillating = autocorr > 0.4 and abs(rate_of_change) < 0.05
    is_declining = rate_of_change < -0.02
    is_stagnant = stagnation_ratio < 0.2
    is_converging = rate_of_change > 0.02 and stagnation_ratio > 0.5

    # Temporal adjustment logic
    if is_oscillating:
        # Cyclic behavior detected: inject diversity via F boost
        F_adjustment = 1.25 * np.clip(F_drift_ratio, 0.8, 1.3)
        Cr_adjustment = 0.85
        self.stagnation_counter += 1
    elif is_declining and is_stagnant:
        # Declining with stagnation: strong exploration push
        F_adjustment = 1.15 * (1.1 - stagnation_ratio) / np.clip(F_drift_ratio, 0.8, 1.4)
        Cr_adjustment = 0.88
        self.stagnation_counter += 1
    elif is_stagnant and stagnation_ratio < 0.1:
        # Deep stagnation: large perturbation
        F_adjustment = 0.7
        Cr_adjustment = 1.2
        self.stagnation_counter += 2
    elif is_converging:
        # Steady improvement: fine-tune toward exploitation
        F_adjustment = 0.95 * np.clip(F_drift_ratio, 0.9, 1.1)
        Cr_adjustment = 1.05
        self.stagnation_counter = max(0, self.stagnation_counter - 1)
    else:
        # Neutral: gentle correction based on rate
        F_adjustment = 1.0 + 0.1 * rate_of_change
        Cr_adjustment = 1.0 - 0.05 * rate_of_change

    # Apply adjustments
    self.F = np.clip(self.F * F_adjustment, 0.1, 2.0)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.05, 0.95)

    # Decay stagnation counter slowly
    self.stagnation_counter = max(0, self.stagnation_counter - 0.1)
```