**Idea: Temporal Momentum with Trend Detection and Stagnation Escape**

Category F: Uses exponential moving averages of improvement rate and parameter autocorrelation over a rolling window to detect convergence trends, with momentum-weighted adaptation and stagnation-triggered perturbation to escape local optima.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using temporal dynamics: trend detection, momentum, and stagnation escape."""
    # Initialize temporal state tracking if not already present
    if not hasattr(self, '_improvement_ema'):
        self._improvement_ema = 0.0
        self._F_ema = self.F
        self._CR_ema = self.CR
        self._F_momentum = 0.0
        self._CR_momentum = 0.0
        self._improvement_history = []
        self._F_autocorr_lag = 1
        self._CR_autocorr_lag = 1
        self._trend_window = 15
        self._stagnation_threshold = 0.02
        self._stagnation_patience = 0
        self._max_patience = 8
        self._F_history_buffer = []
        self._CR_history_buffer = []
    
    # Track improvement rate history for temporal analysis
    self._improvement_history.append(improvement_rate)
    if len(self._improvement_history) > self._trend_window:
        self._improvement_history.pop(0)
    
    # Compute exponential moving average of improvement rate (temporal smoothing)
    alpha_imp = 0.25
    self._improvement_ema = (1 - alpha_imp) * self._improvement_ema + alpha_imp * improvement_rate
    
    # Store parameter history for autocorrelation analysis
    self._F_history_buffer.append(self.F)
    self._CR_history_buffer.append(self.CR)
    if len(self._F_history_buffer) > self._trend_window:
        self._F_history_buffer.pop(0)
    if len(self._CR_history_buffer) > self._trend_window:
        self._CR_history_buffer.pop(0)
    
    # Compute autocorrelation of F and CR (temporal dependency detection)
    def autocorr(series):
        if len(series) < 3:
            return 0.0
        n = len(series)
        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-12:
            return 0.0
        c0 = np.sum((np.array(series) - mean) ** 2) / n
        c1 = np.sum((np.array(series[:-1]) - mean) * (np.array(series[1:]) - mean)) / n
        return c1 / (c0 + 1e-12)
    
    F_autocorr = autocorr(self._F_history_buffer)
    CR_autocorr = autocorr(self._CR_history_buffer)
    
    # Detect convergence trend: high autocorrelation + low improvement = stagnation risk
    high_autocorr = (abs(F_autocorr) > 0.5) or (abs(CR_autocorr) > 0.5)
    low_improvement = self._improvement_ema < self._stagnation_threshold
    
    if high_autocorr and low_improvement:
        self._stagnation_patience += 1
    else:
        self._stagnation_patience = max(0, self._stagnation_patience - 1)
    
    # Compute rate of change in improvement rate (acceleration/deceleration)
    if len(self._improvement_history) >= 5:
        recent_trend = np.polyfit(range(5), self._improvement_history[-5:], 1)[0]
    else:
        recent_trend = 0.0
    
    # Adaptation logic based on temporal analysis
    momentum = 0.4
    base_lr = 0.08
    
    # Adaptive learning rate: slower when parameters are autocorrelated (stuck in pattern)
    effective_lr_F = base_lr * (1.0 - 0.3 * abs(F_autocorr))
    effective_lr_CR = base_lr * (1.0 - 0.3 * abs(CR_autocorr))
    
    # Determine adaptation direction based on improvement dynamics
    if self._improvement_ema > 0.15:
        # Good improvement: exploit (reduce F slightly, increase CR for convergence)
        target_F_change = -0.02
        target_CR_change = 0.03
    elif self._improvement_ema > 0.05:
        # Moderate improvement: balanced
        target_F_change = 0.0
        target_CR_change = 0.0
    elif recent_trend < -0.005:
        # Declining improvement: explore more (increase F)
        target_F_change = 0.05
        target_CR_change = -0.02
    else:
        # Low or stable improvement: minor perturbations
        target_F_change = 0.01 * np.sign(np.random.randn())
        target_CR_change = 0.01 * np.sign(np.random.randn())
    
    # Apply momentum to parameter changes
    self._F_momentum = momentum * self._F_momentum + (1 - momentum) * target_F_change
    self._CR_momentum = momentum * self._CR_momentum + (1 - momentum) * target_CR_change
    
    # Update parameters with momentum and adaptive learning rate
    new_F = self.F + effective_lr_F * self._F_momentum
    new_CR = self.CR + effective_lr_CR * self._CR_momentum
    
    # Stagnation escape: when stuck, apply larger perturbation
    if self._stagnation_patience >= self._max_patience:
        # Strong perturbation to break out of local optima
        new_F = np.clip(new_F + np.random.uniform(-0.3, 0.3), 0.3, 1.0)
        new_CR = np.clip(new_CR + np.random.uniform(-0.2, 0.2), 0.3, 0.95)
        self._stagnation_patience = 0
        self._improvement_ema = self._improvement_ema * 0.5  # Reset expectation
    
    # Update exponential moving averages for tracking
    self._F_ema = 0.9 * self._F_ema + 0.1 * new_F
    self._CR_ema = 0.9 * self._CR_ema + 0.1 * new_CR
    
    # Final clipping to valid ranges
    self.F = np.clip(new_F, 0.3, 1.0)
    self.CR = np.clip(new_CR, 0.3, 0.95)
    
    # Record history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
    
    # Keep history bounded
    max_history = 500
    if len(self.F_history) > max_history:
        self.F_history = self.F_history[-max_history:]
    if len(self.CR_history) > max_history:
        self.CR_history = self.CR_history[-max_history:]
```