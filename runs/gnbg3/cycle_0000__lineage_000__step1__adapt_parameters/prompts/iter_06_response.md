**Idea: Derivative-Driven Momentum with Stagnation Detection**
Use exponential moving averages of improvement rate, first/second derivatives to detect acceleration/deceleration patterns, and momentum-based parameter updates tuned for fine-grained convergence on task 4.

```python
def _adapt_parameters(self, improvement_rate):
    # Temporal state initialization
    if not hasattr(self, '_temp_improvement_history'):
        self._temp_improvement_history = []
        self._temp_ema_improvement = improvement_rate
        self._temp_ema_derivative = 0.0
        self._temp_F_momentum = 0.0
        self._temp_CR_momentum = 0.0
        self._temp_stagnation_count = 0
        self._temp_prev_improvement = improvement_rate
        self._temp_window_size = 12
    
    # Track improvement rate over time
    self._temp_improvement_history.append(improvement_rate)
    if len(self._temp_improvement_history) > self._temp_window_size:
        self._temp_improvement_history.pop(0)
    
    # Exponential moving average of improvement rate (temporal smoothing)
    alpha = 0.25
    self._temp_ema_improvement = alpha * improvement_rate + (1 - alpha) * self._temp_ema_improvement
    
    # First derivative: rate of change of improvement
    if len(self._temp_improvement_history) >= 3:
        deriv = self._temp_improvement_history[-1] - self._temp_improvement_history[-3]
    else:
        deriv = improvement_rate - self._temp_prev_improvement
    self._temp_prev_improvement = improvement_rate
    
    # EMA of derivative (smoother trend signal)
    self._temp_ema_derivative = 0.4 * deriv + 0.6 * self._temp_ema_derivative
    
    # Second derivative: acceleration of improvement rate
    if len(self._temp_improvement_history) >= 5:
        accel = (self._temp_improvement_history[-1] - self._temp_improvement_history[-3]) - \
                (self._temp_improvement_history[-3] - self._temp_improvement_history[-5])
    else:
        accel = 0.0
    
    # Stagnation detection across time window
    if improvement_rate < 0.03:
        self._temp_stagnation_count += 1
    else:
        self._temp_stagnation_count = max(0, self._temp_stagnation_count - 1)
    
    # Compute delta_F and delta_CR based on temporal signals
    delta_F = 0.0
    delta_CR = 0.0
    
    # Acceleration-based adjustment (second derivative)
    if accel < -0.02:
        delta_F += 0.08
        delta_CR -= 0.03
    elif accel > 0.02:
        delta_F -= 0.05
        delta_CR += 0.02
    
    # First derivative: trending improvement or decline
    if self._temp_ema_derivative < -0.03:
        delta_F += 0.12
        delta_CR -= 0.04
    elif self._temp_ema_derivative > 0.03:
        delta_F -= 0.08
        delta_CR += 0.03
    
    # Stagnation: trigger exploration mode
    if self._temp_stagnation_count > 10:
        delta_F += 0.18
        delta_CR -= 0.06
    elif self._temp_stagnation_count > 5:
        delta_F += 0.08
        delta_CR -= 0.03
    
    # Fine-grained convergence mode when improvement is steady and high
    if self._temp_ema_improvement > 0.4 and self._temp_stagnation_count == 0:
        delta_CR += 0.02
        delta_F -= 0.02
    
    # Momentum-based update (smooths transitions across generations)
    momentum = 0.65
    self._temp_F_momentum = momentum * self._temp_F_momentum + (1 - momentum) * delta_F
    self._temp_CR_momentum = momentum * self._temp_CR_momentum + (1 - momentum) * delta_CR
    
    # Apply momentum to parameters
    self.F = np.clip(self.F + self._temp_F_momentum, 0.3, 1.5)
    self.CR = np.clip(self.CR + self._temp_CR_momentum, 0.1, 0.9)
    
    # Record for external tracking
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```