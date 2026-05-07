**Idea: Diversity-Momentum & Stagnation-Triggered Inertia**

Category F: Temporal / dynamical — uses exponential moving average of diversity, rate-of-change detection, and stagnation windows to adapt inertia weight across generations (not just snapshot-based).

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using temporal dynamics: diversity momentum, rate-of-change, and stagnation."""
    current_diversity = self._compute_diversity()
    
    # Initialize temporal state on first call
    if not hasattr(self, '_ema_diversity'):
        self._ema_diversity = current_diversity
        self._prev_diversity = current_diversity
        self._diversity_history = [current_diversity]
        self._fitness_stagnation_window = []
        self._inertia_adjustment_momentum = 0.0
    
    # Exponential moving average of diversity (smoothing across generations)
    alpha = 0.2  # EMA smoothing factor
    self._ema_diversity = alpha * current_diversity + (1 - alpha) * self._ema_diversity
    
    # Rate of change of diversity (temporal derivative)
    diversity_delta = current_diversity - self._prev_diversity
    self._prev_diversity = current_diversity
    
    # Accumulate diversity history for window-based analysis
    self._diversity_history.append(current_diversity)
    if len(self._diversity_history) > 50:
        self._diversity_history.pop(0)
    
    # Stagnation detection: track if fitness is improving
    if len(self._fitness_stagnation_window) < 20:
        self._fitness_stagnation_window.append(self.global_best_fitness)
    else:
        self._fitness_stagnation_window.pop(0)
        self._fitness_stagnation_window.append(self.global_best_fitness)
    
    # Detect fitness stagnation over window
    fitness_improving = False
    if len(self._fitness_stagnation_window) >= 5:
        recent_improvement = self._fitness_stagnation_window[0] - self._fitness_stagnation_window[-1]
        fitness_improving = recent_improvement > 1e-6
    
    # Compute trend: linear regression slope over diversity history
    trend_slope = 0.0
    if len(self._diversity_history) >= 5:
        n = len(self._diversity_history)
        x = np.arange(n)
        x_mean = np.mean(x)
        y = np.array(self._diversity_history)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2) + 1e-10
        trend_slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    
    # Momentum-based adjustment: track direction consistency
    momentum_decay = 0.8
    self._inertia_adjustment_momentum = (
        momentum_decay * self._inertia_adjustment_momentum + 
        (1 - momentum_decay) * np.sign(diversity_delta)
    )
    
    # Determine inertia adjustment based on temporal signals
    if self._ema_diversity < self.diversity_threshold_low:
        # Low diversity: increase exploration
        if trend_slope < -1e-6 and not fitness_improving:
            # Diversity declining AND fitness stagnating: aggressive exploration
            adjustment = 1.15
        elif abs(self._inertia_adjustment_momentum) > 0.5:
            # Momentum indicates sustained direction: amplify response
            adjustment = 1.1
        else:
            adjustment = 1.05
        self.inertia_weight = min(0.95, self.inertia_weight * adjustment)
        
    elif self._ema_diversity > self.diversity_threshold_high:
        # High diversity: increase exploitation
        if trend_slope > 1e-6 and fitness_improving:
            # Diversity growing AND fitness improving: moderate reduction
            adjustment = 0.9
        else:
            adjustment = 0.95
        self.inertia_weight = max(0.4, self.inertia_weight * adjustment)
        
    else:
        # Normal range: use trend to modulate schedule
        base_inertia = 0.729 - 0.1 * (self.generation / 1000)
        
        # If diversity has been steadily declining (convergence pressure), 
        # counteract with slightly higher inertia
        if trend_slope < -0.5:
            base_inertia += 0.1
        # If diversity has been steadily increasing (dispersion pressure),
        # allow slightly lower inertia for faster convergence
        elif trend_slope > 0.5:
            base_inertia -= 0.05
        
        self.inertia_weight = np.clip(base_inertia, 0.4, 0.95)
```