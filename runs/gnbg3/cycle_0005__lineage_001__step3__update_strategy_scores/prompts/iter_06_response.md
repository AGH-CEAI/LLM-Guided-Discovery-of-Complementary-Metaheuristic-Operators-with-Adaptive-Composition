**Idea: EMA Momentum with Temporal Drift Detection**

Use exponential moving averages of per-strategy success rates, detect momentum shifts (acceleration/deceleration of improvement), and apply temporal drift correction to strategy scores. This captures how strategy effectiveness changes over time rather than just instantaneous performance.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Temporal credit assignment using EMA momentum and drift detection."""
    n_strategies = len(self.strategy_scores)
    
    # Initialize temporal tracking attributes
    if not hasattr(self, '_ema_rates'):
        self._ema_rates = np.ones(n_strategies) * 0.5
    if not hasattr(self, '_ema_alpha'):
        self._ema_alpha = 0.3  # EMA smoothing factor
    if not hasattr(self, '_prev_improvement_rates'):
        self._prev_improvement_rates = np.zeros(n_strategies)
    if not hasattr(self, '_temporal_momentum'):
        self._temporal_momentum = np.zeros(n_strategies)
    if not hasattr(self, '_drift_history'):
        self._drift_history = []
    
    # Compute per-strategy improvement rates for this generation
    current_rates = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() > 0:
            current_rates[s] = improved[mask].mean()
    
    # Update exponential moving averages with momentum
    alpha = self._ema_alpha
    new_ema = alpha * current_rates + (1 - alpha) * self._ema_rates
    
    # Compute momentum: rate of change in EMA
    momentum = new_ema - self._ema_rates
    
    # Detect temporal drift: sustained directional change
    self._temporal_momentum = 0.7 * self._temporal_momentum + 0.3 * momentum
    
    # Store for next iteration
    self._prev_improvement_rates = current_rates.copy()
    self._ema_rates = new_ema.copy()
    
    # Drift detection: compare recent trend to historical baseline
    drift_signal = np.zeros(n_strategies)
    if len(self._drift_history) >= 3:
        baseline = np.mean(self._drift_history[-3:], axis=0)
        recent = new_ema
        drift_signal = recent - baseline
    self._drift_history.append(new_ema.copy())
    if len(self._drift_history) > 10:
        self._drift_history.pop(0)
    
    # Compute adaptive learning rate based on signal confidence
    rate_variance = np.var(current_rates)
    adaptive_lr = 0.1 / (1.0 + rate_variance * 10)
    
    # Credit assignment: combine momentum and drift
    # Positive momentum + positive drift = accelerating strategy
    # Negative momentum + negative drift = declining strategy
    credit = self._temporal_momentum * 0.6 + drift_signal * 0.4
    
    # Apply momentum-boosted credit to scores
    self.strategy_scores += adaptive_lr * credit * 10.0
    
    # Temporal decay: slowly reduce scores of underperforming strategies
    decay_mask = (current_rates < 0.1) & (self._ema_rates < 0.2)
    self.strategy_scores[decay_mask] *= 0.97
    
    # Soft normalization to prevent extreme divergence
    self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)
    
    # Ensure minimum score diversity
    min_diff = 0.1
    for i in range(n_strategies):
        for j in range(i + 1, n_strategies):
            diff = abs(self.strategy_scores[i] - self.strategy_scores[j])
            if diff < min_diff:
                self.strategy_scores[i] += min_diff * 0.5
                self.strategy_scores[j] -= min_diff * 0.5
```