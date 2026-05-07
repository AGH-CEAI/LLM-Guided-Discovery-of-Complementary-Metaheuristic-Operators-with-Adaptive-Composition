**Idea: Temporal Momentum with Adaptive Learning Rate**

Use exponential moving averages of success rate, momentum-based trend detection, and an adaptive learning rate that responds to the volatility of success patterns across generations. This captures the temporal dynamics of strategy effectiveness rather than just single-generation snapshots.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Temporal credit assignment with EMA, momentum, and volatility-adaptive learning."""
    n_strategies = len(self.strategy_scores)
    
    # Initialize temporal state tracking
    if not hasattr(self, '_ema_success'):
        self._ema_success = np.ones(n_strategies) * 0.2
        self._ema_sq_success = np.ones(n_strategies) * 0.04
        self._momentum = np.zeros(n_strategies)
        self._success_history = []
        self._gen_count = 0
    
    self._gen_count += 1
    
    # Compute per-strategy success rate this generation
    gen_success = np.zeros(n_strategies)
    for s, imp in zip(strategy_used, improved):
        gen_success[s] += 1
    gen_success /= max(np.sum(improved), 1)
    
    # Store in rolling history (last 10 generations)
    self._success_history.append(gen_success)
    if len(self._success_history) > 10:
        self._success_history.pop(0)
    
    # Exponential moving average of success rate (alpha=0.3)
    alpha = 0.3
    self._ema_success = alpha * gen_success + (1 - alpha) * self._ema_success
    self._ema_sq_success = alpha * (gen_success ** 2) + (1 - alpha) * self._ema_sq_success
    
    # Compute variance from EMA moments
    variance = np.maximum(self._ema_sq_success - self._ema_success ** 2, 1e-10)
    std_dev = np.sqrt(variance)
    
    # Momentum: rate of change of EMA (positive = improving, negative = degrading)
    if len(self._success_history) >= 3:
        recent_trend = self._ema_success - np.array(self._success_history[-3]).mean(axis=0)
        self._momentum = 0.6 * self._momentum + 0.4 * recent_trend
    else:
        self._momentum *= 0.9
    
    # Volatility-adaptive learning rate: high volatility -> higher exploration weight
    # Normalize std_dev relative to theoretical max (0.5 for Bernoulli)
    rel_volatility = std_dev / 0.5
    adaptive_lr = 0.08 * (1.0 + 0.5 * rel_volatility)
    adaptive_lr = np.clip(adaptive_lr, 0.03, 0.18)
    
    # Combine EMA performance with momentum trend
    # Weight: 70% current performance, 30% momentum direction
    temporal_credit = 0.7 * self._ema_success + 0.3 * np.clip(self._momentum + 0.2, 0, 1)
    
    # Normalize by generation-scaled recency: newer strategies gain slight boost
    recency_factor = np.ones(n_strategies)
    
    # Update scores with momentum-weighted learning
    score_delta = adaptive_lr * temporal_credit * (1.0 + 0.2 * self._momentum)
    self.strategy_scores += score_delta
    
    # Soft floor to prevent collapse
    self.strategy_scores = np.maximum(self.strategy_scores, 0.05)
    
    # Decay all scores slightly to allow re-exploration
    self.strategy_scores *= 0.998
```