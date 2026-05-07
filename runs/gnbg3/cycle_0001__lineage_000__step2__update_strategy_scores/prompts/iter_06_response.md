**Idea: EMA Momentum with Trend Detection**

Category F: Temporal / dynamical credit assignment using exponential moving averages of per-strategy improvement rates, momentum-based score updates, and trend detection to reward consistent performers while penalizing declining strategies.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Temporal strategy credit with EMA, momentum, and trend detection."""
    n_strategies = len(self.strategy_scores)
    
    # Initialize temporal state if first call
    if not hasattr(self, '_ema_rates'):
        self._ema_rates = np.zeros(n_strategies)
        self._prev_ema_rates = np.zeros(n_strategies)
        self._strategy_momentum = np.zeros(n_strategies)
        self._history_buffer = [np.zeros(n_strategies) for _ in range(5)]
        self._history_idx = 0
    
    # Compute per-strategy improvement rates for this generation
    current_rates = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.any():
            current_rates[s] = improved[mask].sum() / self.NP
    
    # Store in circular buffer for trend analysis
    self._history_buffer[self._history_idx] = current_rates.copy()
    self._history_idx = (self._history_idx + 1) % len(self._history_buffer)
    
    # Exponential moving average (alpha=0.3, favor recent data)
    alpha = 0.3
    self._prev_ema_rates = self._ema_rates.copy()
    self._ema_rates = alpha * current_rates + (1 - alpha) * self._ema_rates
    
    # Trend detection: compare current EMA to previous
    # Positive trend = improving, negative = declining
    trends = self._ema_rates - self._prev_ema_rates
    
    # Per-strategy momentum: weighted combination of current and trend
    # Momentum decays if strategy not used recently
    beta = 0.7  # momentum retention
    gamma = 0.15  # trend weight
    self._strategy_momentum = (
        beta * self._strategy_momentum +
        (1 - beta) * (self._ema_rates + gamma * np.sign(trends) * np.sqrt(np.abs(trends)))
    )
    
    # Compute adaptive learning rate per strategy (slower for consistent, faster for volatile)
    recent_variance = np.var(self._history_buffer, axis=0) + 1e-10
    adaptive_lr = 0.15 / (1.0 + 5.0 * recent_variance)
    
    # Update scores: EMA rate drives credit, momentum adds stability bonus
    score_delta = adaptive_lr * self._ema_rates + 0.05 * self._strategy_momentum
    
    # Penalize strategies showing sustained decline (negative trend)
    for s in range(n_strategies):
        recent_trend = sum(
            self._history_buffer[(self._history_idx - i) % len(self._history_buffer)][s]
            for i in range(1, 4)
        )
        if recent_trend < 0.02:  # Consistently low performance
            score_delta[s] -= 0.08
    
    # Apply update with momentum
    self.strategy_scores += score_delta
    
    # Normalize to prevent score divergence
    self.strategy_scores -= self.strategy_scores.mean()
    self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
```