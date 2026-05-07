**Idea: EMA-Momentum Hybrid with Confidence-Weighted Selection**

Hybrid Category H approach for task 17: combines (1) exponential moving average of per-strategy success rates for temporal smoothness with (2) momentum-based trend detection to identify improving vs declining strategies, weighted by bootstrap confidence to handle sparse data. This is fundamentally different from pure fitness-rank, temporal-only, or geometric approaches.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Hybrid credit: EMA-based success rate + momentum trend, confidence-weighted."""
    n_strategies = len(self.strategy_scores)
    
    # Initialize tracking attributes if needed
    if not hasattr(self, '_strategy_ema'):
        self._strategy_ema = np.ones(n_strategies) * 0.5
    if not hasattr(self, '_strategy_momentum'):
        self._strategy_momentum = np.zeros(n_strategies)
    if not hasattr(self, '_strategy_history'):
        self._strategy_history = [np.ones(n_strategies) * 0.5] * 5
    if not hasattr(self, '_ema_alpha'):
        self._ema_alpha = 0.3  # Smoothing factor
    
    # Compute per-strategy success rate this generation
    strategy_success = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() > 0:
            strategy_success[s] = improved[mask].mean()
    
    # Mechanism 1: Exponential Moving Average for temporal smoothness
    alpha = self._ema_alpha
    new_ema = alpha * strategy_success + (1 - alpha) * self._strategy_ema
    
    # Mechanism 2: Momentum-based trend detection
    self._strategy_history.append(strategy_success)
    if len(self._strategy_history) > 8:
        self._strategy_history.pop(0)
    
    # Compute trend via linear regression slope over history
    trends = np.zeros(n_strategies)
    if len(self._strategy_history) >= 4:
        T = len(self._strategy_history)
        x = np.arange(T)
        x_mean = (T - 1) / 2.0
        for s in range(n_strategies):
            y = np.array([h[s] for h in self._strategy_history])
            slope = np.sum((x - x_mean) * (y - y.mean())) / max(np.sum((x - x_mean) ** 2), 1e-10)
            trends[s] = np.clip(slope, -0.5, 0.5)
    
    # Confidence weighting: use sample count to downweight sparse data
    sample_counts = np.array([np.sum(strategy_used == s) for s in range(n_strategies)])
    total_samples = max(sample_counts.sum(), 1)
    confidence = np.clip(sample_counts / total_samples * n_strategies, 0.1, 1.0)
    
    # Combine mechanisms: EMA base + momentum boost, confidence-weighted
    momentum_weight = 0.25
    hybrid_score = new_ema + momentum_weight * trends * confidence
    
    # Update momentum accumulator
    self._strategy_momentum = 0.7 * self._strategy_momentum + 0.3 * trends
    
    # Update EMA tracker
    self._strategy_ema = new_ema
    
    # Adaptive alpha: faster adaptation when momentum indicates change
    momentum_magnitude = np.abs(self._strategy_momentum).mean()
    self._ema_alpha = np.clip(0.2 + 0.3 * momentum_magnitude * 5, 0.15, 0.5)
    
    # Apply softmax to get probabilities, with floor to prevent starvation
    score_floor = np.percentile(hybrid_score, 25)
    adjusted = np.maximum(hybrid_score - score_floor, 0.0) + 0.1
    probs = np.exp(adjusted * 5)
    probs /= probs.sum()
    
    # Update strategy scores for selection (with recency bias)
    recency_bonus = 0.1 * (np.random.rand(n_strategies) - 0.5)
    self.strategy_scores = self.strategy_scores * 0.85 + (probs * 10 + recency_bonus) * 0.15
    
    # Ensure minimum score to prevent complete extinction
    self.strategy_scores = np.maximum(self.strategy_scores, 0.1)
```