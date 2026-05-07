**Idea: Rank-Success Hybrid with Variance-Weighted Confidence**

Combine (1) fitness-rank-based credit with (2) binary success-rate credit, weighted by the inverse-variance of each signal to give more weight to consistent signals. Uses EMA smoothing and momentum for temporal stability.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Hybrid credit: rank improvement + success rate, weighted by signal confidence."""
    n_strategies = len(self.strategy_scores)
    n = len(strategy_used)
    
    # Initialize tracking attributes
    if not hasattr(self, '_ema_rank_improvement'):
        self._ema_rank_improvement = np.zeros(n_strategies)
        self._ema_success_rate = np.zeros(n_strategies)
        self._rank_improvement_history = [[] for _ in range(n_strategies)]
        self._success_rate_history = [[] for _ in range(n_strategies)]
        self._momentum = 0.3
    
    # Compute per-strategy rank improvement (D: fitness-landscape based)
    # Higher rank improvement = better credit
    rank_improvement = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() > 0:
            rank_improvement[s] = improved[mask].sum() / mask.sum()
    
    # Compute per-strategy success rate (D: fitness-landscape based)
    success_rate = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() > 0:
            success_rate[s] = improved[mask].sum() / mask.sum()
    
    # Update history for variance computation
    for s in range(n_strategies):
        self._rank_improvement_history[s].append(rank_improvement[s])
        self._success_rate_history[s].append(success_rate[s])
        # Keep bounded history
        if len(self._rank_improvement_history[s]) > 10:
            self._rank_improvement_history[s].pop(0)
        if len(self._success_rate_history[s]) > 10:
            self._success_rate_history[s].pop(0)
    
    # Compute EMA with momentum for temporal stability (F: temporal/dynamical)
    alpha = 0.3
    for s in range(n_strategies):
        self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
        self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]
    
    # Compute variance of each signal per strategy (principled weighting signal)
    rank_var = np.zeros(n_strategies)
    success_var = np.zeros(n_strategies)
    for s in range(n_strategies):
        if len(self._rank_improvement_history[s]) >= 3:
            rank_var[s] = np.var(self._rank_improvement_history[s])
        if len(self._success_rate_history[s]) >= 3:
            success_var[s] = np.var(self._success_rate_history[s])
    
    # Weight by inverse variance (data-driven): consistent signals get more weight
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    rank_weight = np.where(rank_var > eps, 1.0 / (rank_var + eps), 1.0 / eps)
    success_weight = np.where(success_var > eps, 1.0 / (success_var + eps), 1.0 / eps)
    
    # Normalize weights
    total_rank_weight = rank_weight.sum()
    total_success_weight = success_weight.sum()
    if total_rank_weight > 0:
        rank_weight /= total_rank_weight
    if total_success_weight > 0:
        success_weight /= total_success_weight
    
    # Compute hybrid signal: weighted combination of rank improvement and success rate
    hybrid_signal = 0.5 * rank_weight * self._ema_rank_improvement + 0.5 * success_weight * self._ema_success_rate
    
    # Comparative advantage: how much better is this strategy vs population mean
    mean_signal = hybrid_signal.mean()
    if mean_signal > 0:
        comparative_advantage = (hybrid_signal - mean_signal) / (mean_signal + eps)
    else:
        comparative_advantage = np.zeros(n_strategies)
    
    # Compute confidence factor based on total effective sample size
    # Higher confidence = lower variance = more consistent performance
    combined_var = rank_var + success_var
    confidence = np.exp(-combined_var * 5)  # Exponential decay with variance
    
    # Score update with momentum and confidence weighting
    delta = np.zeros(n_strategies)
    for s in range(n_strategies):
        # Combine comparative advantage with absolute performance
        delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
        delta[s] *= (0.5 + 0.5 * confidence[s])  # Scale by confidence
    
    # Apply update
    self.strategy_scores += delta
    
    # Ensure scores stay positive (softmax needs positive values)
    self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
```