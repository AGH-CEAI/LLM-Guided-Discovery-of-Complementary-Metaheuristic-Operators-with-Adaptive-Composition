**Idea: Rank-Correlation Adaptive Credit Assignment**

This approach uses Spearman rank correlation between strategy usage and fitness rank improvement, combined with percentile-based improvement signals and adaptive learning rates based on population-wide performance variance. It operates entirely on fitness-derived signals (ranks, percentiles) without any geometric or distance computations.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment using rank-correlation and fitness percentiles."""
    NP = len(strategy_used)
    n_strategies = len(self.strategy_scores)
    
    # Compute fitness rank improvement signals (Category D: fitness-based only)
    improved_int = improved.astype(float)
    
    # Calculate percentile-based improvement signal: how much did improved individuals
    # rank above the median in the population?
    improved_ranks = np.zeros(NP)
    if improved.sum() > 0:
        # Rank individuals by their improvement contribution (favor strategies that
        # helped already-good individuals)
        rank_signal = np.zeros(NP)
        rank_signal[improved] = 1.0 + np.linspace(0.5, 1.0, improved.sum())
        improved_ranks = rank_signal
    
    # Compute Spearman correlation between each strategy and improvement signal
    correlations = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() >= 2:
            # Rank correlation: how often does using strategy s coincide with improvement
            try:
                corr = np.corrcoef(
                    mask.astype(float), 
                    improved_int
                )[0, 1]
                correlations[s] = np.clip(corr, -1.0, 1.0)
            except Exception:
                correlations[s] = 0.0
    
    # Compute strategy-specific improvement rates
    strategy_improvement = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() > 0:
            strategy_improvement[s] = improved_int[mask].mean()
    
    # Population-wide baseline for adaptive learning rate
    pop_improvement_rate = improved.mean()
    pop_variance = max(0.01, np.var(strategy_improvement))
    adaptive_lr = np.clip(0.1 + 0.2 * pop_variance, 0.05, 0.4)
    
    # Initialize rolling history if needed
    if not hasattr(self, '_strategy_history'):
        self._strategy_history = np.zeros((n_strategies, 10))
        self._history_idx = 0
    
    # Store current correlations in rolling history
    self._strategy_history[:, self._history_idx % 10] = correlations
    self._history_idx += 1
    
    # Exponential moving average of correlations for stability
    ema_alpha = 0.3
    smoothed_corr = np.zeros(n_strategies)
    for s in range(n_strategies):
        history = self._strategy_history[s, :min(self._history_idx, 10)]
        if len(history) > 0:
            smoothed_corr[s] = np.mean(history)
    
    # Compute performance ratio: strategy improvement vs population baseline
    # This is a fitness-landscape signal (relative fitness improvement)
    perf_ratio = np.zeros(n_strategies)
    for s in range(n_strategies):
        if pop_improvement_rate > 0.01:
            perf_ratio[s] = strategy_improvement[s] / (pop_improvement_rate + 0.05)
        else:
            perf_ratio[s] = 1.0
    
    # Combine rank correlation and performance ratio
    # Weight: 60% correlation signal, 40% improvement rate signal
    combined_signal = 0.6 * smoothed_corr + 0.4 * (perf_ratio - 1.0) * 0.5
    combined_signal = np.clip(combined_signal, -2.0, 2.0)
    
    # Update scores with momentum
    momentum = 0.7
    score_delta = adaptive_lr * combined_signal
    self.strategy_scores = momentum * self.strategy_scores + score_delta
    
    # Clip to prevent score explosion
    self.strategy_scores = np.clip(self.strategy_scores, -10.0, 10.0)
    
    # Temperature-controlled softmax for selection probabilities
    # Higher temperature = more exploration of worse strategies
    temperature = max(0.5, 2.0 - 0.5 * pop_variance)
    scores = self.strategy_scores - self.strategy_scores.max()
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores / exp_scores.sum()
    
    # Ensure no strategy gets zero probability
    probs = np.clip(probs, 0.01, 0.99)
    probs /= probs.sum()
```