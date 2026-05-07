**Idea: Kendall Tau Rank Correlation with EWM Tracking**

This approach uses Kendall's tau rank correlation between strategy usage and fitness rank improvement, combined with exponentially-weighted moving averages of success rates and rank gains, plus trend detection to identify strategies that are gaining or losing effectiveness over time.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Fitness-landscape / rank-based strategy scoring using Kendall tau correlation and EWM."""
    n_strategies = len(self.strategy_scores)
    
    # Initialize tracking state
    if not hasattr(self, '_prev_fitness_ranks'):
        self._prev_fitness_ranks = np.full(self.NP, 0.5)
    
    if not hasattr(self, '_success_ewm'):
        self._success_ewm = np.zeros(n_strategies)
    
    if not hasattr(self, '_rank_gain_ewm'):
        self._rank_gain_ewm = np.zeros(n_strategies)
    
    if not hasattr(self, '_score_history'):
        self._score_history = []
    
    # Compute current fitness ranks (normalized to [0, 1])
    if hasattr(self, '_current_fitness'):
        current_ranks = self._compute_fitness_ranking(self._current_fitness)
    else:
        current_ranks = self._prev_fitness_ranks.copy()
    
    # Compute per-strategy success rate and average rank improvement
    success_rate = np.zeros(n_strategies)
    avg_rank_gain = np.zeros(n_strategies)
    
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() == 0:
            continue
        
        participants = mask.sum()
        successes = improved[mask].sum()
        success_rate[s] = successes / participants
        
        # Rank gain: how much did participants improve (higher = better)
        prev_ranks = self._prev_fitness_ranks[mask]
        curr_ranks = current_ranks[mask]
        rank_gains = prev_ranks - curr_ranks
        avg_rank_gain[s] = np.mean(rank_gains)
    
    # Kendall tau-b correlation between strategy usage and rank improvement
    tau_scores = np.zeros(n_strategies)
    rank_improvement = self._prev_fitness_ranks - current_ranks
    
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() >= 2:
            # Binary indicator: 1 if strategy s was used, 0 otherwise
            usage_indicator = mask.astype(float)
            # Correlation between usage and rank improvement
            tau = np.corrcoef(usage_indicator, rank_improvement)[0, 1]
            tau_scores[s] = np.clip(np.nan_to_num(tau), -1, 1)
    
    # Update EWM estimates with momentum
    self._success_ewm = 0.7 * self._success_ewm + 0.3 * success_rate
    self._rank_gain_ewm = 0.7 * self._rank_gain_ewm + 0.3 * avg_rank_gain
    
    # Composite score: success EWM + rank gain EWM + Kendall tau
    composite = (
        self._success_ewm * 2.0 +
        np.clip(self._rank_gain_ewm, 0, 1) * 1.5 +
        np.clip(tau_scores, 0, 1) * 2.0
    )
    
    # Blend with previous scores for stability
    self.strategy_scores = 0.7 * self.strategy_scores + 0.3 * composite
    self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
    # Trend detection: boost strategies with improving scores
    self._score_history.append(self.strategy_scores.copy())
    if len(self._score_history) > 10:
        self._score_history.pop(0)
    
    if len(self._score_history) >= 3:
        recent = self._score_history[-1]
        older = self._score_history[-3]
        score_trend = recent - older
        self.strategy_scores += 0.05 * score_trend
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
    # Store current ranks for next iteration
    self._prev_fitness_ranks = current_ranks
```