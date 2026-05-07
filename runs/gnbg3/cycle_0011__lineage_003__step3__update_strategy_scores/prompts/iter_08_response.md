Looking at the task requirements and historical performance on task 10 (best: 1.127e-07), I need a hybrid approach that combines two distinct mechanisms for credit assignment. The key insight is that task 10 shows disagreement between variants, suggesting it benefits from adaptive operator selection.

**Idea: Confidence-weighted hybrid with EMA + rank improvement signal**

Combines temporal success-rate tracking (EMA with momentum) with rank-based improvement magnitude, weighted by per-strategy sample confidence. The confidence weighting naturally schedules between exploration (low confidence → equal weight) and exploitation (high confidence → preference for better signal).

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Hybrid credit: EMA success rate + rank improvement, weighted by sample confidence."""
    n_strategies = len(self.strategy_scores)
    NP = len(strategy_used)
    
    # Initialize tracking attributes if needed
    if not hasattr(self, '_strategy_ema_success'):
        self._strategy_ema_success = np.ones(n_strategies) * 0.5
        self._strategy_ema_rank_gain = np.zeros(n_strategies)
        self._strategy_samples = np.zeros(n_strategies)
        self._strategy_rank_history = [[] for _ in range(n_strategies)]
    
    # Compute per-individual rank improvement (how much did trial improve relative rank)
    if not hasattr(self, '_prev_fitness_ranks'):
        self._prev_fitness_ranks = np.ones(NP) * 0.5
    
    current_ranks = np.ones(NP) * 0.5
    for i in range(NP):
        if improved[i] and self._prev_fitness_ranks[i] < 0.99:
            rank_gain = self._prev_fitness_ranks[i] - 0.5  # Improvement from median
        else:
            rank_gain = 0.0
        current_ranks[i] = rank_gain
    
    # Update per-strategy statistics
    for s in range(n_strategies):
        mask = strategy_used == s
        count = np.sum(mask)
        if count > 0:
            self._strategy_samples[s] += count
            success_rate = np.mean(improved[mask])
            mean_rank_gain = np.mean(current_ranks[mask])
            
            # EMA update with momentum (higher momentum = more stable)
            ema_alpha = 0.3  # Base learning rate
            self._strategy_ema_success[s] = (1 - ema_alpha) * self._strategy_ema_success[s] + ema_alpha * success_rate
            self._strategy_ema_rank_gain[s] = (1 - ema_alpha) * self._strategy_ema_rank_gain[s] + ema_alpha * mean_rank_gain
            
            # Track rank history for variance estimation
            self._strategy_rank_history[s].extend(current_ranks[mask].tolist())
            # Keep bounded history
            if len(self._strategy_rank_history[s]) > 50:
                self._strategy_rank_history[s] = self._strategy_rank_history[s][-50:]
    
    # Compute confidence weights based on sample count and signal consistency
    min_samples = 5
    confidence = np.clip(self._strategy_samples / (self._strategy_samples + min_samples), 0.0, 0.95)
    
    # Also weight by inverse variance of rank gains (stable signals get higher confidence)
    for s in range(n_strategies):
        history = self._strategy_rank_history[s]
        if len(history) >= 5:
            var = np.var(history) + 1e-8
            # Lower variance = higher confidence (capped at 1.5x boost)
            variance_factor = np.clip(1.0 / (1.0 + np.sqrt(var)), 0.5, 1.5)
            confidence[s] *= variance_factor
    
    # Normalize confidence
    confidence_sum = confidence.sum() + 1e-10
    confidence /= confidence_sum
    
    # Hybrid score: confidence-weighted combination of success rate and rank improvement
    # Success rate component (normalized to [0,1])
    success_norm = self._strategy_ema_success.copy()
    
    # Rank gain component (shifted to be positive, normalized)
    rank_gain_norm = np.zeros(n_strategies)
    rank_offset = abs(self._strategy_ema_rank_gain.min()) + 0.1
    for s in range(n_strategies):
        rank_gain_norm[s] = self._strategy_ema_rank_gain[s] + rank_offset
    rank_gain_norm /= rank_gain_norm.max() + 1e-10
    
    # Combine with confidence weighting
    # When confidence is high, trust the signal more; when low, blend toward uniform
    hybrid_score = np.zeros(n_strategies)
    uniform_prior = 1.0 / n_strategies
    
    for s in range(n_strategies):
        signal_quality = 0.6 * success_norm[s] + 0.4 * rank_gain_norm[s]
        hybrid_score[s] = confidence[s] * signal_quality + (1 - confidence[s]) * uniform_prior
    
    # Apply softmax for final selection weights
    hybrid_score -= hybrid_score.max()
    exp_scores = np.exp(hybrid_score * 2.0)  # Temperature = 0.5
    self.strategy_scores = np.log(exp_scores / exp_scores.sum() + 1e-10) + 1.0
    
    # Store current ranks for next iteration
    self._prev_fitness_ranks = current_ranks.copy()
```