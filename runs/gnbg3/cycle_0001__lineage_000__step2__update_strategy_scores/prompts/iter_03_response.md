**Idea: KL Divergence Strategy Credit with Improvement Entropy**

Use information-theoretic measures to score strategies: compute how much each strategy's improvement distribution diverges from an "ideal" delta-focused distribution using KL divergence, combined with entropy of the improvement distribution to reward consistent performers.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment using information-theoretic analysis of improvement distributions."""
    NP = len(strategy_used)
    n_strategies = len(self.strategy_names)
    
    # Compute per-individual improvement magnitude (relative to current fitness)
    improvement_magnitude = np.zeros(NP)
    for i in range(NP):
        if hasattr(self, '_prev_fitness') and self._prev_fitness[i] not in [np.nan, np.inf]:
            delta = self._prev_fitness[i] - max(self._prev_fitness[i] * 0.01, 1e-10)
            improvement_magnitude[i] = np.clip(delta / (np.abs(self._prev_fitness[i]) + 1e-10), -10, 10)
        else:
            improvement_magnitude[i] = 0.0
    
    self._prev_fitness = None  # Reset after use
    
    # Bin improvements into histogram for each strategy
    n_bins = 10
    bin_edges = np.linspace(-2, 2, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Compute empirical distribution per strategy
    kl_scores = np.zeros(n_strategies)
    entropy_bonus = np.zeros(n_strategies)
    
    for s in range(n_strategies):
        mask = strategy_used == s
        count = np.sum(mask)
        
        if count < 2:
            kl_scores[s] = -5.0  # Penalize insufficient data
            entropy_bonus[s] = 0.0
            continue
        
        # Compute histogram of improvement magnitudes
        hist, _ = np.histogram(improvement_magnitude[mask], bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        hist = np.clip(hist, 1e-10, 1.0)
        hist /= hist.sum() + 1e-10  # Renormalize
        
        # Ideal distribution: concentrated at positive improvements
        ideal = np.exp(-0.5 * ((bin_centers - 0.5) / 0.8) ** 2)
        ideal = np.clip(ideal, 1e-10, 1.0)
        ideal /= ideal.sum() + 1e-10
        
        # KL divergence from ideal (positive = better than uniform random)
        kl_div = np.sum(hist * np.log(hist / ideal))
        kl_scores[s] = kl_div
        
        # Entropy bonus: lower entropy = more consistent = bonus
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        max_entropy = np.log(n_bins)
        entropy_bonus[s] = 1.0 - (entropy / (max_entropy + 1e-10))
    
    # Combine KL divergence and entropy bonus
    combined = kl_scores + 0.3 * entropy_bonus
    
    # Also consider raw success rate for reinforcement
    success_rate = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if np.sum(mask) > 0:
            success_rate[s] = np.mean(improved[mask])
    
    # Blend information-theoretic score with success rate
    alpha = 0.6  # Weight for info-theoretic score
    final_scores = alpha * combined + (1 - alpha) * success_rate
    
    # Update scores with exponential moving average
    momentum = 0.7
    self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * final_scores
    
    # Ensure scores don't collapse
    self.strategy_scores = np.clip(self.strategy_scores, -10.0, 10.0)
```