**Idea: UCB-Based Strategy Selection**

Instead of simple success-weighted probabilities, use Upper Confidence Bound to balance exploitation of known good strategies with exploration of under-explored ones. This addresses the exploration-exploitation tradeoff that basic success weighting ignores, which is critical for tasks like Task 21 where marginal improvements matter.

```python
def _adapt_strategy_probabilities(self):
    # Use Upper Confidence Bound to balance exploitation vs exploration
    attempts = self.strategy_attempt_count + 1e-10
    successes = self.strategy_success_count + 1e-10
    
    # Success rate
    success_rate = successes / attempts
    
    # UCB bonus term: decreases with more attempts (encourages exploration of under-tested)
    ucb_bonus = np.sqrt(2.0 * np.log(np.sum(attempts) + 1) / attempts)
    
    # UCB values = exploitation + exploration bonus
    ucb_values = success_rate + 0.5 * ucb_bonus
    
    # Clip to avoid numerical issues
    ucb_values = np.clip(ucb_values, 1e-10, None)
    
    # Normalize to probabilities
    probs = ucb_values / (np.sum(ucb_values) + 1e-10)
    
    for i in range(self.NP):
        self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```