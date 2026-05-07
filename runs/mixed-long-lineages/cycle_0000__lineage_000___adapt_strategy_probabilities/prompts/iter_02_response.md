Looking at the uncovered tasks (3, 5, 10, 12, 13, 15, 16, 17, 18, 20), the current variant makes performance **worse** on all of them. The key patterns are:

1. **Small-error tasks** (10, 12): The current adaptation is too disruptive to already-converged solutions
2. **Large-scale/hard tasks** (5, 13, 16, 17): The strategy selection loses diversity or picks wrong strategies
3. **Moderate tasks** (3, 15, 18, 20): Current approach lacks robustness for these landscapes

The current method only uses success **count** (how many improvements per strategy) without considering the **magnitude** of improvements or the current state of convergence. It also lacks diversity pressure, leading to premature convergence to wrong strategies.

**Idea: Magnitude-Weighted Entropy-Balanced Strategy Selection**
Use a fundamentally different signal: combine improvement magnitude with entropy-based diversity bonus. This penalizes strategies that produce tiny/noisy improvements and rewards exploration when the current best strategies aren't delivering meaningful gains.

```python
def _adapt_strategy_probabilities(self):
    # Compute magnitude-weighted scores (not just counts)
    # Weight recent improvements by their absolute contribution
    total_attempts = np.sum(self.strategy_attempt_count) + 1e-10
    success_rate = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
    
    # Base score: combination of success rate and recency-weighted fitness contribution
    # Use log-scaled fitness contribution to prevent dominance by outliers
    log_fitness = np.log1p(np.maximum(self.strategy_success_fitness, 0))
    base_score = 0.7 * success_rate + 0.3 * (log_fitness / (np.max(log_fitness) + 1e-10))
    
    # Entropy bonus: boost less-used strategies when best strategy is underperforming
    # This prevents premature convergence to a single strategy
    uniform_prob = 1.0 / self.n_strategies
    current_probs = base_score / (np.sum(base_score) + 1e-10)
    entropy_bonus = uniform_prob - current_probs
    entropy_bonus = np.clip(entropy_bonus, 0, 0.3)
    
    # Compute convergence indicator: if best strategy dominates, reduce exploration pressure
    max_score = np.max(current_probs)
    convergence_factor = 1.0 - (max_score - uniform_prob) / (1.0 - uniform_prob + 1e-10)
    convergence_factor = np.clip(convergence_factor, 0.1, 1.0)
    
    # Final probabilities with entropy balancing
    probs = current_probs + entropy_bonus * convergence_factor
    probs = probs / (np.sum(probs) + 1e-10)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    probs = probs / (np.sum(probs) + 1e-10)
    
    for i in range(self.NP):
        self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```