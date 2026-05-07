**Idea: Fitness-Weighted Success Rate with Temporal Forgetting**

Uses a generation-dependent adaptation factor that blends success RATE (for reliability) with cumulative success COUNT (for impact), with exponential weighting to amplify high-performing strategies and temporal forgetting to track recent performance trends.

```python
def _adapt_strategy_probabilities(self):
        # Compute success rate per strategy
        rates = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
        
        # Blend rate (reliability) with count (impact), weighted by generation
        # Early: favor reliability, Late: favor impact
        gen = len(self.diversity_history) + 1
        alpha = min(0.9, 0.3 + 0.005 * gen)  # 0.30 -> 0.90 over generations
        
        # Exponential weighting to amplify differences; log1p for numerical stability
        weighted_scores = alpha * self.strategy_success_count + (1 - alpha) * np.log1p(self.strategy_success_count)
        
        # Blend with success rate for balanced selection
        scores = alpha * rates + (1 - alpha) * weighted_scores
        
        # Ensure non-negative and numerically robust
        scores = np.maximum(scores, 1e-15)
        
        # Normalize to probabilities
        total = np.sum(scores)
        probs = scores / (total + 1e-10)
        
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```