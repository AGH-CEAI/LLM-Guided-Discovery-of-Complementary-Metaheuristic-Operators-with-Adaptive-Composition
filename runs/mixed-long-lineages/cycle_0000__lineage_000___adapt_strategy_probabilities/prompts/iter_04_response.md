**Idea: Rank-Based Softmax with Temperature Scaling**
Uses rank-transformed success with adaptive temperature softmax instead of raw proportional probabilities, targeting fine-grained optimization tasks where raw counts mislead.

```python
def _adapt_strategy_probabilities(self):
        # Compute performance score combining success count and improvement magnitude
        scores = np.zeros(self.n_strategies)
        for s in range(self.n_strategies):
            count = self.strategy_success_count[s]
            fitness_improvement = self.strategy_success_fitness[s]
            scores[s] = np.log1p(count) * (1.0 + np.log1p(fitness_improvement) + 1e-10)
        
        # Rank-based weighting: higher rank = higher weight
        ranks = np.argsort(np.argsort(scores)) + 1
        weighted_scores = scores * ranks
        
        # Adaptive temperature: lower T when variance is high (force exploitation)
        score_variance = np.var(weighted_scores) + 1e-10
        T = 0.5 / (1.0 + np.log1p(score_variance))
        
        # Softmax with temperature
        scores_shifted = weighted_scores - np.max(weighted_scores)
        exp_scores = np.exp(scores_shifted / (T + 1e-10))
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        
        # Fallback for edge cases
        if np.any(np.isnan(probs)) or np.sum(probs) < 1e-10:
            probs = np.ones(self.n_strategies) / self.n_strategies
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / np.sum(probs)
        
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```