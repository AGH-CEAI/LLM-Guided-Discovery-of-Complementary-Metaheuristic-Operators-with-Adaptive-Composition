**Idea: Entropy-Regularized Strategy Selection**

A fundamentally different approach using Shannon entropy as an exploration bonus. When all strategies perform poorly (low success fitness), the entropy term promotes diversity by boosting underutilized strategies. This addresses the likely convergence issue on Task 21 where improvements are marginal (~5%), suggesting premature convergence to local optima.

```python
def _adapt_strategy_probabilities(self):
    base_probs = self.strategy_success_fitness / (np.sum(self.strategy_success_fitness) + 1e-10)
    base_probs = np.clip(base_probs, 1e-10, 1.0)
    base_probs = base_probs / (np.sum(base_probs) + 1e-10)
    
    entropy = -np.sum(base_probs * np.log(base_probs + 1e-10))
    max_entropy = np.log(self.n_strategies + 1e-10)
    normalized_entropy = entropy / (max_entropy + 1e-10)
    
    exploration_bonus = normalized_entropy * 0.5
    exploration_factor = np.ones(self.n_strategies) * 0.5 + exploration_bonus
    
    combined = base_probs * exploration_factor
    combined = combined / (np.sum(combined) + 1e-10)
    combined = np.clip(combined, 1e-10, 1.0)
    combined = combined / (np.sum(combined) + 1e-10)
    
    for i in range(self.NP):
        self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=combined)
```