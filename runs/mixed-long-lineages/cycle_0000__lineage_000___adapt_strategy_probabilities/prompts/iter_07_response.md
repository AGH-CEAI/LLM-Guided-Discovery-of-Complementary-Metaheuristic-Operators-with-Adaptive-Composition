Looking at the uncovered tasks:
- **Task 12**: Already near-optimal (error 7.3e-05), all variants make it worse
- **Task 21**: Only 5% improvement ceiling, needs conservative exploitation

The core problem: current adaptation uses raw success counts which favor exploration-heavy strategies. For near-optimal or smooth landscapes, we need **conservative exploitation** with controlled exploration.

**Idea: Entropy-Regularized Softmax Selection**
A fundamentally different approach using softmax with temperature based on convergence state. When the population is near convergence, reduce exploration temperature to favor exploitation of best strategies.

```python
def _adapt_strategy_probabilities(self):
    # Compute success rates with minimum floor
    attempts = self.strategy_attempt_count + 1
    success_rate = self.strategy_success_fitness / (attempts + 1e-10)
    
    # Normalize success scores
    total = np.sum(self.strategy_success_fitness) + 1e-10
    normalized = self.strategy_success_fitness / total
    
    # Entropy-based temperature: high when uniform (explore), low when peaked (exploit)
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    max_entropy = np.log(self.n_strategies + 1e-10)
    normalized_entropy = entropy / (max_entropy + 1e-10)
    
    # Temperature: 0.1 (conservative) to 2.0 (exploratory)
    temperature = 0.1 + 1.9 * normalized_entropy
    temperature = np.clip(temperature, 0.05, 2.0)
    
    # Softmax with temperature
    weights = self.strategy_success_fitness + 1e-10
    logits = np.log(weights) / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / (np.sum(exp_logits) + 1e-10)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / np.sum(probs)
    
    for i in range(self.NP):
        self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```