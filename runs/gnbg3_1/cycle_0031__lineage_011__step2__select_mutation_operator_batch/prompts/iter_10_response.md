**Idea: Entropy-Adaptive Softmax Selection with Performance Diversity**

This approach replaces UCB with a softmax-based selection driven by entropy of recent operator success rates. When all operators perform similarly (low variance), entropy rises, triggering aggressive exploration via a high temperature. When performance diverges, temperature drops to exploit the best operators. This fundamentally differs from UCB's deterministic "best + small perturbation" by using true probabilistic selection based on performance diversity, specifically targeting the stuck states causing 10^1-10^3 errors on hard tasks.

```python
def _select_mutation_operator_batch(self):
    """Select mutation operator using entropy-adaptive softmax selection."""
    # Track exponential moving average of success rates per operator
    decay = 0.9
    if not hasattr(self, 'operator_success_ema'):
        self.operator_success_ema = np.zeros(self.n_strategies)
    
    # Compute current success rates
    total_counts = np.sum(self.operator_counts) + 1e-10
    current_rates = self.operator_rewards / np.maximum(self.operator_counts, 1)
    
    # Update EMA with exponential decay (recent performance weighted higher)
    self.operator_success_ema = decay * self.operator_success_ema + (1 - decay) * current_rates
    
    # Compute variance and entropy of success rates for adaptive temperature
    rate_variance = np.var(self.operator_success_ema)
    rate_max = np.max(self.operator_success_ema)
    rate_min = np.min(self.operator_success_ema)
    
    # Entropy-based adaptive temperature:
    # Low variance -> high entropy -> high temperature (explore)
    # High variance -> low entropy -> low temperature (exploit)
    if rate_max > rate_min + 1e-10:
        normalized_rates = (self.operator_success_ema - rate_min) / (rate_max - rate_min + 1e-10)
        normalized_rates = np.clip(normalized_rates, 1e-10, 1.0)
        entropy = -np.sum(normalized_rates * np.log(normalized_rates + 1e-10))
        max_entropy = np.log(self.n_strategies + 1e-10)
        entropy_factor = max(0.1, entropy / (max_entropy + 1e-10))
    else:
        entropy_factor = 1.0
    
    # Temperature inversely proportional to variance but boosted by entropy
    temperature = max(0.1, 2.0 * entropy_factor / (rate_variance * 10 + 0.1))
    
    # Softmax probabilities
    scaled = self.operator_success_ema / (temperature + 1e-10)
    scaled = scaled - np.max(scaled)  # Numerical stability
    exp_rates = np.exp(scaled)
    probs = exp_rates / (np.sum(exp_rates) + 1e-10)
    
    # Ensure valid probabilities
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / (np.sum(probs) + 1e-10)
    
    # Probabilistic selection (fundamentally different from UCB's argmax)
    selected = np.random.choice(self.n_strategies, p=probs)
    
    return selected
```