Looking at the task coverage, all 22 non-trivial tasks are already covered, but I should still propose a fundamentally different approach to potentially improve performance further, especially on tasks where improvements are marginal (Task 21 at -5%, Task 15 at -19%, Task 14 at -30%).

**Idea: Entropy-Regularized Softmax with Diversity Floor**
Uses softmax temperature scaling with an entropy floor to maintain exploration, preventing premature strategy concentration.

```python
def _adapt_strategy_probabilities(self):
    # Compute success rates per strategy
    rates = self.strategy_success_fitness / (self.strategy_attempt_count + 1e-10)
    rates = np.clip(rates, 1e-10, None)
    
    # Softmax with temperature: high temp = more exploration
    temperature = 0.5 + 0.5 * (1.0 - np.max(rates) / (np.mean(rates) + 1e-10))
    temperature = np.clip(temperature, 0.1, 5.0)
    exp_rates = np.exp(rates / temperature)
    base_probs = exp_rates / np.sum(exp_rates)
    
    # Entropy floor: if entropy too low, boost rare strategies
    entropy = -np.sum(base_probs * np.log(base_probs + 1e-10))
    max_entropy = np.log(self.n_strategies)
    entropy_ratio = entropy / (max_entropy + 1e-10)
    
    if entropy_ratio < 0.5:
        # Boost underrepresented strategies by log-inverse probability
        boost = np.log(1.0 / (base_probs + 1e-10))
        boost = boost / np.sum(boost)
        blend = 0.3
        probs = (1.0 - blend) * base_probs + blend * boost
    else:
        probs = base_probs
    
    # Ensure valid probability distribution
    probs = np.clip(probs, 1e-10, None)
    probs = probs / np.sum(probs)
    
    for i in range(self.NP):
        self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```