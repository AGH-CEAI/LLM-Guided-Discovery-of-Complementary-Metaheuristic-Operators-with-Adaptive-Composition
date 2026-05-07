**Idea: Adaptive Epsilon-Greedy**
A classic multi-armed bandit approach with time-varying exploration rate, fundamentally different from probabilistic Beta sampling.

```python
def _select_operator_thompson(self):
    """Select operator using Adaptive Epsilon-Greedy strategy."""
    # Compute empirical mean reward for each operator
    operator_avg = np.zeros(self.num_operators)
    for i in range(self.num_operators):
        if len(self.operator_rewards[i]) > 0:
            operator_avg[i] = float(np.mean(self.operator_rewards[i]))
    
    # Adaptive epsilon: starts high (exploration), decreases over time (exploitation)
    total_selections = float(np.sum(self.operator_counts))
    epsilon = 1.0 / (1.0 + total_selections / max(self.dim, 10.0))
    epsilon = float(np.clip(epsilon, 0.01, 0.5))
    
    # Epsilon-greedy selection
    if np.random.random() < epsilon:
        # Explore: random operator with bias toward less-selected ones
        selection_probs = 1.0 / (self.operator_counts + 1.0)
        selection_probs /= np.sum(selection_probs)
        selection_probs = np.clip(selection_probs, 1e-10, 1.0 - 1e-10)
        self.current_operator = int(np.random.choice(self.num_operators, p=selection_probs))
    else:
        # Exploit: select operator with highest empirical reward
        self.current_operator = int(np.argmax(operator_avg))
    
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```