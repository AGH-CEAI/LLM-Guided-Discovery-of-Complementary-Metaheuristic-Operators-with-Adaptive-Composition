Looking at the problem: all 24 tasks have errors between 1e-04 and 8e+01, with the worst tasks (16, 17, 23, 20, 19) having errors in the 10-80 range. The current Thompson Sampling with Beta distributions appears too conservative for these severely stuck cases.

**Analysis of the failure mode:**
- Errors in the 10-80 range suggest the algorithm is trapped in local optima
- The current Beta-based Thompson Sampling samples from a distribution that can repeatedly select the same "mediocre but not terrible" operator
- We need an operator selection mechanism that **forces exploration** when stuck

**Fundamentally different strategy:** Replace probabilistic Thompson Sampling with **softmax selection using recency-weighted exploration**. When the algorithm is stagnating, force selection toward operators that haven't been used recently. This directly attacks the trapping problem by ensuring all operators get tried.

**Idea: Recency-Weighted Softmax with Forced Exploration**

```python
def _select_operator_thompson(self):
    """Select operator using recency-weighted softmax with forced exploration during stagnation."""
    # Compute average reward per operator
    recent_rewards = np.zeros(self.num_operators)
    for i in range(self.num_operators):
        if len(self.operator_rewards[i]) >= 3:
            recent_rewards[i] = np.mean(self.operator_rewards[i])
        elif len(self.operator_rewards[i]) > 0:
            recent_rewards[i] = np.mean(self.operator_rewards[i]) * 0.5
    
    # Compute time since last selection for each operator
    if not hasattr(self, 'last_selection_gen'):
        self.last_selection_gen = np.zeros(self.num_operators, dtype=int)
    
    generations_since_selection = self.generation - self.last_selection_gen
    max_gens = max(generations_since_selection.max(), 1)
    recency_scores = generations_since_selection / max_gens
    
    # Adaptive temperature based on stagnation
    if self.stagnation_counter > self.max_stagnation // 2:
        temperature = 5.0  # Very high exploration
        exploration_weight = 0.8  # Favor recency heavily
    elif self.stagnation_counter > self.max_stagnation // 4:
        temperature = 2.0
        exploration_weight = 0.5
    else:
        temperature = 0.5
        exploration_weight = 0.2
    
    # Combine reward-based and recency-based scores
    reward_normalized = (recent_rewards - recent_rewards.min() + 1e-10) / (recent_rewards.max() - recent_rewards.min() + 1e-10)
    combined_scores = (1.0 - exploration_weight) * reward_normalized + exploration_weight * recency_scores
    
    # Add small noise to break ties
    combined_scores += np.random.randn(self.num_operators) * 0.01
    
    # Softmax selection
    exp_scores = np.exp((combined_scores - combined_scores.max()) / temperature)
    probs = exp_scores / np.sum(exp_scores)
    
    # Ensure valid probabilities
    probs = np.clip(probs, 1e-10, 1.0)
    probs /= np.sum(probs)
    
    # Sample from distribution
    self.current_operator = int(np.random.choice(self.num_operators, p=probs))
    self.operator_counts[self.current_operator] += 1
    self.last_selection_gen[self.current_operator] = self.generation
    
    return self.current_operator
```