Looking at the priority tasks (16, 17, 23, 20, 19, 18) with errors of 10-50+, these represent extremely challenging optimization landscapes. The current Thompson Sampling with Beta distributions is too conservative and doesn't adapt quickly enough when operators perform poorly.

**Idea: Adaptive Ensemble Selection with Performance Scaled Rewards**
Use exponentially-weighted moving average of per-operator reward with softmax-based selection and explicit exploration bonus for operators not recently selected, designed to aggressively discover which operator works best for each difficult task.

```python
def _select_operator_thompson(self):
    """Select operator using adaptive ensemble with performance-scaled softmax selection."""
    if not hasattr(self, 'op_scores'):
        self.op_scores = np.zeros(self.num_operators)
    if not hasattr(self, 'op_selections'):
        self.op_selections = np.zeros(self.num_operators)
    if not hasattr(self, 'op_last_selected'):
        self.op_last_selected = np.zeros(self.num_operators, dtype=int)
    
    # Compute per-operator reward from recent fitness improvement
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    
    # Log-scaled reward with performance scaling
    if improvement > 1e-15:
        reward = float(np.log1p(improvement * 1e10) / 10.0)
    else:
        reward = 0.0
    
    # Apply performance-scaled update
    decay = 0.9
    self.op_scores = decay * self.op_scores + (1.0 - decay) * reward
    
    # Exploration bonus: reward operators not recently selected
    current_gen = self.generation
    for i in range(self.num_operators):
        generations_since = current_gen - self.op_last_selected[i]
        if generations_since > 5:
            self.op_scores[i] += 0.5 * (generations_since - 5) / 10.0
    
    # Diversity bonus when stagnant
    if self.stagnation_counter > self.max_stagnation // 3:
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        if diversity < 0.05:
            for i in range(self.num_operators):
                if self.selection_counts[i] < np.mean(self.selection_counts) * 0.5:
                    self.op_scores[i] += 1.0
    
    # Softmax selection with temperature
    scores = self.op_scores - np.max(self.op_scores) + 1e-10
    temperature = 0.5
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores / np.sum(exp_scores)
    
    # Sample from categorical distribution
    cumsum = np.cumsum(probs)
    r = np.random.random()
    self.current_operator = int(np.searchsorted(cumsum, r))
    self.current_operator = min(self.current_operator, self.num_operators - 1)
    
    self.operator_counts[self.current_operator] += 1
    self.op_selections[self.current_operator] += 1
    self.op_last_selected[self.current_operator] = current_gen
    
    return self.current_operator
```