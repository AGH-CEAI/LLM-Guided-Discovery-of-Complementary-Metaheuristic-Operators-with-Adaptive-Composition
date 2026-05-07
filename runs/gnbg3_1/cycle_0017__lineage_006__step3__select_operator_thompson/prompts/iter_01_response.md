**Idea: UCB-Based Operator Selection**
Use Upper Confidence Bound selection with optimistic initialization to systematically balance exploration vs exploitation, providing a fundamentally different uncertainty-driven approach vs Beta sampling.
```python
def _select_operator_thompson(self):
    """Select operator using Upper Confidence Bound (UCB) with optimistic initialization."""
    total_selections = np.sum(self.selection_counts)
    
    # Initialize with small epsilon reward if no data
    if total_selections == 0:
        self.current_operator = int(np.random.randint(0, self.num_operators))
    else:
        # Compute mean rewards for each operator
        mean_rewards = np.zeros(self.num_operators)
        valid_ops = np.zeros(self.num_operators, dtype=bool)
        
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) > 0:
                mean_rewards[i] = np.mean(self.operator_rewards[i])
                valid_ops[i] = True
        
        # UCB exploration bonus: sqrt(2 * ln(total) / count)
        exploration = np.zeros(self.num_operators)
        nonzero_mask = self.selection_counts > 0
        exploration[nonzero_mask] = np.sqrt(
            2.0 * np.log(max(total_selections, 1)) / np.maximum(self.selection_counts[nonzero_mask], 1)
        )
        
        # Optimistic initialization: small boost to operators with no data
        exploration[~nonzero_mask] = np.sqrt(2.0 * np.log(max(total_selections, 1)) + 1)
        
        # UCB score = mean reward + exploration bonus
        ucb_scores = mean_rewards + exploration
        
        # If some operators have no data, prioritize them initially
        if np.any(~valid_ops):
            unvisited = np.where(~valid_ops)[0]
            ucb_scores[unvisited] += np.max(ucb_scores[valid_ops]) + 1.0
        
        # Select operator with highest UCB score
        self.current_operator = int(np.argmax(ucb_scores))
    
    self.selection_counts[self.current_operator] += 1
    return self.current_operator
```