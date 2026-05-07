**Idea: Difficulty-Adaptive Operator Selection with Progress Tracking**

This approach replaces Thompson Sampling with a deterministic selection that uses a rolling window to detect when optimization stagnates (no improvement over generations). When stagnant, it switches to exploration-focused operators; when progressing, it exploits the best-performing operator. This directly targets the priority tasks where errors of 10-50 suggest premature convergence to local optima.

```python
def _select_operator_thompson(self):
    """Difficulty-adaptive operator selection with progress tracking."""
    # Track recent best improvements across sliding window
    if not hasattr(self, 'progress_history'):
        self.progress_history = []
    
    current_improvement = max(0.0, self.f_opt_prev - self.f_opt)
    self.progress_history.append(current_improvement)
    
    # Keep window bounded
    window_size = 5 * self.dim
    if len(self.progress_history) > window_size:
        self.progress_history.pop(0)
    
    # Compute smoothed progress rate
    if len(self.progress_history) >= 3:
        recent = self.progress_history[-3:]
        avg_progress = np.mean(recent)
    else:
        avg_progress = current_improvement
    
    # Detect stagnation: no meaningful improvement in recent generations
    is_stagnant = (avg_progress < 1e-12 * max(abs(self.f_opt), 1.0)) and len(self.progress_history) >= self.dim
    
    # Compute operator quality scores from cumulative rewards
    quality_scores = np.zeros(self.num_operators)
    for op in range(self.num_operators):
        if len(self.operator_rewards[op]) > 0:
            quality_scores[op] = np.mean(self.operator_rewards[op])
    
    if is_stagnant:
        # Exploration mode: select operator with lowest count or highest variance in rewards
        if np.sum(self.operator_counts) > self.dim * 3:
            # Favor under-explored operators
            inv_counts = 1.0 / (self.operator_counts + 1.0)
            self.current_operator = int(np.argmax(inv_counts))
        else:
            # Random exploration when very stagnant
            self.current_operator = int(np.random.randint(0, self.num_operators))
    else:
        # Exploitation mode: select operator with highest quality score
        if np.max(quality_scores) > 0:
            self.current_operator = int(np.argmax(quality_scores))
        else:
            # Fallback to least-used operator
            self.current_operator = int(np.argmin(self.operator_counts))
    
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```