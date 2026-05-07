**Idea: Task-Difficulty-Aware Thompson Sampling with Performance-Shaped Rewards**

Instead of uniform Beta prior sampling, this approach dynamically shapes reward signals based on per-task convergence rates, uses success-rate-normalized rewards for fair operator comparison, and increases exploration probability specifically when facing the hardest tasks (detected by stagnation or large best errors). This targets the worst unsolved tasks (errors ~10-50) that likely involve deceptive basins requiring diverse exploration.

```python
def _select_operator_thompson(self):
    """Select operator using task-difficulty-aware Thompson Sampling."""
    # Track per-operator recent performance
    if not hasattr(self, 'op_reward_buffer'):
        self.op_reward_buffer = [[] for _ in range(self.num_operators)]
    
    # Compute current performance metrics
    current_error = abs(self.f_opt)
    log_error = np.log10(max(current_error, 1e-15))
    target_log = -8.0
    error_gap = max(0.0, log_error - target_log)
    
    # Stagnation detection
    is_stagnant = self.stagnation_counter > self.max_stagnation // 3
    
    # Compute task difficulty proxy (0=hard, 1=easy)
    if error_gap > 8.0:
        difficulty = 0.0  # Very hard
    elif error_gap > 4.0:
        difficulty = 0.25
    elif error_gap > 1.0:
        difficulty = 0.5
    else:
        difficulty = 0.75
    
    # Success rate for current operator
    recent_wins = 0
    recent_total = 0
    for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
        if self.trial_fitness[i] < self.fitness[i]:
            recent_wins += 1
        recent_total += 1
    success_rate = recent_wins / max(recent_total, 1)
    
    # Shape reward based on improvement and difficulty
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    if improvement > 0:
        base_reward = np.log1p(improvement * 1e10) / 10.0
    else:
        base_reward = -0.5 if is_stagnant else 0.0
    
    # Difficulty-adjusted reward: boost exploration on hard tasks
    if difficulty < 0.5:
        exploration_bonus = 2.0 * (0.5 - difficulty)
        base_reward += exploration_bonus
    
    # Normalize by operator selection count to prevent bias toward frequent operators
    op_count = max(self.operator_counts[self.current_operator], 1)
    normalized_reward = base_reward / np.sqrt(op_count)
    normalized_reward = np.clip(normalized_reward, -5.0, 5.0)
    
    # Update reward buffer
    self.op_reward_buffer[self.current_operator].append(normalized_reward)
    buffer_size = 15
    if len(self.op_reward_buffer[self.current_operator]) > buffer_size:
        self.op_reward_buffer[self.current_operator].pop(0)
    
    # Compute per-operator statistics
    op_means = np.zeros(self.num_operators)
    op_vars = np.zeros(self.num_operators)
    op_counts_snapshot = self.operator_counts.copy()
    
    for op in range(self.num_operators):
        if len(self.op_reward_buffer[op]) >= 3:
            op_means[op] = np.mean(self.op_reward_buffer[op])
            op_vars[op] = np.var(self.op_reward_buffer[op]) + 1e-10
        else:
            op_means[op] = 0.0
            op_vars[op] = 1.0
    
    # Thompson Sampling with performance-shaped Beta parameters
    # Higher mean reward -> higher alpha (more likely to be selected)
    # Lower variance -> more confident estimate
    alpha_samples = np.zeros(self.num_operators)
    beta_samples = np.zeros(self.num_operators)
    
    for op in range(self.num_operators):
        count = max(op_counts_snapshot[op], 1)
        confidence = 1.0 / np.sqrt(count)
        
        # Beta parameters shaped by performance
        alpha_base = max(1.0, 1.0 + op_means[op] * 5.0)
        beta_base = max(1.0, 1.0 - op_means[op] * 5.0)
        
        # Adjust by confidence (higher confidence = tighter distribution)
        alpha_param = alpha_base * (1.0 + confidence)
        beta_param = beta_base * (1.0 + confidence)
        
        alpha_samples[op] = alpha_param
        beta_samples[op] = beta_param
    
    # Sample from Beta distributions
    samples = np.array([np.random.beta(alpha_samples[op], beta_samples[op]) 
                        for op in range(self.num_operators)])
    
    # Exploration boost for hard tasks: occasionally sample uniformly
    if difficulty < 0.5 or is_stagnant:
        explore_prob = 0.3 if difficulty < 0.25 else 0.15
        if np.random.random() < explore_prob:
            samples = np.random.uniform(size=self.num_operators)
    
    self.current_operator = int(np.argmax(samples))
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```