**Idea: Exponentially-Weighted Cumulative Improvement with Stagnation Detection**

Replaces sliding-window credit assignment with cumulative exponential decay tracking and explicit stagnation detection to better reward sustained improvement on hard deceptive tasks (Tasks 16-12 with errors ~8-10 decades above target).
```python
def _update_operator_rewards(self):
    """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    
    # Initialize cumulative tracking if needed
    if not hasattr(self, 'operator_cumulative_reward'):
        self.operator_cumulative_reward = np.zeros(self.num_operators)
    if not hasattr(self, 'operator_decay_sum'):
        self.operator_decay_sum = np.zeros(self.num_operators)
    
    # Apply exponential decay to accumulated reward (recent rewards weighted more)
    decay = 0.95
    self.operator_cumulative_reward *= decay
    self.operator_decay_sum *= decay
    
    # Log-scaled improvement reward (more sensitive to small improvements)
    if improvement > 0:
        reward = float(np.log1p(improvement * 1e10) / 10.0)
    else:
        reward = 0.0
    
    # Explicit stagnation detection
    is_stagnant = self.stagnation_counter > self.max_stagnation // 2
    
    # Diversity metric
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Exploration bonus when stagnant and diversity is low
    if is_stagnant and diversity < 0.1:
        reward *= 2.0
    
    # Update cumulative weighted reward
    self.operator_cumulative_reward[self.current_operator] += reward
    self.operator_decay_sum[self.current_operator] += 1.0
    
    # Normalize by decay sum to get comparable reward values
    norm = max(self.operator_decay_sum[self.current_operator], 1.0)
    normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
    normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))
    
    op = self.current_operator
    self.operator_rewards[op].append(normalized_reward)
    
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)
    
    n = len(self.operator_rewards[op])
    if n >= 3:
        mean_reward = float(np.mean(self.operator_rewards[op]))
        var_reward = float(np.var(self.operator_rewards[op]))
        denom = max(var_reward * n + 1e-10, 1e-10)
        self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
        self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
    elif n >= 1:
        sum_reward = float(np.sum(self.operator_rewards[op]))
        if sum_reward > 0:
            self.alpha[op] = 1.0 + sum_reward
        else:
            self.beta[op] = 1.0 - sum_reward
```