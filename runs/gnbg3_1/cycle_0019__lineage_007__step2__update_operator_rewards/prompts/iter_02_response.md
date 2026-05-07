**Idea: Rank-Based Stagnation-Escape Credit Assignment**

Replaces raw improvement magnitude with rank-based credit that explicitly rewards operators for improving population diversity and escaping stagnation — critical for the worst tasks (16-23) stuck at errors ~10-80.

```python
def _update_operator_rewards(self):
    """Rank-based credit assignment with stagnation escape detection."""
    # Compute rank-based improvement: how much did f_opt improve relative to population spread?
    pop_range = max(np.max(self.fitness) - np.min(self.fitness), 1e-10)
    rank_improvement = max(0.0, self.f_opt_prev - self.f_opt) / pop_range
    
    # Compute diversity signal (normalized)
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Stagnation escape detection: reward operators when stagnation breaks
    is_escaping = (self.stagnation_counter == 0 and 
                   hasattr(self, 'prev_stagnation') and 
                   self.prev_stagnation >= self.max_stagnation // 2)
    escape_bonus = 5.0 if is_escaping else 0.0
    self.prev_stagnation = self.stagnation_counter
    
    # Compute rank-based reward: operator's contribution to population improvement
    op = self.current_operator
    
    # Track per-operator best fitness history for rank computation
    if not hasattr(self, 'op_best_history'):
        self.op_best_history = {i: [] for i in range(self.num_operators)}
    
    # Compute how operator's sampled trials rank in the trial population
    if len(self.trial_fitness) >= self.NP // 4:
        sorted_trial_fitness = np.sort(self.trial_fitness)
        best_trial_rank = np.searchsorted(sorted_trial_fitness, self.f_opt, side='left') / len(sorted_trial_fitness)
        rank_reward = float(1.0 - best_trial_rank)  # Higher = better (closer to best trial)
    else:
        rank_reward = 0.5
    
    # Composite reward: rank contribution + diversity boost + escape bonus
    reward = float(2.0 * rank_reward + 0.5 * diversity + escape_bonus)
    reward = float(np.clip(reward, -10.0, 20.0))  # Allow higher positive rewards for escape
    
    # Sliding window with longer memory for rank-based signals
    window_size = max(self.reward_window_size, 15)
    self.operator_rewards[op].append(reward)
    if len(self.operator_rewards[op]) > window_size:
        self.operator_rewards[op].pop(0)
    
    n = len(self.operator_rewards[op])
    if n >= 3:
        mean_reward = float(np.mean(self.operator_rewards[op]))
        # Use exponential moving average for more responsive tracking
        if hasattr(self, 'ema_reward'):
            self.ema_reward[op] = 0.7 * self.ema_reward[op] + 0.3 * reward
        else:
            self.ema_reward = {i: 0.0 for i in range(self.num_operators)}
            self.ema_reward[op] = reward
        
        # Adaptive Beta parameters based on EMA and window statistics
        ema = self.ema_reward[op]
        var_reward = float(np.var(self.operator_rewards[op]))
        
        # Reward scaling: amplify differences between operators
        scaled_mean = np.clip(ema * 2.0, 0.0, 1.0)
        scaled_var = np.clip(var_reward * 4.0, 0.01, 1.0)
        
        self.alpha[op] = float(max(1.0, scaled_mean * (scaled_mean * (n - 1) / (scaled_var + 1e-10) + 1)))
        self.beta[op] = float(max(1.0, (1 - scaled_mean) * ((n - 1) * (1 - scaled_mean) / (scaled_var + 1e-10) + 1)))
    elif n >= 1:
        sum_reward = float(np.sum(self.operator_rewards[op]))
        if sum_reward > 0:
            self.alpha[op] = 1.0 + sum_reward
        else:
            self.beta[op] = 1.0 - sum_reward
```