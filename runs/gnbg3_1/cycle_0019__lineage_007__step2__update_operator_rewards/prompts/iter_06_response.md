**Idea: Max-Reward-Optimist with EMA Smoothing**

Replace sliding-window credit assignment with a max-reward tracking approach that explicitly rewards BREAKTHROUGH performance and uses EMA-smoothed maximum to avoid over-crediting lucky operators. This targets the hardest tasks (16-20) which are stuck at ~10^1 error because current rewards don't differentiate operators that occasionally make large leaps from those making consistent small gains.

```python
def _update_operator_rewards(self):
    """Update operator rewards using max-reward tracking with EMA smoothing."""
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    current_reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
    current_reward = float(np.clip(current_reward, -10.0, 10.0))
    
    op = self.current_operator
    
    # Initialize max-reward EMA tracking per operator
    if not hasattr(self, 'op_max_reward_ema'):
        self.op_max_reward_ema = np.ones(self.num_operators) * 0.5
    if not hasattr(self, 'op_max_reward_count'):
        self.op_max_reward_count = np.zeros(self.num_operators)
    
    # Update max-reward EMA: tracks the best reward level this operator achieves
    prev_max_ema = self.op_max_reward_ema[op]
    if current_reward > prev_max_ema:
        # New breakthrough: fast update toward it
        self.op_max_reward_ema[op] = 0.7 * prev_max_ema + 0.3 * current_reward
    else:
        # No breakthrough: slow decay toward current reward
        self.op_max_reward_ema[op] = 0.98 * prev_max_ema + 0.02 * current_reward
    
    # Track how many times this operator set a new max
    if current_reward >= prev_max_ema and self.op_max_reward_count[op] > 0:
        self.op_max_reward_count[op] += 1
    elif self.op_max_reward_count[op] == 0 and current_reward > 0:
        self.op_max_reward_count[op] = 1
    
    # Combined reward: current + weighted max-reward component
    # This rewards operators that achieve HIGH peak performance
    max_component = 0.5 * max(self.op_max_reward_ema[op], 0.0)
    reward = 0.6 * current_reward + 0.4 * max_component
    reward = float(np.clip(reward, -10.0, 10.0))
    
    self.operator_rewards[op].append(reward)
    
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