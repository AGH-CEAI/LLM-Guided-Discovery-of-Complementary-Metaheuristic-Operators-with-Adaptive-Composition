**Idea: Trend-Aware Multi-Signal Reward with Landscape Analysis**

Replace immediate-improvement rewards with trend-based learning using fitness landscape analysis (variance, rank entropy) and exponential moving averages of improvement rates. This rewards operators that consistently improve over time rather than those with occasional lucky gains—critical for the hardest multimodal tasks (16, 17, 20, 23) where operators must sustain progress across many generations.

```python
def _update_operator_rewards(self):
    """Update operator rewards using trend-aware multi-signal credit assignment with landscape analysis."""
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    
    if not hasattr(self, 'ema_improvement'):
        self.ema_improvement = 0.0
        self.ema_reward = 0.0
        self.op_trend_count = {i: 0 for i in range(self.num_operators)}
        self.op_last_reward = {i: 0.0 for i in range(self.num_operators)}
    
    # Trend-based improvement signal: smooth improvement rate
    self.ema_improvement = 0.7 * self.ema_improvement + 0.3 * improvement
    trend_signal = np.log1p(max(self.ema_improvement * 1e10, 1e-15))
    
    # Fitness landscape analysis
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Rank entropy for ruggedness detection
    ranks = np.argsort(np.argsort(self.fitness))
    rank_probs = (ranks + 1) / (self.NP + 1)
    rank_probs = np.clip(rank_probs, 1e-10, 1.0)
    rank_entropy = -np.mean(rank_probs * np.log(rank_probs))
    rank_entropy /= np.log(self.NP + 1) + 1e-10
    
    # Consistency signal: reward operators that improve steadily
    op = self.current_operator
    if improvement > 1e-15:
        if self.op_last_reward[op] > 0.0 or self.op_trend_count[op] > 0:
            self.op_trend_count[op] += 1
        else:
            self.op_trend_count[op] = 1
    else:
        self.op_trend_count[op] = max(0, self.op_trend_count[op] - 1)
    self.op_last_reward[op] = improvement
    
    consistency = np.tanh(self.op_trend_count[op] / 5.0)
    
    # Multi-signal reward combination
    base_reward = trend_signal / 10.0
    diversity_reward = 0.15 * diversity
    ruggedness_penalty = 0.05 * (1.0 - rank_entropy)
    consistency_reward = 0.3 * consistency
    
    reward = base_reward + diversity_reward + ruggedness_penalty + consistency_reward
    reward = float(np.clip(reward, -10.0, 10.0))
    
    # Exponential moving average of reward for trend tracking
    self.ema_reward = 0.9 * self.ema_reward + 0.1 * reward
    
    self.operator_rewards[op].append(reward)
    
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)
    
    n = len(self.operator_rewards[op])
    if n >= 3:
        mean_reward = float(np.mean(self.operator_rewards[op]))
        ema_reward = float(self.ema_reward)
        combined = 0.6 * mean_reward + 0.4 * ema_reward
        
        var_reward = float(np.var(self.operator_rewards[op]))
        denom = max(var_reward * n + 1e-10, 1e-10)
        
        self.alpha[op] = float(max(1.0, combined * (combined * (n - 1) / denom + 1)))
        self.beta[op] = float(max(1.0, (1 - combined) * ((n - 1) * (1 - combined) / denom + 1)))
    elif n >= 1:
        sum_reward = float(np.sum(self.operator_rewards[op]))
        if sum_reward > 0:
            self.alpha[op] = 1.0 + sum_reward
        else:
            self.beta[op] = 1.0 - sum_reward
```