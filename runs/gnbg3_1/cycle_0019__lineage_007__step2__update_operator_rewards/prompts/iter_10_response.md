**Idea: Multi-Signal Reward with Trajectory Tracking**
Replace single-signal reward with multi-objective reward combining immediate improvement, EMA of improvement rate, convergence status, and stagnation penalty. This targets the worst tasks (16-23 with errors 10-50) which appear stuck in deceptive basins where immediate improvement is unreliable.

```python
def _update_operator_rewards(self):
        """Update operator rewards using multi-signal trajectory tracking."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Signal 1: Immediate improvement (log-scaled, clamped)
        reward_immediate = float(np.log1p(improvement * 1e10) / 10.0)
        reward_immediate = np.clip(reward_immediate, -10.0, 10.0)
        
        # Signal 2: Exponential moving average of improvement rate
        if not hasattr(self, 'reward_ema'):
            self.reward_ema = 0.0
        self.reward_ema = 0.7 * self.reward_ema + 0.3 * reward_immediate
        
        # Signal 3: Cumulative improvement over sliding window
        if not hasattr(self, 'reward_cumsum'):
            self.reward_cumsum = 0.0
        self.reward_cumsum += improvement
        if not hasattr(self, 'cumsum_buffer'):
            self.cumsum_buffer = 0.0
        self.cumsum_buffer += 1
        if self.cumsum_buffer > 20:
            self.reward_cumsum *= 0.95
            self.cumsum_buffer *= 0.95
        
        # Signal 4: Convergence status bonus/penalty
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Stagnation detection
        if not hasattr(self, 'stagnation_for_reward'):
            self.stagnation_for_reward = 0
        if improvement > 1e-15:
            self.stagnation_for_reward = 0
        else:
            self.stagnation_for_reward += 1
        
        # Multi-signal combination: immediate (30%) + EMA (40%) + diversity (15%) + cumsum (15%)
        reward = (0.30 * reward_immediate + 
                  0.40 * self.reward_ema + 
                  0.15 * diversity + 
                  0.15 * np.clip(np.log1p(self.reward_cumsum * 1e5) / 10.0, -5.0, 5.0))
        
        # Stagnation penalty: reduce reward when stuck
        if self.stagnation_for_reward > 10:
            stagnation_factor = max(0.0, 1.0 - (self.stagnation_for_reward - 10) / 50.0)
            reward *= stagnation_factor
        
        # Exploitation bonus when converging well
        if self.stagnation_for_reward == 0 and self.reward_ema > 1.0:
            reward += 0.5
        
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
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