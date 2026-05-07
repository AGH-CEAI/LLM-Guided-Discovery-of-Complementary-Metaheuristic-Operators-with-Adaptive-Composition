Looking at the unsolved tasks, the worst performers (Tasks 16-23) are stuck at errors of 10-52, suggesting they're trapped in severe local optima or have pathological landscapes. The key insight: **the current reward system treats all tasks equally** - an improvement of 0.1 when stuck at error=50 is treated the same as when stuck at error=0.01. For the hardest tasks, we need **task-difficulty-weighted credit assignment** that amplifies rewards when making progress on already-bad fitness values.

**Idea: Inverse-Error Task-Difficulty Weighting**
Rewards are weighted by how bad the current error is, so operators that make progress on the worst tasks get much higher credit than operators that make marginal gains on already-converged tasks.

```python
def _update_operator_rewards(self):
        """Update operator rewards using inverse-error task-difficulty weighting.
        
        Amplifies rewards for improvements on tasks with large errors, so operators
        that escape severe local optima receive substantially higher credit.
        """
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Task difficulty weighting: amplify credit when stuck at high error
        # Scale factor grows with absolute fitness magnitude
        error_magnitude = max(abs(self.f_opt), 1.0)
        difficulty_weight = np.log1p(error_magnitude) + 1.0
        difficulty_weight = float(np.clip(difficulty_weight, 1.0, 100.0))
        
        # Diversity component
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Base reward scaled by task difficulty
        base_reward = float(np.log1p(improvement * 1e10) / 10.0)
        reward = base_reward * difficulty_weight + 0.1 * diversity
        reward = float(np.clip(reward, -20.0, 20.0))
        
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