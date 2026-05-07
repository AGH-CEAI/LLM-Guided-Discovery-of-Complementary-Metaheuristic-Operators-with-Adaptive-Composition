**Idea: Immediate Sigmoid Reward with EMA**

A fundamentally different approach using immediate bounded sigmoid rewards, exponential moving average (EMA) tracking, and standard Beta distribution updates without sliding window complexity. This provides scale-invariant, numerically robust credit assignment.

```python
def _update_operator_rewards(self):
    """Update operator rewards using immediate sigmoid reward with EMA tracking."""
    # Compute signed improvement (negative = worsening)
    improvement = self.f_opt_prev - self.f_opt
    
    # Scale-invariant relative improvement
    scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
    relative_improvement = improvement / scale
    
    # Bounded sigmoid reward in [-1, 1] - scale-invariant and numerically stable
    reward = 2.0 / (1.0 + np.exp(-5.0 * np.clip(relative_improvement, -3.0, 3.0))) - 1.0
    reward = float(np.clip(reward, -1.0, 1.0))
    
    op = self.current_operator
    
    # Initialize EMA state for each operator if needed
    if not hasattr(self, 'reward_ema'):
        self.reward_ema = np.zeros(self.num_operators)
        self.reward_count = np.zeros(self.num_operators)
    
    # Exponential moving average update (EMA alpha = 0.3)
    if self.reward_count[op] == 0:
        self.reward_ema[op] = reward
    else:
        self.ema_alpha = 0.3
        self.reward_ema[op] = (1.0 - self.ema_alpha) * self.reward_ema[op] + self.ema_alpha * reward
    
    self.reward_count[op] += 1
    
    # Standard Beta distribution update with bounded reward
    # Maps reward from [-1, 1] to [0, 1] for Beta parameters
    reward_mapped = (reward + 1.0) / 2.0
    reward_mapped = float(np.clip(reward_mapped, 1e-10, 1.0 - 1e-10))
    
    # Update Beta distribution parameters (Thompson Sampling)
    self.alpha[op] += reward_mapped
    self.beta[op] += (1.0 - reward_mapped)
    
    # Keep parameters bounded to prevent numerical issues
    self.alpha[op] = float(np.clip(self.alpha[op], 1e-10, 1e6))
    self.beta[op] = float(np.clip(self.beta[op], 1e-10, 1e6))
```