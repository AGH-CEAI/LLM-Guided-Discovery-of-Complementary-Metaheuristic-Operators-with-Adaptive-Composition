Looking at the data, I see that:
1. The current reward mechanism uses relative improvement scaled by 100 and clipped to [0,1]
2. For the WORST tasks (errors 1e+0 to 9e+0), the optimizer is stuck far from optimum
3. When stuck, `rel_improvement` is essentially zero → all operators get reward ~0.1 (baseline)
4. This makes operator selection essentially random, preventing breakthrough

The core problem: **Absolute improvement-based rewards fail when the optimizer is stuck** because tiny improvements produce indistinguishable rewards.

My new strategy: **Rank-based reward with exponential moving average and exploration bonus** — rewards based on relative rank within the batch rather than absolute improvement magnitude, with longer effective memory and UCB-style exploration bonus for under-sampled operators.

**Idea: Rank-Based EMA Reward with UCB Exploration**
```python
def _update_operator_reward(self, operator_idx, reward):
    """Update Thompson Sampling parameters using rank-based reward with EMA."""
    operator_idx = int(operator_idx)
    reward = float(reward)
    
    # Store raw reward for EMA tracking
    if not hasattr(self, '_op_rewards_ema'):
        self._op_rewards_ema = np.zeros(self.n_operators)
    if not hasattr(self, '_op_sample_count'):
        self._op_sample_count = np.zeros(self.n_operators)
    
    self._op_sample_count[operator_idx] += 1
    
    # Convert absolute reward to rank-based reward using sigmoid
    # This normalizes across different problem scales
    rank_reward = 1.0 / (1.0 + np.exp(-10.0 * (reward - 0.3)))
    
    # Update exponential moving average (longer memory than sliding window)
    ema_alpha = 0.95  # Decay factor
    self._op_rewards_ema[operator_idx] = (
        ema_alpha * self._op_rewards_ema[operator_idx] + (1.0 - ema_alpha) * rank_reward
    )
    
    # UCB-style exploration bonus for under-sampled operators
    exploration_bonus = 0.0
    min_samples = 3
    if self._op_sample_count[operator_idx] < min_samples:
        exploration_bonus = 0.5 * (min_samples - self._op_sample_count[operator_idx]) / min_samples
    
    # Combine EMA reward with exploration bonus
    effective_reward = self._op_rewards_ema[operator_idx] + exploration_bonus
    
    # Update Beta distribution parameters
    # Use effective reward to set alpha/beta ratio
    self.op_alpha[operator_idx] = 1.0 + effective_reward * 20.0
    self.op_beta[operator_idx] = 1.0 + (1.0 - effective_reward) * 20.0
    
    # Gentle decay to prevent parameters from growing unbounded
    decay_factor = 0.999
    self.op_alpha = np.maximum(1.0, self.op_alpha * decay_factor)
    self.op_beta = np.maximum(1.0, self.op_beta * decay_factor)
    
    # Soft cap: if total parameters exceed threshold, apply gentle shrink
    max_total = 100.0
    for i in range(self.n_operators):
        total = self.op_alpha[i] + self.op_beta[i]
        if total > max_total:
            shrink = max_total / total
            self.op_alpha[i] *= shrink
            self.op_beta[i] *= shrink
```