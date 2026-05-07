**Idea: UCB-Inspired Success-Rate with Immediate Credit**

Uses immediate binary success signals and UCB-style exploration bonuses instead of log-scaled decaying rewards, directly updating Thompson Sampling parameters from empirical success rates.

```python
def _update_operator_rewards(self):
    """UCB-inspired success rate tracking with immediate credit assignment."""
    # Immediate improvement signal
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    success = 1.0 if improvement > 0 else 0.0

    # Initialize tracking if needed
    if not hasattr(self, 'operator_successes'):
        self.operator_successes = np.zeros(self.num_operators)
    if not hasattr(self, 'operator_trials'):
        self.operator_trials = np.zeros(self.num_operators)
    if not hasattr(self, 'operator_ema_reward'):
        self.operator_ema_reward = np.zeros(self.num_operators)

    op = self.current_operator
    self.operator_trials[op] += 1.0

    # Update EMA reward for current operator
    self.operator_ema_reward[op] = 0.9 * self.operator_ema_reward[op] + 0.1 * success

    # Credit assignment: immediate success signal
    if success > 0:
        self.operator_successes[op] += success

    # Compute UCB-style exploration bonus for all operators
    ucb_scores = np.zeros(self.num_operators)
    for i in range(self.num_operators):
        trials_i = max(self.operator_trials[i], 1.0)
        success_rate = self.operator_successes[i] / trials_i
        # UCB bonus: higher for less-explored operators
        exploration_bonus = np.sqrt(2.0 * np.log(max(self.generation + 1, 1)) / trials_i)
        ucb_scores[i] = success_rate + exploration_bonus

    # Scale exploration bonus by current exploration temperature
    exploration_temp = getattr(self, 'exploration_temp', 1.0)
    ucb_bonus_scaled = ucb_scores[op] - (self.operator_successes[op] / max(self.operator_trials[op], 1.0))
    ucb_bonus_scaled *= exploration_temp

    # Compute reward for current operator
    reward = float(np.clip(self.operator_ema_reward[op] + ucb_bonus_scaled, 0.0, 1.0))

    # Update operator rewards sliding window
    self.operator_rewards.setdefault(op, [])
    self.operator_rewards[op].append(reward)
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)

    # Direct Beta distribution update from empirical success rate
    total_trials = max(self.operator_trials[op], 1.0)
    empirical_success_rate = self.operator_successes[op] / total_trials

    # Combine EMA reward with current success for robust estimate
    combined_rate = 0.7 * empirical_success_rate + 0.3 * self.operator_ema_reward[op]
    combined_rate = float(np.clip(combined_rate, 0.0, 1.0))

    # Update Beta parameters with Laplace smoothing
    alpha_prior = 1.0
    beta_prior = 1.0
    self.alpha[op] = alpha_prior + self.operator_successes[op]
    self.beta[op] = beta_prior + (self.operator_trials[op] - self.operator_successes[op])

    # Apply reward-based adjustment to Beta parameters
    reward_adjustment = reward * 5.0
    self.alpha[op] = max(1.0, self.alpha[op] * (1.0 + reward_adjustment))
    self.beta[op] = max(1.0, self.beta[op] * (1.0 + (1.0 - reward) * 2.0))

    # Ensure alpha + beta stays bounded for numerical stability
    total_ab = self.alpha[op] + self.beta[op]
    if total_ab > 1000.0:
        scale = 1000.0 / total_ab
        self.alpha[op] *= scale
        self.beta[op] *= scale
```