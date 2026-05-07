**Idea: Elite-Relative Progress Reward**

Instead of only crediting operators when the global best improves, this variant rewards operators based on how many trial individuals beat the current best (elite-relative progress), which provides dense reward signals even when the global best is stuck. It also uses adaptive scaling based on task difficulty and applies a "breakthrough bonus" when any operator achieves a new best, addressing the sparse reward problem on the hardest tasks.

```python
def _update_operator_rewards(self):
    """Credit assignment via elite-relative progress and adaptive scaling for hard tasks."""
    # Initialize tracking if needed
    if not hasattr(self, 'operator_cumulative_reward'):
        self.operator_cumulative_reward = np.zeros(self.num_operators)
    if not hasattr(self, 'operator_decay_sum'):
        self.operator_decay_sum = np.zeros(self.num_operators)

    # Apply exponential decay
    decay = 0.9
    self.operator_cumulative_reward *= decay
    self.operator_decay_sum *= decay

    # Elite-relative progress: count how many trials beat current best
    elite = self.f_opt
    trial_array = np.asarray(self.trial_fitness).flatten()
    n_trials = len(trial_array)
    if n_trials > 0:
        beaten = np.sum(trial_array < elite)
        elite_progress = beaten / n_trials
    else:
        elite_progress = 0.0

    # Log-scaled improvement reward (crushes large errors better)
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    if improvement > 0:
        base_reward = float(np.log1p(improvement * 1e10) / 10.0)
    else:
        base_reward = 0.0

    # Combine: 70% elite-relative progress + 30% absolute improvement
    # This provides dense signals even on hard tasks with rare global improvements
    reward = 0.7 * elite_progress * 5.0 + 0.3 * base_reward

    # Task-difficulty-adaptive scaling: harder tasks get exploration boost
    # The worst tasks have errors ~1e+1 to 5e+1, so scale rewards accordingly
    task_difficulty = max(1.0, np.log10(max(abs(self.f_opt), 1e-3) + 1.0))
    difficulty_scale = 1.0 + 0.5 * min(task_difficulty, 5.0)
    reward *= difficulty_scale

    # Stagnation bonus: when stuck, reward operators that explore diverse regions
    is_stagnant = self.stagnation_counter > self.max_stagnation // 2
    if is_stagnant:
        reward *= 1.5

    # Breakthrough bonus for new global best
    if improvement > 0:
        reward += 2.0

    # Update cumulative reward
    self.operator_cumulative_reward[self.current_operator] += reward
    self.operator_decay_sum[self.current_operator] += 1.0

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