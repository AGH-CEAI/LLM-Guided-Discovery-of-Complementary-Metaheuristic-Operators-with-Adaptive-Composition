**Idea: Diversity-First Reward with Negative Transfer Penalty**

Replace improvement-based rewards with a multi-factor scoring that rewards operators for (1) increasing population diversity, (2) exploring new regions (distance from current mean), and (3) maintaining exploration when stagnant — while penalizing operators that reduce diversity or push the population toward convergence when the algorithm is stuck.

```python
def _update_operator_rewards(self):
    """Reward operators for diversity contribution and exploration, not just improvement."""
    # Initialize tracking state
    if not hasattr(self, 'operator_cumulative_reward'):
        self.operator_cumulative_reward = np.zeros(self.num_operators)
    if not hasattr(self, 'operator_decay_sum'):
        self.operator_decay_sum = np.zeros(self.num_operators)
    if not hasattr(self, 'prev_population'):
        self.prev_population = self.population.copy()
    if not hasattr(self, 'prev_mean'):
        self.prev_mean = self.mean.copy()

    # Compute diversity metrics
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    current_diversity = pop_variance / (expected_var + 1e-10)

    prev_variance = np.mean(np.var(self.prev_population, axis=0))
    prev_diversity = prev_variance / (expected_var + 1e-10)

    # Diversity change reward (positive if operator increased spread)
    diversity_delta = current_diversity - prev_diversity
    diversity_reward = np.log1p(max(diversity_delta, 1e-15)) - np.log1p(max(-diversity_delta, 1e-15))
    diversity_reward = float(np.clip(diversity_reward, -3.0, 3.0))

    # Exploration reward: measure of population movement away from current mean
    pop_distances = np.linalg.norm(self.population - self.mean, axis=1)
    mean_movement = np.linalg.norm(self.mean - self.prev_mean)
    exploration_score = float(np.mean(pop_distances) / (mean_movement + 1e-10))
    exploration_reward = float(np.clip(np.log1p(exploration_score) / 3.0, -2.0, 2.0))

    # Improvement reward (keep for tiebreaking when improvement happens)
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    improvement_reward = 0.0
    if improvement > 0:
        improvement_reward = float(np.clip(np.log1p(improvement * 1e10) / 15.0, 0.0, 5.0))

    # Stagnation detection and bonus
    is_stagnant = self.stagnation_counter > self.max_stagnation // 2
    stagnation_bonus = 1.5 if is_stagnant else 1.0

    # Combine rewards: diversity and exploration weighted higher when stagnant
    if is_stagnant:
        total_reward = (0.15 * improvement_reward + 
                       0.45 * diversity_reward + 
                       0.40 * exploration_reward) * stagnation_bonus
    else:
        total_reward = (0.50 * improvement_reward + 
                       0.25 * diversity_reward + 
                       0.25 * exploration_reward)

    total_reward = float(np.clip(total_reward, -5.0, 5.0))

    # Apply exponential decay to accumulated reward
    decay = 0.95
    self.operator_cumulative_reward *= decay
    self.operator_decay_sum *= decay

    # Update cumulative reward for selected operator
    self.operator_cumulative_reward[self.current_operator] += total_reward
    self.operator_decay_sum[self.current_operator] += 1.0

    # Normalize reward
    norm = max(self.operator_decay_sum[self.current_operator], 1.0)
    normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
    normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

    # Store in sliding window
    op = self.current_operator
    self.operator_rewards[op].append(normalized_reward)
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)

    # Update Beta distribution parameters for Thompson Sampling
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

    # Update tracking state
    self.prev_population = self.population.copy()
    self.prev_mean = self.mean.copy()
```