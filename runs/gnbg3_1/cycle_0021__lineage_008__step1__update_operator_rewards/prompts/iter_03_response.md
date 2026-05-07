Looking at the benchmark data, I see a critical pattern: the WORST unsolved tasks (16, 17, 23, 20, 19) all have errors in the 10-70 range, meaning the algorithm gets trapped in terrible local optima. The current reward mechanism is **reactive** (only rewards improvement) and doesn't differentiate between operators that explore new basins vs. those that just make small local improvements.

**Key insight:** For escaping bad local optima, we need to reward **exploration quality** and **personal-best discovery** rather than just relative improvement. The current approach gives low rewards on hard tasks because nothing improves, making Thompson Sampling unable to distinguish good exploration from stagnation.

**Idea: Personal-Best Centric with Exploration Boost**
Track each operator's personal-best fitness independently. Reward based on:
1. Relative improvement from operator's own personal best (not global best)
2. Diversity bonus when population collapses (encourages escaping bad optima)
3. Strong bonus for discovering new personal bests

```python
def _update_operator_rewards(self):
    """Personal-best centric credit assignment with exploration awareness."""
    op = self.current_operator
    
    # Initialize per-operator personal best tracking
    if not hasattr(self, 'operator_best_fitness'):
        self.operator_best_fitness = np.full(self.num_operators, float('inf'))
    if not hasattr(self, 'operator_cumulative_reward'):
        self.operator_cumulative_reward = np.zeros(self.num_operators)
    if not hasattr(self, 'operator_decay_sum'):
        self.operator_decay_sum = np.zeros(self.num_operators)
    
    # Get current best fitness in population
    current_fitness = float(np.min(self.fitness))
    
    # Track if this operator found a new personal best
    is_new_best = False
    if current_fitness < self.operator_best_fitness[op]:
        improvement = self.operator_best_fitness[op] - current_fitness
        self.operator_best_fitness[op] = current_fitness
        is_new_best = True
    else:
        improvement = 0.0
    
    # Relative improvement from operator's own best (0 to 1 scale)
    best_op = self.operator_best_fitness[op]
    worst_op = float(np.max(self.fitness))
    fitness_range = max(worst_op - best_op, 1e-10)
    relative_improvement = improvement / fitness_range
    improvement_reward = float(np.clip(relative_improvement * 10.0, 0.0, 5.0))
    
    # Diversity-based exploration reward
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
    diversity_reward = float(2.0 * (1.0 - diversity))
    
    # Combine rewards: exploration when stagnant, improvement otherwise
    reward = 0.6 * improvement_reward + 0.4 * diversity_reward
    
    # Strong bonus for new personal best (key for escaping bad local optima)
    if is_new_best:
        new_best_bonus = 5.0 * float(np.log1p(max(improvement, 1e-10) / max(abs(best_op), 1e-10)))
        reward += new_best_bonus
    
    # Apply exponential decay and update cumulative reward
    decay = 0.95
    self.operator_cumulative_reward *= decay
    self.operator_decay_sum *= decay
    self.operator_cumulative_reward[op] += reward
    self.operator_decay_sum[op] += 1.0
    
    # Normalize for comparability
    norm = max(self.operator_decay_sum[op], 1.0)
    normalized_reward = float(np.clip(self.operator_cumulative_reward[op] / norm, -5.0, 10.0))
    
    # Update sliding window
    self.operator_rewards[op].append(normalized_reward)
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)
    
    # Update Beta distribution for Thompson Sampling
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