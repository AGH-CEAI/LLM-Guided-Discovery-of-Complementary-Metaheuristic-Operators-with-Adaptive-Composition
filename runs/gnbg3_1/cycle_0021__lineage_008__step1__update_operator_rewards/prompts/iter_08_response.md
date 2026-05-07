Looking at the task coverage data, I see that ALL 24 tasks remain unsolved with the worst tasks (16, 17, 20, 19, 23) having errors in the 9-60 range, indicating highly multi-modal or deceptive landscapes where operators fail to escape local optima.

**Analysis of the problem:**
The current `_update_operator_rewards` uses:
- Log-scaled improvement (compressed scale, loses discrimination at large errors)
- Simple stagnation counter (crude binary signal)
- Uniform decay (0.95) that may lose critical early signals
- Single-objective reward (just improvement magnitude)

For the worst tasks with errors ~10-60, the log-scaled reward becomes nearly identical regardless of actual progress, making it impossible for Thompson Sampling to distinguish which operator is actually best.

**Idea: Comparative Rank-Based Multi-Objective Reward with Adaptive Difficulty Scaling**

This fundamentally different approach:
1. Uses **comparative ranking** across operators per-generation (not absolute improvement)
2. Applies **difficulty-aware scaling** that amplifies rewards when facing hard problems
3. Tracks **consecutive success streaks** to identify consistently good operators
4. Uses **population-level diversity contribution** as a secondary reward signal

```python
def _update_operator_rewards(self):
    """Comparative rank-based multi-objective credit assignment with difficulty-aware scaling."""
    # Initialize tracking structures
    if not hasattr(self, 'operator_rank_history'):
        self.operator_rank_history = {i: [] for i in range(self.num_operators)}
    if not hasattr(self, 'operator_streak'):
        self.operator_streak = np.zeros(self.num_operators)
    if not hasattr(self, 'best_known_errors'):
        self.best_known_errors = np.full(24, np.inf)
    if not hasattr(self, 'generation_counter'):
        self.generation_counter = 0

    self.generation_counter += 1
    
    # Compute improvement and normalize by problem scale
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    problem_scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
    relative_improvement = improvement / problem_scale if problem_scale > 0 else 0.0
    
    # Difficulty-aware scaling: amplify rewards for harder problems
    # Large errors (multi-modal/deceptive) need stronger signals
    if self.f_opt > 1.0:
        difficulty_mult = min(10.0, np.log10(self.f_opt + 1.0) + 1.0)
    elif self.f_opt > 1e-3:
        difficulty_mult = 2.0
    elif self.f_opt > 1e-6:
        difficulty_mult = 1.5
    else:
        difficulty_mult = 1.0
    
    # Compute comparative rank of this operator vs others in recent history
    op = self.current_operator
    recent_rewards = []
    for i in range(self.num_operators):
        if len(self.operator_rewards[i]) > 0:
            recent_rewards.append((i, np.mean(self.operator_rewards[i][-5:])))
        else:
            recent_rewards.append((i, 0.0))
    
    # Rank operators by recent performance (1 = best)
    sorted_by_perf = sorted(recent_rewards, key=lambda x: x[1], reverse=True)
    rank_map = {op_idx: rank + 1 for rank, (op_idx, _) in enumerate(sorted_by_perf)}
    current_rank = rank_map[op]
    
    # Rank-based reward: top rank gets high reward, bottom gets penalty
    rank_score = (self.num_operators - current_rank) / max(self.num_operators - 1, 1)
    rank_score = 2.0 * rank_score - 1.0  # Map to [-1, 1]
    
    # Improvement-based component (not log-compressed for large improvements)
    if improvement > 0:
        # Use sqrt scaling to preserve discrimination at large errors
        improvement_score = np.sqrt(improvement) / (np.sqrt(problem_scale) + 1.0)
        improvement_score = min(improvement_score, 5.0)
    else:
        improvement_score = 0.0
    
    # Streak tracking: reward consistent performers
    if improvement > 1e-15:
        self.operator_streak[op] += 1
    else:
        self.operator_streak[op] = max(0, self.operator_streak[op] - 1)
    
    streak_bonus = np.tanh(self.operator_streak[op] / 10.0) * 0.5
    
    # Diversity contribution reward
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Check if this operator helped maintain/increase diversity
    diversity_bonus = 0.0
    if hasattr(self, 'prev_diversity'):
        if diversity > self.prev_diversity * 1.1:
            diversity_bonus = 0.3 * (diversity - self.prev_diversity) / (self.prev_diversity + 1e-10)
    self.prev_diversity = diversity
    
    # Combine all reward components
    combined_reward = (
        0.4 * rank_score +
        0.4 * improvement_score * difficulty_mult +
        0.1 * streak_bonus +
        0.1 * diversity_bonus
    )
    combined_reward = float(np.clip(combined_reward, -5.0, 10.0))
    
    # Update rank history
    self.operator_rank_history[op].append(current_rank)
    if len(self.operator_rank_history[op]) > 20:
        self.operator_rank_history[op].pop(0)
    
    # Update reward window
    self.operator_rewards[op].append(combined_reward)
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)
    
    # Update Beta distribution parameters for Thompson Sampling
    n = len(self.operator_rewards[op])
    if n >= 3:
        mean_reward = float(np.mean(self.operator_rewards[op]))
        var_reward = float(np.var(self.operator_rewards[op]))
        
        # Use rank consistency as confidence multiplier
        avg_rank = np.mean(self.operator_rank_history[op])
        rank_confidence = 1.0 / (1.0 + avg_rank / self.num_operators)
        
        effective_mean = mean_reward * (0.5 + 0.5 * rank_confidence)
        denom = max(var_reward * n + 1e-10, 1e-10)
        
        self.alpha[op] = float(max(0.5, effective_mean * (effective_mean * (n - 1) / denom + 1)))
        self.beta[op] = float(max(0.5, (1 - effective_mean) * ((n - 1) * (1 - effective_mean) / denom + 1)))
    elif n >= 1:
        sum_reward = float(np.sum(self.operator_rewards[op]))
        if sum_reward > 0:
            self.alpha[op] = 1.0 + sum_reward
            self.beta[op] = 1.0
        else:
            self.alpha[op] = 1.0
            self.beta[op] = 1.0 - sum_reward + 1.0
```