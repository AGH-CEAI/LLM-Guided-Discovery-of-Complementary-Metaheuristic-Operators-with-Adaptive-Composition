**Idea: Per-Operator Rank-Normalized Reward with Diversity-Sensitive Learning**

This variant fundamentally changes reward computation by using per-operator rank-normalized improvement (not just global improvement), operator-specific success rates, and diversity-weighted credit assignment. This creates much stronger discrimination between operators on difficult tasks.
```python
def _update_operator_rewards(self):
    """Update operator rewards using per-operator rank-normalized credit assignment."""
    # Compute global metrics
    f_vals = np.concatenate([self.fitness, self.trial_fitness])
    f_prev = self.f_opt_prev
    
    # Per-operator success tracking
    if not hasattr(self, 'op_success_history'):
        self.op_success_history = {i: [] for i in range(self.num_operators)}
    if not hasattr(self, 'op_improvement_history'):
        self.op_improvement_history = {i: [] for i in range(self.num_operators)}
    
    # Compute operator-specific improvement (based on selection counts)
    op = self.current_operator
    self.selection_counts[op] += 1
    
    # Individual improvement contribution
    if len(self.trial_fitness) >= self.NP:
        individual_improvement = max(0.0, self.f_opt_prev - np.min(self.trial_fitness))
    else:
        individual_improvement = max(0.0, self.f_opt_prev - self.f_opt)
    
    # Store in per-operator history
    self.op_improvement_history[op].append(individual_improvement)
    if len(self.op_improvement_history[op]) > self.reward_window_size:
        self.op_improvement_history[op].pop(0)
    
    # Per-operator success rate
    recent_improved = 0
    recent_total = min(self.NP, len(self.trial_fitness))
    if recent_total > 0:
        for i in range(recent_total):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
    op_success_rate = recent_improved / max(recent_total, 1)
    
    self.op_success_history[op].append(op_success_rate)
    if len(self.op_success_history[op]) > self.reward_window_size:
        self.op_success_history[op].pop(0)
    
    # Population diversity
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Fitness landscape roughness (variance of recent improvements)
    all_improvements = []
    for i in range(self.NP):
        if i < len(self.trial_fitness) and i < len(self.fitness):
            imp = max(0.0, self.fitness[i] - self.trial_fitness[i])
            all_improvements.append(imp)
    
    roughness = 0.0
    if len(all_improvements) > 1:
        roughness = np.var(all_improvements) / (np.mean(all_improvements) ** 2 + 1e-10)
    roughness = np.clip(roughness, 0.0, 10.0)
    
    # Rank-normalized improvement: compare this operator's recent performance
    # to what it achieved in previous windows
    op_improvement_mean = np.mean(self.op_improvement_history[op]) if self.op_improvement_history[op] else 0.0
    op_success_mean = np.mean(self.op_success_history[op]) if self.op_success_history[op] else 0.0
    
    # Normalize by operator's own baseline (adaptive comparison)
    improvement_ratio = 1.0
    if op_improvement_mean > 1e-15:
        improvement_ratio = np.clip(individual_improvement / (op_improvement_mean + 1e-15), 0.0, 5.0)
    
    success_ratio = 1.0
    if op_success_mean > 1e-15:
        success_ratio = np.clip(op_success_rate / (op_success_mean + 1e-15), 0.0, 5.0)
    
    # Combined reward with multiple signals
    reward = 0.0
    
    # 1. Log-scaled improvement (core signal)
    reward += np.log1p(individual_improvement * 1e10) / 10.0
    
    # 2. Per-operator rank normalization bonus
    reward += 0.5 * (improvement_ratio - 1.0)
    reward += 0.3 * (success_ratio - 1.0)
    
    # 3. Diversity contribution (diversity = good, low diversity = penalize)
    reward += 0.2 * diversity
    
    # 4. Landscape roughness adaptation
    # High roughness = complex landscape = reward exploration
    reward += 0.1 * np.tanh(roughness - 1.0)
    
    reward = float(np.clip(reward, -10.0, 10.0))
    
    # Update sliding window
    self.operator_rewards[op].append(reward)
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)
    
    # Update Beta distribution parameters
    n = len(self.operator_rewards[op])
    if n >= 3:
        mean_reward = float(np.mean(self.operator_rewards[op]))
        var_reward = float(np.var(self.operator_rewards[op]))
        
        # Adaptive learning rate based on variance
        cv_sq = var_reward / (mean_reward ** 2 + 1e-10)  # coefficient of variation squared
        lr_scale = 1.0 / (1.0 + np.sqrt(max(cv_sq, 0.01)))
        lr_scale = np.clip(lr_scale, 0.3, 3.0)
        
        denom = max(var_reward * n + 1e-10, 1e-10)
        alpha_update = mean_reward * (mean_reward * (n - 1) / denom + 1) * lr_scale
        beta_update = (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1) * lr_scale
        
        self.alpha[op] = float(max(1.0, alpha_update))
        self.beta[op] = float(max(1.0, beta_update))
    elif n >= 1:
        sum_reward = float(np.sum(self.operator_rewards[op]))
        if sum_reward > 0:
            self.alpha[op] = max(1.0, 1.0 + sum_reward)
        else:
            self.beta[op] = max(1.0, 1.0 - sum_reward)
```