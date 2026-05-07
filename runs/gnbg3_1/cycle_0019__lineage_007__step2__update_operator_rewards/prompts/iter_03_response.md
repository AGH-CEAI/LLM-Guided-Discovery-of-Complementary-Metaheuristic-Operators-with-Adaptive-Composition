**Idea: Regret-Minimizing Reward with Historical Operator Comparison**

Instead of using only immediate fitness improvement, this approach computes **regret** by comparing each operator's performance against a historical best baseline, combined with a multi-fidelity reward that captures both short-term gains and long-term contribution to convergence. This fundamentally differs from the previous variants by explicitly tracking how much better/worse the current operator performs compared to what a simple baseline would have achieved.

```python
def _update_operator_rewards(self):
    """Update operator rewards using regret-minimizing multi-fidelity credit assignment."""
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    
    # Compute population diversity
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Compute fitness variance ratio for regime detection
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Detect convergence regime: 0=exploration, 1=exploitation, 2=stuck
    if rel_var > 0.1 and diversity > 0.3:
        regime = 0  # Exploration phase
    elif rel_var < 0.01 or diversity < 0.05:
        regime = 2  # Stuck/converged
    else:
        regime = 1  # Exploitation phase
    
    op = self.current_operator
    
    # Initialize tracking for this operator if needed
    if not hasattr(self, 'op_improvement_history'):
        self.op_improvement_history = {i: [] for i in range(self.num_operators)}
        self.op_regret_history = {i: [] for i in range(self.num_operators)}
        self.op_regret_baseline = {i: self.f_opt for i in range(self.num_operators)}
    
    # Update improvement history (capped window)
    self.op_improvement_history[op].append(improvement)
    if len(self.op_improvement_history[op]) > 8:
        self.op_improvement_history[op].pop(0)
    
    # Compute regret: how much better/worse than baseline trajectory
    # Baseline assumes no operator change (constant improvement rate)
    baseline_improvement = 0.0
    if len(self.op_regret_history[op]) > 0:
        # Exponential moving average of past improvements as baseline
        baseline_improvement = np.mean(self.op_improvement_history[op]) if self.op_improvement_history[op] else 0.0
    
    regret = improvement - baseline_improvement
    self.op_regret_history[op].append(regret)
    if len(self.op_regret_history[op]) > 5:
        self.op_regret_history[op].pop(0)
    
    # Regret-based reward: positive when outperforming baseline, negative when underperforming
    regret_reward = float(np.tanh(regret * 1e5))  # Bounded [-1, 1], scales with regret magnitude
    
    # Diversity bonus: operators that maintain diversity get slight reward
    diversity_bonus = 0.2 * diversity if regime == 0 else 0.05 * diversity
    
    # Regime-aware bonus: reward exploration in stuck phase, exploitation otherwise
    if regime == 2:  # Stuck - reward any improvement heavily
        regime_bonus = 2.0 * np.log1p(improvement * 1e10) / 10.0
    elif regime == 0:  # Exploration - reward maintaining diversity
        regime_bonus = 0.5 * diversity
    else:  # Exploitation - reward convergence
        regime_bonus = 0.5 * np.log1p(improvement * 1e10) / 10.0
    
    # Combine all reward components
    reward = regret_reward + diversity_bonus + regime_bonus
    reward = float(np.clip(reward, -10.0, 10.0))
    
    # Update sliding window of rewards
    self.operator_rewards[op].append(reward)
    if len(self.operator_rewards[op]) > self.reward_window_size:
        self.operator_rewards[op].pop(0)
    
    # Update Beta distribution parameters for Thompson Sampling
    n = len(self.operator_rewards[op])
    if n >= 3:
        mean_reward = float(np.mean(self.operator_rewards[op]))
        var_reward = float(np.var(self.operator_rewards[op]))
        
        # Bayesian update with variance-aware parameters
        denom = max(var_reward * n + 1e-10, 1e-10)
        self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
        self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
    elif n >= 1:
        sum_reward = float(np.sum(self.operator_rewards[op]))
        if sum_reward > 0:
            self.alpha[op] = 1.0 + sum_reward
        else:
            self.beta[op] = 1.0 - sum_reward
    
    # Update baseline for this operator
    if hasattr(self, 'op_regret_baseline'):
        self.op_regret_baseline[op] = self.f_opt
```