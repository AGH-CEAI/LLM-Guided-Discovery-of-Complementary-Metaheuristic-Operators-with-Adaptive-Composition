**Idea: Long-Term Memory with Entropy-Maximizing Thompson Sampling**

One-line description: Replace short-window Beta-TS with entropy-maximizing selection that tracks long-term operator performance, uses log-scaled reward aggregation, and adds exploration bonus inversely proportional to cumulative selection count to escape local optima on the worst tasks.

```python
def _select_operator_thompson(self):
    """Thompson Sampling with long-term memory and entropy-promoting exploration."""
    n_ops = self.num_operators
    
    # Long-term reward tracking: cumulative log-reward sum and count
    if not hasattr(self, 'lt_alpha'):
        self.lt_alpha = np.ones(n_ops)
        self.lt_beta = np.ones(n_ops)
        self.lt_reward_sum = np.zeros(n_ops)
        self.lt_reward_count = np.zeros(n_ops)
    
    # Compute improvement and diversity reward
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Log-scaled reward (handles both large and small improvements)
    reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.15 * diversity)
    reward = float(np.clip(reward, -5.0, 5.0))
    
    op = self.current_operator
    self.lt_reward_sum[op] += reward
    self.lt_reward_count[op] += 1.0
    
    # Update Beta parameters using empirical Bayes with Laplace smoothing
    epsilon = 1e-6
    for i in range(n_ops):
        if self.lt_reward_count[i] >= 3.0:
            mean_r = self.lt_reward_sum[i] / max(self.lt_reward_count[i], 1.0)
            var_r = 1.0 / max(self.lt_reward_count[i], 1.0)
            mean_r = np.clip(mean_r, epsilon, 1.0 - epsilon)
            
            denom = max(var_r * self.lt_reward_count[i] + epsilon, epsilon)
            shape1 = max(mean_r * (mean_r * (self.lt_reward_count[i] - 1.0) / denom + 1.0), 1.0)
            shape2 = max((1.0 - mean_r) * ((self.lt_reward_count[i] - 1.0) * (1.0 - mean_r) / denom + 1.0), 1.0)
            
            self.lt_alpha[i] = float(shape1)
            self.lt_beta[i] = float(shape2)
        elif self.lt_reward_count[i] >= 1.0:
            total = float(np.sum(self.lt_reward_sum))
            if total > 0:
                self.lt_alpha[i] = 1.0 + max(self.lt_reward_sum[i], 0.0)
                self.lt_beta[i] = 1.0 + max(-self.lt_reward_sum[i], 0.0)
            else:
                self.lt_alpha[i] = 1.0
                self.lt_beta[i] = 1.5
    
    # Entropy-maximizing exploration bonus
    # Inversely proportional to selection count to promote under-explored operators
    exploration_bonus = np.zeros(n_ops)
    total_selections = float(np.sum(self.operator_counts) + 1.0)
    for i in range(n_ops):
        sel_frac = self.operator_counts[i] / total_selections
        # Bonus peaks for rarely-selected operators, decays for frequently-selected ones
        exploration_bonus[i] = 2.0 * np.exp(-5.0 * sel_frac)
    
    # Sample from Beta and add exploration bonus
    samples = np.random.beta(self.lt_alpha, self.beta)
    samples += exploration_bonus
    
    self.current_operator = int(np.argmax(samples))
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```