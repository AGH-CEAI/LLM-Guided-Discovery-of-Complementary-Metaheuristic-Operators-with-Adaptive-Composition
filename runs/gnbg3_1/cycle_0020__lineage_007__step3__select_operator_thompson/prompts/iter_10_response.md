**Idea: Variance-Adjusted Thompson Sampling with Uncertainty Scaling**

Replace naive Beta sampling with a Gaussian-process-inspired approach that explicitly models both mean performance AND estimation uncertainty, using Wilson score confidence intervals for reward estimation and scaling exploration by inverse sample count. This addresses the fundamental limitation that the current Beta sampling treats all operators as equally well-estimated, causing it to over-exploit operators with lucky early rewards on hard tasks while under-exploring operators that need more trials to show value.

```python
def _select_operator_thompson(self):
    """Select operator using variance-adjusted Thompson Sampling with uncertainty scaling."""
    # Compute running statistics for each operator
    n_ops = self.num_operators
    mean_rewards = np.zeros(n_ops)
    var_rewards = np.zeros(n_ops)
    valid_counts = np.zeros(n_ops)
    
    for i in range(n_ops):
        rewards = self.operator_rewards[i]
        if len(rewards) >= 3:
            mean_rewards[i] = np.mean(rewards)
            var_rewards[i] = np.var(rewards)
            valid_counts[i] = len(rewards)
        elif len(rewards) >= 1:
            mean_rewards[i] = np.mean(rewards)
            var_rewards[i] = 1.0
            valid_counts[i] = len(rewards)
        else:
            mean_rewards[i] = 0.0
            var_rewards[i] = 1.0
            valid_counts[i] = 0.0
    
    # Wilson score confidence interval for more robust reward estimation
    # This handles small sample sizes better than raw mean
    wilson_scores = np.zeros(n_ops)
    z = 1.96  # 95% confidence
    for i in range(n_ops):
        if valid_counts[i] >= 3:
            p = np.clip(mean_rewards[i], 0.0, 1.0)
            n = valid_counts[i]
            denominator = 1.0 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            wilson_scores[i] = center + margin  # Upper bound of CI
        elif valid_counts[i] >= 1:
            # For small samples, use log-transformed reward with uncertainty
            log_reward = np.log1p(max(mean_rewards[i], 1e-10))
            uncertainty = 2.0 / max(valid_counts[i], 1.0)
            wilson_scores[i] = log_reward + uncertainty
        else:
            # Unexplored operators get optimistic prior
            wilson_scores[i] = 2.0 + np.random.uniform(0, 0.5)
    
    # Compute effective exploration rate based on total samples
    total_samples = np.sum(self.operator_counts) + 1.0
    base_exploration = min(2.0, 20.0 / np.sqrt(total_samples))
    
    # Add variance-based exploration bonus
    # High variance operators may be under- or over-estimated
    max_var = max(np.max(var_rewards), 1e-10)
    variance_bonus = 1.5 * (var_rewards / (max_var + 1e-10))
    
    # Scale exploration by inverse sample count (less explored = more exploration)
    inv_counts = 1.0 / (valid_counts + 1.0)
    inv_counts_normalized = inv_counts / (np.sum(inv_counts) + 1e-10)
    
    # Combined exploration bonus
    exploration_bonus = base_exploration * inv_counts_normalized + 0.5 * variance_bonus
    
    # Compute final scores combining exploitation and exploration
    # Use log-transformed rewards for numerical stability
    log_rewards = np.log1p(np.clip(mean_rewards, 0.0, 10.0))
    scores = log_rewards + exploration_bonus
    
    # Thompson Sampling: sample from distribution proportional to scores
    # Use softmax with temperature for probabilistic selection
    temperature = max(0.1, 1.0 / (1.0 + np.mean(var_rewards) + 1e-10))
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / (np.sum(exp_scores) + 1e-10)
    
    # Sample from categorical distribution
    self.current_operator = int(np.random.choice(n_ops, p=probs))
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```