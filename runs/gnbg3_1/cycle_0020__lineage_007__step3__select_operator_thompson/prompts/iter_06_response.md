Looking at the data, I need to understand why all 24 tasks remain unsolved. The worst tasks (16, 17, 20, 23) have errors ~1e+01, suggesting extreme difficulty - likely highly multimodal, ill-conditioned, or deceptive landscapes. The current Thompson Sampling approach samples from Beta distributions but doesn't adequately handle:

1. **Exploration-exploitation balance**: When all operators struggle on hard tasks, the sampling remains too conservative
2. **Per-task operator affinity**: Different operators may have hidden strengths on specific task types that aren't being discovered
3. **Failure diversity**: Operators that consistently fail should be deprioritized faster

The current implementation uses uniform Beta(1,1) priors with reward-based updates - a standard but slow approach. I need something fundamentally different that aggressively explores the operator space when convergence stalls.

**Idea: Entropy-Maximizing Thompson Sampling with Diversity Pressure**

This variant uses entropy of the operator reward distribution to dynamically adjust exploration pressure. When rewards are similar (high entropy = no clear winner), we increase exploration by widening the Beta distribution parameters. When one operator clearly dominates (low entropy), we exploit by narrowing distributions. This ensures the adaptive selector doesn't get stuck in local optima of the operator selection space itself.

```python
def _select_operator_thompson(self):
    """Select operator using entropy-weighted Thompson Sampling with diversity pressure."""
    # Compute entropy-based exploration control
    total_reward = np.sum(self.alpha) + np.sum(self.beta)
    if total_reward > self.num_operators * 2:
        # Compute entropy of reward distribution
        alpha_sum = np.sum(self.alpha)
        beta_sum = np.sum(self.beta)
        probs = self.alpha / (alpha_sum + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(self.num_operators)
        entropy_ratio = entropy / (max_entropy + 1e-10)
        
        # High entropy = explore more (widen distributions)
        # Low entropy = exploit (narrow distributions)
        exploration_scale = 0.5 + 2.0 * (1.0 - entropy_ratio)
        exploration_scale = np.clip(exploration_scale, 0.5, 3.0)
    else:
        exploration_scale = 1.5  # Initial exploration bias
    
    # Sample from scaled Beta distributions
    samples = np.random.beta(self.alpha * exploration_scale, self.beta * exploration_scale)
    
    # Apply diversity pressure: boost probability of under-explored operators
    if np.sum(self.selection_counts) > self.num_operators * 3:
        min_selections = np.min(self.selection_counts)
        selection_deficit = min_selections / (self.selection_counts + 1e-10)
        diversity_boost = 0.1 * selection_deficit
        samples = samples + diversity_boost * np.random.random(self.num_operators)
    
    self.current_operator = int(np.argmax(samples))
    self.operator_counts[self.current_operator] += 1
    self.selection_counts[self.current_operator] += 1
    return self.current_operator
```