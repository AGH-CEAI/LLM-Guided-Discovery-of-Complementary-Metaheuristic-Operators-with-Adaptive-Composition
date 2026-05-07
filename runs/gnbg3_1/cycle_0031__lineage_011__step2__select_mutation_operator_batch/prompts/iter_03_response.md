**Idea: Fitness-Aware Thompson Sampling with Diversity Boost**
Thompson Sampling with fitness-based Bayesian update that explicitly boosts exploration when performance is poor across operators, targeting the multimodal worst-case tasks (16, 17, 19, 23) that are stuck ~10+ decades above target.

```python
def _select_mutation_operator_batch(self):
    """Select mutation operator using Thompson Sampling with fitness-aware exploration boost."""
    total_counts = np.sum(self.operator_counts) + 1e-10
    
    # Compute empirical success rates as Bayesian pseudo-rewards
    success_rates = self.operator_rewards / np.maximum(self.operator_counts, 1)
    
    # Detect exploration deficiency: if all operators have similar low success,
    # we're stuck in a local optimum and need aggressive exploration boost
    rate_variance = np.var(success_rates)
    rate_mean = np.mean(success_rates)
    
    # Thompson Sampling: sample from Beta distribution for each operator
    # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
    alpha = self.operator_rewards + 1.0
    beta = self.operator_counts - self.operator_rewards + 1.0
    
    # Clip to avoid numerical issues
    alpha = np.clip(alpha, 1e-10, 1e10)
    beta = np.clip(beta, 1e-10, 1e10)
    
    # Sample from Beta distribution for each operator
    sampled_values = np.random.beta(alpha, beta)
    
    # Exploration boost for stuck scenarios: add entropy-based bonus
    # High entropy = we don't know which operator is best = explore more
    if rate_variance < 0.01 and rate_mean < 0.3:
        # Stuck in local optimum: boost exploration by adding random noise to samples
        exploration_noise = np.random.uniform(0, 0.5, size=self.n_strategies)
        sampled_values += exploration_noise
    
    # Bias toward operators with higher variance in recent performance (novelty signal)
    # This helps escape local optima by favoring operators we haven't fully explored
    novelty_bonus = np.random.uniform(0, 0.2) * (1.0 / np.maximum(self.operator_counts, 1))
    sampled_values += novelty_bonus
    
    # Select operator with highest sampled value
    selected = int(np.argmax(sampled_values))
    
    # Ensure valid index
    selected = np.clip(selected, 0, self.n_strategies - 1)
    
    return selected
```