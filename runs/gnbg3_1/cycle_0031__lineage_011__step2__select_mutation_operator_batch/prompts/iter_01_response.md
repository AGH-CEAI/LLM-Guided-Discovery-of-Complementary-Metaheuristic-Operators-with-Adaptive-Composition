**Idea: Thompson Sampling with Beta Distributions**

A Bayesian multi-armed bandit approach that samples from posterior Beta distributions, providing adaptive exploration without explicit exploration parameters.

```python
def _select_mutation_operator_batch(self):
    """Select mutation operator using Thompson Sampling with Beta posterior distributions."""
    # Thompson Sampling: sample from Beta(successes+1, failures+1) for each operator
    # successes = operator_rewards, failures = operator_counts - operator_rewards
    successes = np.clip(self.operator_rewards, 0, self.operator_counts)
    failures = np.maximum(self.operator_counts - successes, 0)
    
    # Sample from Beta distribution for each operator (add 1 for prior)
    # Use shared random state for batch efficiency
    beta_samples = np.random.beta(successes + 1, failures + 1)
    
    # Handle numerical edge cases (all zeros -> uniform selection)
    if np.sum(beta_samples) == 0 or np.any(np.isnan(beta_samples)):
        return np.random.randint(0, self.n_strategies)
    
    # Select operator with highest sampled value
    selected = np.argmax(beta_samples)
    
    return selected
```