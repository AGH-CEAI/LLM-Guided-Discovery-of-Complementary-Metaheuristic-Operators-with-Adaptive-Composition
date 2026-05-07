**Idea: Thompson Sampling Strategy Selection**
Use Thompson sampling (Bayesian sampling from Beta posterior) to select strategies for each individual based on historical success counts.

```python
def _adapt_strategy_probabilities(self):
    # Thompson sampling for adaptive operator selection
    # For each strategy, model success probability with Beta distribution
    # using observed successes and failures.
    for i in range(self.NP):
        samples = np.empty(self.n_strategies)
        for s in range(self.n_strategies):
            successes = self.strategy_success_count[s]
            attempts = self.strategy_attempt_count[s]
            # Posterior parameters: alpha = successes + 1 (prior 1)
            # beta = failures + 1 (prior 1)
            failures = max(attempts - successes, 0.0)
            alpha = successes + 1.0
            beta = failures + 1.0
            # Sample from Beta distribution
            samples[s] = self.rng.beta(alpha, beta)
        # Choose strategy with highest sampled value
        self.strategy_idx[i] = np.argmax(samples)
```