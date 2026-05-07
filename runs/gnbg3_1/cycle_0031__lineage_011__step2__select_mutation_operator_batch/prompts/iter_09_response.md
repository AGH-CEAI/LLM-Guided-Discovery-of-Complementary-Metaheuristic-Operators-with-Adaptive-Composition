Looking at the priority targets, I see errors ranging from ~10^1 to ~10^3, indicating these are likely multimodal, deceptive, or ill-conditioned problems where operators get stuck in local optima. The key issue is that the current UCB-based selection with a tiny random perturbation (-1, 0, +1) is essentially deterministic and doesn't provide sufficient diversity for tackling these challenging landscapes.

**Idea: Thompson Sampling with Adaptive Prior**
Thompson Sampling with Beta distributions models uncertainty in operator success rates and samples proportionally to the probability of being optimal, naturally balancing exploration-exploitation and adapting rapidly to each task's specific structure.

```python
def _select_mutation_operator_batch(self):
    """Select mutation operator using Thompson Sampling with Beta distributions."""
    # Track successes and failures for each operator
    if not hasattr(self, 'operator_successes'):
        self.operator_successes = np.ones(self.n_strategies, dtype=float)
    if not hasattr(self, 'operator_failures'):
        self.operator_failures = np.ones(self.n_strategies, dtype=float)
    
    # Sample from Beta distributions for each operator
    samples = np.array([np.random.beta(self.operator_successes[i] + 0.1, 
                                        self.operator_failures[i] + 0.1) 
                        for i in range(self.n_strategies)])
    
    # Add landscape-aware perturbation to bias toward exploration on hard problems
    fitness = getattr(self, 'last_fitness', None)
    if fitness is not None and len(fitness) > 1:
        f_range = np.max(fitness) - np.min(fitness)
        # High variance landscape: boost exploration by adding noise to samples
        if f_range > 1e3:
            samples += np.random.uniform(0, 0.3, size=self.n_strategies)
        elif f_range > 1e1:
            samples += np.random.uniform(0, 0.15, size=self.n_strategies)
    
    # Select operator with highest sampled value
    selected = np.argmax(samples)
    return selected
```