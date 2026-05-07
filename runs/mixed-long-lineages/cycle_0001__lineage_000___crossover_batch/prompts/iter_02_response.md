**Idea: Dimension-Aware Adaptive Crossover**
Crossover that learns which dimensions consistently contribute to fitness improvements and biases offspring inheritance accordingly, using success-history per dimension.

```python
def _crossover_batch(self, population, trial_population):
    n_trials, dim = trial_population.shape
    # Track per-dimension success: positive when trial improves fitness at that dim
    if not hasattr(self, 'dim_success_rate'):
        self.dim_success_rate = np.zeros(dim)
        self.dim_attempt_count = np.zeros(dim)
    # Compute per-dimension improvement signals
    trial_vs_parent = trial_population - population[:n_trials]
    trial_better = (trial_population < population[:n_trials]).astype(float)
    dim_signal = np.mean(trial_better, axis=0)
    self.dim_attempt_count += 1
    # Running average with momentum
    alpha = 0.1
    self.dim_success_rate = alpha * dim_signal + (1 - alpha) * self.dim_success_rate
    # Build adaptive CR per individual based on dimension history
    cr_batch = np.ones(n_trials) * 0.3  # Start conservative
    for i in range(n_trials):
        # Use global dim success rate to weight dimension selection
        dim_probs = self.dim_success_rate.copy()
        dim_probs = np.clip(dim_probs, 0.01, 0.99)
        dim_probs = dim_probs / (np.sum(dim_probs) + 1e-10)
        cr_batch[i] = 0.3 + 0.4 * np.mean(dim_probs)
    cr_batch = np.clip(cr_batch, 0.0, 1.0)
    # Binomial crossover with adaptive per-dim probabilities
    cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        cross_mask[i, j_rand[i]] = True
    offspring = np.where(cross_mask, trial_population, population[:n_trials])
    return offspring
```