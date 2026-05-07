**Idea: Diversity-Aware Adaptive Selection**

A fundamentally different selection strategy that maintains a sliding window of recent fitness improvements, dynamically computes population diversity, and uses adaptive acceptance thresholds. When diversity drops below a critical threshold (indicating premature convergence, the likely cause of the worst unsolved tasks being stuck at error ~10^2 to ~10^0), the selection becomes probabilistically more accepting of worse solutions to inject fresh genetic material and escape local optima.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    # Guard against invalid values
    fitness = np.clip(fitness, -1e50, 1e50)
    trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

    # Track recent improvements for adaptive pressure
    improvements = fitness - trial_fitness
    n_improved = int(np.sum(improvements > 0))
    total_improvement = float(np.sum(improvements[improvements > 0]))

    # Add to sliding window
    self.improvement_buffer.append((n_improved, total_improvement))
    if len(self.improvement_buffer) > 10:
        self.improvement_buffer.pop(0)

    # Compute population diversity using coefficient of variation
    pop_std = np.std(population, axis=0)
    pop_range = self.upper - self.lower
    dim_diversity = pop_std / (pop_range + 1e-10)
    diversity = float(np.mean(dim_diversity))

    # Compute adaptive threshold based on recent improvement trends
    if len(self.improvement_buffer) >= 3:
        recent_improvements = [x[1] for x in self.improvement_buffer[-3:]]
        avg_improvement = float(np.mean(recent_improvements))
        improvement_trend = float(np.mean(recent_improvements)) / float(np.mean([max(1e-10, abs(f)) for f in fitness]))
    else:
        avg_improvement = 1.0
        improvement_trend = 1.0

    # Adaptive threshold: lower when stagnating or diversity is low
    base_threshold = 1.0
    diversity_factor = max(0.1, 1.0 - (0.2 * (0.1 - diversity) / 0.1)) if diversity < 0.1 else 1.0
    stagnation_factor = max(0.1, 1.0 - 0.2 * max(0, 10 - self.stagnation_count) / 10.0)
    adaptive_threshold = base_threshold * diversity_factor * stagnation_factor

    new_population = np.copy(population)
    new_fitness = np.copy(fitness)

    # Standard greedy selection with adaptive threshold
    better = trial_fitness < fitness
    new_population[better] = trials[better]
    new_fitness[better] = trial_fitness[better]

    # Diversity injection: probabilistically accept worse solutions when diversity is critically low
    if diversity < 0.1 and self.stagnation_count > 5:
        worse = ~better
        n_worse = int(np.sum(worse))
        if n_worse > 0:
            fitness_gap = fitness[worse] - trial_fitness[worse]
            # Accept worse solutions with probability that increases with diversity crisis
            accept_prob = np.clip(0.15 * (0.1 - diversity) / 0.1, 0.0, 0.5)
            accept_mask = np.random.rand(n_worse) < accept_prob
            accept_idx = np.where(worse)[0][accept_mask]
            if len(accept_idx) > 0:
                new_population[accept_idx] = trials[accept_idx]
                new_fitness[accept_idx] = trial_fitness[accept_idx]

    return new_population, new_fitness
```