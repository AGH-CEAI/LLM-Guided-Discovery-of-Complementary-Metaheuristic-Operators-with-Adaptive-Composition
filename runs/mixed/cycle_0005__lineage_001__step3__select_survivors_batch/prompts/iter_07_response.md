**Idea: Adaptive Temperature Probabilistic Selection**

Instead of purely greedy (1-to-1) selection, this variant probabilistically accepts non-improving trials based on an adaptive "temperature" that responds to stagnation and diversity loss. This allows the population to escape local optima on deceptive multimodal tasks (Tasks 16, 8, 21, 12) by occasionally accepting worse solutions to maintain diversity, while still preferring improvements.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    better = trial_fitness < fitness
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)
    new_population[better] = trials[better]
    new_fitness[better] = trial_fitness[better]

    not_better = ~better
    n_not_better = int(np.sum(not_better))
    if n_not_better == 0:
        return new_population, new_fitness

    # Adaptive temperature: high temp = more exploration
    # Increases with stagnation, decreases when improving
    base_temp = 1e-3
    if self.stagnation_count > 15:
        temp = base_temp * (1.0 + 0.5 * min(self.stagnation_count - 15, 50))
    else:
        temp = base_temp * max(0.1, 1.0 - 0.05 * self.stagnation_count)

    # Also increase temperature when diversity is low
    diversity = self._compute_diversity_batch(population)
    if diversity < self.diversity_threshold * 2:
        temp *= (2.0 - diversity / (self.diversity_threshold * 2 + 1e-10))
    temp = float(np.clip(temp, 1e-6, 10.0))

    # Probabilistically accept some non-improving trials
    not_better_idx = np.where(not_better)[0]
    fitness_not_better = fitness[not_better_idx]
    trial_fitness_not_better = trial_fitness[not_better_idx]

    if len(fitness_not_better) > 0:
        fitness_range = np.ptp(fitness_not_better) + np.ptp(trial_fitness_not_better) + 1e-10
        diff = trial_fitness_not_better - fitness_not_better
        acceptance_probs = np.exp(-diff / (temp * fitness_range + 1e-10))
        acceptance_probs = np.clip(acceptance_probs, 0.0, 1.0)

        rand_vals = np.random.rand(len(acceptance_probs))
        accept_mask = rand_vals < acceptance_probs
        accept_indices = not_better_idx[accept_mask]

        if len(accept_indices) > 0:
            new_population[accept_indices] = trials[accept_indices]
            new_fitness[accept_indices] = trial_fitness[accept_indices]

    return new_population, new_fitness
```