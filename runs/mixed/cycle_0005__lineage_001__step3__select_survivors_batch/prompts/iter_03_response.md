**Idea: Stagnation-Triggered Boltzmann Exploration**

Uses temperature-scaled probabilistic acceptance to escape local optima. When stagnant, increases temperature to allow worse-but-diverse solutions to survive, breaking the convergence trap that leaves Tasks 16/8/21/12 stuck at massive errors.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    T = max(1e-4, 1.0 / (1.0 + self.generation * 0.02))
    if self.stagnation_count > 15:
        T = min(1.0, T * (1.0 + self.stagnation_count * 0.05))
    elif self.stagnation_count > 5:
        T = min(0.5, T * (1.0 + self.stagnation_count * 0.02))

    improvement = fitness - trial_fitness
    accept_prob = np.clip(np.exp(improvement / T), 0.0, 1.0)
    accept = (trial_fitness < fitness) | (np.random.rand(self.NP) < accept_prob)

    new_population = np.copy(population)
    new_population[accept] = trials[accept]
    new_fitness = np.copy(fitness)
    new_fitness[accept] = trial_fitness[accept]

    elite = min(2, self.NP // 4)
    sorted_idx = np.argsort(fitness)
    for i in range(elite):
        idx = sorted_idx[i]
        if not accept[idx]:
            new_population[idx] = population[idx]
            new_fitness[idx] = fitness[idx]

    return new_population, new_fitness
```