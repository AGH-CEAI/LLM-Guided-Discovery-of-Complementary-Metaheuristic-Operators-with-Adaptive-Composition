**Idea: Stagnation-Driven Boltzmann Selection with Elitism**
A fundamentally different survivor selection that uses adaptive temperature-based acceptance of worse solutions during stagnation to escape local optima, combined with deterministic elitism to preserve global best.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    # Track global best separately for deterministic elitism
    best_idx = np.argmin(fitness)
    global_best = population[best_idx].copy()
    global_best_fit = float(fitness[best_idx])
    
    # Adaptive temperature based on stagnation
    # High stagnation -> high temperature -> accept worse solutions (explore)
    # Low stagnation -> low temperature -> greedy selection (exploit)
    if self.stagnation_count > 20:
        temperature = min(0.5, 0.01 * (self.stagnation_count - 20))
    elif self.stagnation_count > 10:
        temperature = 0.01
    else:
        temperature = 0.001
    
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)
    
    for i in range(len(population)):
        delta = float(fitness[i]) - float(trial_fitness[i])
        if delta > 0:
            # Trial is better: always accept
            new_population[i] = trials[i]
            new_fitness[i] = trial_fitness[i]
        elif self.stagnation_count > 15 and temperature > 1e-6:
            # Trial is worse but we're stagnant: Boltzmann acceptance
            # Higher temperature = more likely to accept worse solutions
            accept_prob = np.exp(delta / temperature)
            if np.random.random() < accept_prob:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
        # else: greedy rejection (standard DE behavior)
    
    # Deterministic elitism: always preserve global best
    new_best_idx = np.argmin(new_fitness)
    if float(new_fitness[new_best_idx]) > global_best_fit:
        new_population[new_best_idx] = global_best
        new_fitness[new_best_idx] = global_best_fit
    
    return new_population, new_fitness
```