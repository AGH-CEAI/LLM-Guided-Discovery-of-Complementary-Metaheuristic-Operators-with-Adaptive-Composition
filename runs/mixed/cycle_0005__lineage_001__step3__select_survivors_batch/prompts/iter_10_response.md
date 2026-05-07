**Idea: Boltzmann Probabilistic Selection with Adaptive Temperature**

A fundamentally different survivor selection approach: instead of deterministic greedy replacement (keep trial only if strictly better), use simulated annealing/Boltzmann selection where worse solutions can survive with probability dependent on (1) how much worse they are, and (2) an adaptive temperature that increases during stagnation to encourage exploration.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    # Boltzmann selection: probabilistic acceptance of worse solutions
    # Temperature adapts to stagnation — high temp = more exploration
    
    # Adaptive temperature based on stagnation state
    if self.stagnation_count > 30:
        temperature = 2.0 + 0.5 * min(self.stagnation_count - 30, 150)
    elif self.stagnation_count > 15:
        temperature = 0.5 + 0.2 * (self.stagnation_count - 15)
    elif self.stagnation_count > 5:
        temperature = 0.2 + 0.15 * (self.stagnation_count - 5)
    else:
        temperature = 0.1
    
    temperature = max(float(temperature), 0.01)
    
    # Compute fitness delta (positive = trial is worse)
    fitness_delta = trial_fitness - fitness
    
    # Boltzmann acceptance probability
    # Better or equal trials (delta <= 0): accept with P = 1.0
    # Worse trials (delta > 0): P = exp(-delta / T)
    # Clamp exponent for numerical stability
    exponent = np.clip(-fitness_delta / temperature, -700, 700)
    accept_prob = np.where(fitness_delta <= 0, 1.0, np.exp(exponent))
    
    # Probabilistic selection
    roll = np.random.rand(len(fitness))
    accept = roll < accept_prob
    
    # Build new population/fitness
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)
    new_population[accept] = trials[accept]
    new_fitness[accept] = trial_fitness[accept]
    
    return new_population, new_fitness
```