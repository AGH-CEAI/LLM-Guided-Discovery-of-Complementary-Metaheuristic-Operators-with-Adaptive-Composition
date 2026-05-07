**Idea: Exponential Crossover**
Replaces binomial crossover with exponential crossover, which copies contiguous segments from the mutant vector, preserving building blocks in separable problems.

```python
def _crossover_batch(self, population, trial_population):
    cr_batch = self.CR.copy()
    n_trials = self.NP
    dim = self.dim
    
    # Initialize offspring as copies of parents
    offspring = population.copy()
    
    # Handle edge case
    if dim == 0 or n_trials == 0:
        return offspring
    
    # For each trial individual, apply exponential crossover
    for i in range(n_trials):
        # Select random starting index
        j_start = self.rng.integers(0, dim)
        
        # Copy elements from mutant starting at j_start
        # Continue copying with probability cr_batch[i] (exponential crossover)
        j = j_start
        while self.rng.random() < cr_batch[i]:
            offspring[i, j] = trial_population[i, j]
            j = (j + 1) % dim
            # Stop if we've wrapped around (at least one element always copied)
            if j == j_start:
                break
    
    return offspring
```