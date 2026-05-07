**Idea: Age-Penalized Hybrid Selection with Elitism**
Replaces greedy 1-for-1 with age-aware selection: always keep global best (elitism), replace based on a weighted score combining fitness improvement and solution age, promoting exploration in multimodal landscapes where the algorithm gets trapped at local optima (Tasks 16, 8, 21, 12).

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    NP = len(population)
    
    # Initialize age tracking if needed
    if not hasattr(self, 'individual_ages'):
        self.individual_ages = np.zeros(NP)
    
    # Increment ages of all existing individuals
    self.individual_ages += 1
    
    # Elitism: always keep the single best individual unchanged
    best_idx = int(np.argmin(fitness))
    
    # Compute improvement scores
    improvements = fitness - trial_fitness
    is_better = improvements > 0
    
    # Create replacement mask and tracking arrays
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)
    new_ages = np.copy(self.individual_ages)
    
    # For individuals that improve: compute replacement score
    # Score = improvement + age_bonus (favor replacing older individuals)
    age_bonus = 0.01 * self.individual_ages  # small bonus for older individuals
    replacement_scores = np.where(is_better, improvements + age_bonus, -np.inf)
    
    # Find best replacement for each trial
    for i in range(NP):
        if i == best_idx:
            continue  # Skip elitist
        if replacement_scores[i] > 0:
            new_population[i] = trials[i]
            new_fitness[i] = trial_fitness[i]
            new_ages[i] = 0  # Reset age on successful replacement
    
    # Periodically inject diversity by replacing worst 20% with random
    # but only if stagnation is high (adaptive diversity injection)
    if self.stagnation_count > 15:
        n_replace = max(1, NP // 5)
        worst_indices = np.argsort(new_fitness)[-n_replace:]
        for idx in worst_indices:
            if idx != best_idx:
                # Generate random individual near best but with perturbation
                random_ind = self.best_solution + np.random.randn(self.dim) * (10.0 + self.stagnation_count * 0.1)
                random_ind = np.clip(random_ind, self.lower, self.upper)
                new_population[idx] = random_ind
                new_fitness[idx] = np.inf
                new_ages[idx] = 0
    
    self.individual_ages = new_ages
    return new_population, new_fitness
```