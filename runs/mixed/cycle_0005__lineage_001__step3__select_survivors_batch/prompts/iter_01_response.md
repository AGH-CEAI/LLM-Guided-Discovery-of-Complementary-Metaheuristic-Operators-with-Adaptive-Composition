**Idea: (μ+λ) Pool-Based Elitist Selection**
Classic evolutionary algorithm approach where parent and offspring pools compete; the best NP individuals survive. This fundamentally differs from the current 1-to-1 greedy replacement by providing implicit elitism and ability to recover from bad mutations.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    # Guard against empty or malformed inputs
    if population.shape[0] == 0 or trials.shape[0] == 0:
        return np.copy(population), np.copy(fitness)
    
    # Ensure matching dimensions
    np_pop = min(population.shape[0], trials.shape[0])
    pop_subset = population[:np_pop]
    fit_subset = fitness[:np_pop]
    trial_subset = trials[:np_pop]
    trial_fit_subset = trial_fitness[:np_pop]
    
    # Combine parent and trial pools
    combined_pop = np.vstack([pop_subset, trial_subset])
    combined_fit = np.concatenate([fit_subset, trial_fit_subset])
    
    # Guard against invalid fitness values
    combined_fit = np.clip(combined_fit, -1e50, 1e50)
    
    # Select best NP individuals (μ+λ style)
    if len(combined_fit) > np_pop:
        # Use argpartition for O(n) partial sort, then sort the selected k
        selected_indices = np.argpartition(combined_fit, np_pop)[:np_pop]
        selected_indices = selected_indices[np.argsort(combined_fit[selected_indices])]
    else:
        selected_indices = np.argsort(combined_fit)[:np_pop]
    
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)
    new_population[:np_pop] = combined_pop[selected_indices]
    new_fitness[:np_pop] = combined_fit[selected_indices]
    
    return new_population, new_fitness
```