**Idea: Covariance Condition Number Guided Selection**

Select survivors based on a combination of fitness and the effect on population covariance condition number. By explicitly measuring and optimizing the condition number of the population covariance matrix, we counteract the anisotropic collapse that prevents progress on the worst tasks (17, 16, 11), which likely have ill-conditioned fitness landscapes with narrow ridges or highly elongated basins.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Select survivors by balancing fitness with covariance condition number health."""
    np_pop, dim = population.shape
    
    # Compute current population condition number (spectral property)
    pop_centered = population - np.mean(population, axis=0)
    cov = np.cov(pop_centered.T)
    if cov.size == 1:
        cov = np.array([[cov]])
    cov += np.eye(dim) * 1e-8
    
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        current_condition = np.max(eigenvalues) / np.min(eigenvalues)
    except:
        current_condition = np.inf
    
    # Compute condition number for each trial's effect on population
    condition_numbers = np.zeros(np_pop)
    for i in range(np_pop):
        temp_pop = population.copy()
        temp_pop[i] = trials[i]
        temp_centered = temp_pop - np.mean(temp_pop, axis=0)
        temp_cov = np.cov(temp_centered.T)
        if temp_cov.size == 1:
            temp_cov = np.array([[temp_cov]])
        temp_cov += np.eye(dim) * 1e-8
        
        try:
            temp_eig = np.linalg.eigvalsh(temp_cov)
            temp_eig = np.maximum(temp_eig, 1e-10)
            condition_numbers[i] = np.max(temp_eig) / np.min(temp_eig)
        except:
            condition_numbers[i] = np.inf
    
    # Trials that improve conditioning (lower condition number) get bonus
    conditioning_bonus = np.clip((current_condition - condition_numbers) / (current_condition + 1e-10), -1.0, 1.0)
    
    # Normalize fitness for combination
    fitness_range = np.max(fitness) - np.min(fitness) + 1e-10
    fitness_normalized = (fitness - np.min(fitness)) / fitness_range
    
    # Combined score: lower is better
    combined_score = fitness_normalized - 0.3 * conditioning_bonus
    
    improved_mask = combined_score < 0
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    return new_population, new_fitness, improved_mask
```