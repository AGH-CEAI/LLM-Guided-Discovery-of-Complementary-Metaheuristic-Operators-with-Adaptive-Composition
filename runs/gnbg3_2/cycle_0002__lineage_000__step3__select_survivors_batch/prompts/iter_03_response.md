**Idea: Entropy-Density Survivor Selection (Category C)**

Information-theoretic survivor selection that evaluates each trial candidate's contribution to population *distribution entropy*, not just fitness. Candidates that improve fitness OR increase population entropy (diversity) are accepted. This addresses the worst tasks (17, 16, 11, 6) which likely suffer from premature convergence by explicitly rewarding diversity during selection.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Entropy-diversity survivor selection (Category C: Information-theoretic)."""
    np_pop, dim = population.shape
    improved_mask = trial_fitness < fitness
    
    # Compute population entropy as sum of per-individual log-volumes (Gaussian KDE)
    pop_normalized = (population - population.mean(axis=0)) / (population.std(axis=0) + 1e-10)
    pop_vars = np.var(pop_normalized, axis=1) + 1e-10
    pop_entropy = 0.5 * np.log(2 * np.pi * np.e * pop_vars ** dim + 1e-100)
    current_entropy = np.sum(pop_entropy)
    
    # Compute candidate entropy contributions
    combined = np.vstack([population, trials])
    combined_norm = (combined - combined.mean(axis=0)) / (combined.std(axis=0) + 1e-10)
    combined_vars = np.var(combined_norm, axis=1) + 1e-10
    combined_entropy = 0.5 * np.log(2 * np.pi * np.e * combined_vars ** dim + 1e-100)
    
    trial_entropy = combined_entropy[np_pop:]
    pop_entropy_ref = combined_entropy[:np_pop]
    
    # Multi-criteria selection: fitness OR entropy
    fit_improvement = fitness - trial_fitness
    entropy_gain = trial_entropy - pop_entropy_ref
    
    fit_threshold = np.percentile(fit_improvement, 70) if np_pop > 3 else 0.0
    entropy_threshold = np.percentile(entropy_gain, 60) if np_pop > 3 else 0.0
    
    accept_criteria = (fit_improvement > fit_threshold) | (entropy_gain > entropy_threshold)
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[accept_criteria] = trials[accept_criteria]
    new_fitness[accept_criteria] = trial_fitness[accept_criteria]
    
    return new_population, new_fitness, improved_mask
```