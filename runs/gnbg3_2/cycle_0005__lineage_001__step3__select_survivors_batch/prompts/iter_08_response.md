**Idea: Adaptive Greedy-Diversity Hybrid Selection**

Combine greedy fitness-based selection with spatial diversity maintenance using a convergence-driven weight schedule. When improvement rate is high, emphasize fitness (exploitation); when stagnant, increase diversity pressure (exploration). The weight is data-driven from generational improvement signals.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Hybrid greedy/diversity selection with convergence-adaptive weighting."""
    np_pop, dim = population.shape
    
    # Mechanism 1: Greedy fitness-based selection
    improved_mask = trial_fitness < fitness
    
    # Mechanism 2: Spatial diversity (distance from centroid)
    centroid = np.mean(population, axis=0)
    parent_diversity = np.linalg.norm(population - centroid, axis=1) + 1e-10
    trial_diversity = np.linalg.norm(trials - centroid, axis=1) + 1e-10
    
    # Principled weight scheduling via convergence detection
    current_best = np.min(fitness)
    if not hasattr(self, 'prev_best_fitness'):
        self.prev_best_fitness = current_best
        self.improvement_ewma = 0.0
    
    improvement = max(0, self.prev_best_fitness - current_best)
    alpha = 0.2
    self.improvement_ewma = alpha * improvement + (1 - alpha) * self.improvement_ewma
    self.prev_best_fitness = current_best
    
    # Data-driven weight: more diversity pressure when stagnant
    stagnation_threshold = 1e-6 * (1 + 0.1 * self.generation)
    if self.improvement_ewma < stagnation_threshold:
        diversity_weight = 0.4  # Exploration mode
    else:
        diversity_weight = 0.15  # Exploitation mode
    fitness_weight = 1.0 - diversity_weight
    
    # Normalize fitness (lower is better, scale to [0,1])
    fit_min, fit_max = np.min(fitness), np.max(fitness)
    fit_range = fit_max - fit_min + 1e-10
    norm_fitness = (fitness - fit_min) / fit_range
    
    # Normalize diversity (higher is better, scale to [0,1])
    div_min, div_max = np.min(parent_diversity), np.max(parent_diversity)
    div_range = div_max - div_min + 1e-10
    norm_parent_div = (parent_diversity - div_min) / div_range
    norm_trial_div = (trial_diversity - div_min) / div_range
    
    # Compute composite scores
    parent_scores = fitness_weight * norm_fitness - diversity_weight * norm_parent_div
    trial_scores = fitness_weight * ((trial_fitness - fit_min) / fit_range) - \
                   diversity_weight * norm_trial_div
    
    # Select based on composite score (lower is better)
    select_trial = trial_scores < parent_scores
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[select_trial] = trials[select_trial]
    new_fitness[select_trial] = trial_fitness[select_trial]
    
    return new_population, new_fitness, improved_mask
```