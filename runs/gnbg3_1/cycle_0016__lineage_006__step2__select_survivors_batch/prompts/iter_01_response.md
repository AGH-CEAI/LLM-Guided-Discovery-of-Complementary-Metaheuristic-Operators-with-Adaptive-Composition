**Idea: Hybrid (μ,λ) Selection with Minimal Elitism**
A fundamentally different survivor selection strategy that combines strict (μ,λ) selection from offspring with a small elite preservation, providing stronger selection pressure while maintaining robustness.
```python
def _select_survivors_batch(self):
    """Select survivors via hybrid (mu, lambda) selection with minimal elitism."""
    n_parents = self.NP
    n_offspring = len(self.trials)
    
    if n_offspring == 0:
        return
    
    if n_offspring < n_parents:
        self.population = self.trials[:n_offspring]
        self.fitness = self.trial_fitness[:n_offspring]
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
        return
    
    # Sort offspring by fitness
    sorted_offspring_idx = np.argsort(self.trial_fitness)
    
    # Minimal elitism: keep 1-2 best parents for safety
    n_elite = max(1, min(2, n_parents // 50))
    parent_sorted_idx = np.argsort(self.fitness)
    elite_indices = parent_sorted_idx[:n_elite]
    
    # (mu, lambda) core: select best offspring to fill remaining slots
    n_from_offspring = n_parents - n_elite
    offspring_selected_idx = sorted_offspring_idx[:n_from_offspring]
    
    # Build new population
    elite_inds = self.population[elite_indices]
    offspring_inds = self.trials[offspring_selected_idx]
    
    self.population = np.vstack([elite_inds, offspring_inds])
    self.fitness = np.concatenate([self.fitness[elite_indices], 
                                    self.trial_fitness[offspring_selected_idx]])
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```