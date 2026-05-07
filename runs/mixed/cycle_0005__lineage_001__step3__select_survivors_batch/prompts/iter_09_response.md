**Idea: Stagnation-Tolerant Probabilistic Survivor Selection**

The current greedy (μ+λ) selection always rejects worse trials, causing premature convergence on deceptive/multi-modal tasks (Tasks 16, 8, 21, 20 stuck at exact values 586.4, 447.5, 50.0, 5.0). This replacement accepts probabilistically worse solutions when stagnant, scaling exploration with stagnation severity to escape local optima basins.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    better = trial_fitness < fitness
    new_population = np.copy(population)
    new_fitness = np.copy(fitness)
    new_population[better] = trials[better]
    new_fitness[better] = trial_fitness[better]
    
    # Probabilistically accept worse solutions when stagnant
    # to escape local optima and maintain diversity
    accept_prob = 1.0 / (1.0 + 0.05 * self.stagnation_count)
    
    worse_mask = ~better_mask
    n_worse = np.sum(worse_mask)
    
    if n_worse > 0:
        worse_probs = np.random.rand(n_worse)
        worse_accepted = worse_probs < accept_prob
        
        worse_indices = np.where(worse_mask)[0]
        accepted_indices = worse_indices[worse_accepted]
        new_population[accepted_indices] = trials[accepted_indices]
        new_fitness[accepted_indices] = trial_fitness[accepted_indices]
    
    return new_population, new_fitness
```