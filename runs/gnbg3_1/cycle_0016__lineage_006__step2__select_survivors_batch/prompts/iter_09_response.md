Looking at the worst unsolved tasks (errors around 1e+1 to 4e+1), the algorithm is clearly stuck in poor local optima. The current survivor selection is purely elitist—combining population and trials, then keeping only the top NP individuals. This leads to rapid loss of diversity, trapping the search in local basins.

The key insight: **for the worst tasks, we need to escape local optima by maintaining diversity through niching-style survivor competition rather than pure elitism.**

**Idea: Deterministic Crowding Survivor Selection**
Replaces each trial's closest population member (by Euclidean distance) if the trial is fitter. This forces competition between similar solutions, preserving diversity and enabling the population to spread across multiple basins—crucial for escaping the deceptive local optima causing the 9+ decade errors on Tasks 17, 16, 23, 20, 11, 12.

```python
def _select_survivors_batch(self):
    """Select survivors via deterministic crowding - fitness-distance competition."""
    new_pop = self.population.copy()
    new_fit = self.fitness.copy()
    
    for i, trial in enumerate(self.trials):
        if i >= len(self.trial_fitness):
            break
        trial_fit = self.trial_fitness[i]
        
        # Find most similar individual in current population
        diffs = self.population - trial
        dists_sq = np.sum(diffs * diffs, axis=1)
        closest_idx = np.argmin(dists_sq)
        closest_fit = new_fit[closest_idx]
        
        # Crowding competition: similar solutions compete on fitness
        if trial_fit < closest_fit:
            new_pop[closest_idx] = trial
            new_fit[closest_idx] = trial_fit
    
    self.population = new_pop
    self.fitness = new_fit
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```