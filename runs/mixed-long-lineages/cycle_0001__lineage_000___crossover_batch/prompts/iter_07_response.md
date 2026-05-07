**Idea: Adaptive Dimensional Crossover with Diversity-Driven CR**
A crossover that adapts per-dimension CR based on population diversity, using lower crossover rates when population is converged to preserve good solutions, and higher rates when diverse to explore more aggressively.

```python
def _crossover_batch(self, population, trial_population):
        n_trials = self.NP
        dim = self.dim
        
        # Compute population diversity for adaptive CR
        fitness_variance = np.var(self.population_fitness) if hasattr(self, 'population_fitness') and len(self.population_fitness) > 1 else 1.0
        fitness_range = np.max(self.population_fitness) - np.min(self.population_fitness) + 1e-10
        diversity_factor = np.clip(fitness_variance / fitness_range, 0.01, 1.0)
        
        # Base CR from individuals, scaled by diversity
        base_cr = np.clip(self.CR.copy(), 0.0, 1.0)
        cr_batch = base_cr * (0.5 + 0.5 * diversity_factor)
        cr_batch = np.clip(cr_batch, 0.1, 0.95)
        
        # Generate crossover masks
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        
        # Force at least one dimension from trial
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        
        # Apply crossover
        offspring = np.where(cross_mask, trial_population, population)
        
        # For highly diverse populations, apply additional dimension-wise mutation
        if diversity_factor > 0.7:
            extra_mut_mask = self.rng.uniform(size=(n_trials, dim)) < 0.05
            extra_mut_mask &= ~cross_mask
            if np.any(extra_mut_mask):
                lo, hi = self.lower, self.upper
                random_values = lo + (hi - lo) * self.rng.uniform(size=(n_trials, dim))
                offspring = np.where(extra_mut_mask, random_values, offspring)
        
        return offspring
```