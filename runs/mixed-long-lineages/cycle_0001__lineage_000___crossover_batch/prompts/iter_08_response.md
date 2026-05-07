**Idea: Proximity-Weighted Rank-Adaptive Crossover**

A crossover that measures how "typical" each dimension is for the parent, then biases inheritance toward typical dimensions more for good individuals (preserving well-converged solutions) and toward atypical dimensions more for poor individuals (enabling escape from local optima).

```python
def _crossover_batch(self, population, trial_population):
    n_trials, dim = self.NP, self.dim
    offspring = population.copy()
    
    # Compute proximity weights: how "central" is each dimension relative to population?
    pop_centroid = np.mean(population, axis=0)
    dim_proximity = 1.0 / (np.abs(population - pop_centroid) + 0.1)
    dim_proximity = dim_proximity / (np.sum(dim_proximity, axis=1, keepdims=True) + 1e-10)
    
    # Rank-based CR scaling: good individuals get conservative crossover
    fitness_ranks = np.argsort(np.argsort(self.population_fitness))
    rank_factors = 0.1 + 0.9 * (fitness_ranks / max(1, self.NP - 1))
    
    for i in range(n_trials):
        # Compute effective CR from rank factor
        effective_cr = self.CR[i] * rank_factors[i]
        
        if self.rng.uniform() < effective_cr:
            # Proximity-weighted selection: prefer dimensions similar to parent
            trial = trial_population[i]
            parent = population[i]
            
            # For each dimension, probability of taking trial value
            # inversely proportional to how "typical" the parent is in that dimension
            dim_deviation = np.abs(parent - pop_centroid)
            deviation_weights = dim_deviation / (np.sum(dim_deviation) + 1e-10)
            
            # Dimensions where parent is close to centroid → keep parent value
            # Dimensions where parent is far from centroid → try trial value
            trial_prob = np.clip(deviation_weights * 3.0, 0.0, 1.0)
            
            # Apply rank-based modulation: good individuals have lower trial_prob
            trial_prob = trial_prob * (1.0 - 0.7 * (1.0 - rank_factors[i]))
            
            # Select dimensions to inherit from trial
            inherit_mask = self.rng.uniform(size=dim) < trial_prob
            
            # Ensure at least one dimension comes from trial
            if not np.any(inherit_mask):
                inherit_mask[self.rng.integers(0, dim)] = True
            
            offspring[i] = np.where(inherit_mask, trial, parent)
        else:
            offspring[i] = population[i]
    
    return offspring
```