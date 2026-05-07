**Idea: Entropy-KL Survivor Selection**

Select survivors using information-theoretic metrics: compute population entropy from fitness distribution, measure information gain via KL divergence for each candidate replacement, and use a weighted score combining fitness improvement with entropy gain to maintain exploration on multimodal landscapes.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """
    Information-theoretic survivor selection using entropy and KL divergence.
    Treats the population as a probability distribution; selects candidates
    that improve fitness AND contribute to population entropy/diversity.
    """
    improved_mask = trial_fitness < fitness
    
    # Compute entropy from fitness distribution (histogram-based)
    # This measures the diversity of fitness values in the population
    fitness_flat = fitness.ravel()
    valid_fitness = fitness_flat[~np.isinf(fitness_flat)]
    
    if len(valid_fitness) >= 10:
        hist, bin_edges = np.histogram(valid_fitness, bins='auto', density=True)
        hist = np.maximum(hist, 1e-10)
        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
        population_entropy = -np.sum(hist * np.log(hist)) * bin_width
    else:
        population_entropy = 1.0  # Default for small populations
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    # For each improved candidate, compute KL divergence-based information gain
    for i in range(len(population)):
        if improved_mask[i]:
            trial_val = trial_fitness[i]
            parent_val = fitness[i]
            
            # Estimate probability of parent vs trial in current distribution
            # Using Gaussian kernel approximation around the candidate
            sigma = np.std(valid_fitness) + 1e-10
            
            # Compute p(parent) and q(trial) under current distribution model
            p_parent = np.exp(-0.5 * ((parent_val - valid_fitness) / sigma) ** 2)
            p_trial = np.exp(-0.5 * ((trial_val - valid_fitness) / sigma) ** 2)
            
            p_parent = np.maximum(p_parent.mean(), 1e-10)
            p_trial = np.maximum(p_trial.mean(), 1e-10)
            
            # KL divergence: information gain from replacing parent with trial
            kl_gain = np.log(p_trial / p_parent + 1e-10)
            
            # Normalize fitness improvement to [0,1] range for combination
            fitness_range = np.max(valid_fitness) - np.min(valid_fitness) + 1e-10
            norm_improvement = (parent_val - trial_val) / fitness_range
            
            # Weighted combination: balance fitness improvement with information gain
            # Higher alpha emphasizes information-theoretic diversity
            alpha = 0.7
            score = alpha * norm_improvement + (1 - alpha) * kl_gain
            
            # Accept if score is positive (net benefit to population)
            if score > 0:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
    
    return new_population, new_fitness, improved_mask
```