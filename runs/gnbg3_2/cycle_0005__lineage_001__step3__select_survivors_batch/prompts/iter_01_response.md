**Idea: Centroid Spread Survivor Selection**
Use centroid-based geometric spread as a secondary criterion in selection: when trial fitness is competitive with current fitness (within 10%), prefer trials that increase population spatial dispersion around the centroid. This maintains geometric diversity while still prioritizing fitness improvements. Fallback to standard greedy selection when population geometric spread is too low or when trial significantly outperforms current individual.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection with centroid-distance geometric diversity maintenance."""
    improved_mask = trial_fitness < fitness
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    # Compute current population centroid (fitness-weighted for robustness)
    weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
    weights = weights / np.sum(weights)
    current_centroid = np.sum(population * weights[:, np.newaxis], axis=0)
    
    # Compute current average distance to centroid (geometric spread metric)
    current_distances = np.linalg.norm(population - current_centroid, axis=1)
    current_avg_spread = np.mean(current_distances)
    
    # Only apply geometric bonus when population has meaningful spread
    min_spread_threshold = 1e-6
    use_geometric_bonus = current_avg_spread > min_spread_threshold
    
    if use_geometric_bonus:
        # Apply selection with geometric diversity bonus
        for i in range(len(population)):
            if trial_fitness[i] < fitness[i]:
                # Compute new centroid if this trial replaces current individual
                temp_weights = weights.copy()
                temp_weights[i] = 1.0 / (trial_fitness[i] - np.min(fitness) + 1e-10)
                temp_weights = temp_weights / np.sum(temp_weights)
                temp_centroid = np.sum(
                    np.where(np.arange(len(population))[:, np.newaxis] == i, 
                             trials[i], population) * temp_weights[:, np.newaxis], 
                    axis=0
                )
                
                # Approximate new distances to centroid (only for changed individual)
                old_dist_i = current_distances[i]
                new_dist_i = np.linalg.norm(trials[i] - temp_centroid)
                
                # Compute diversity gain: positive if trial increases geometric spread
                diversity_gain = new_dist_i - old_dist_i
                normalized_gain = diversity_gain / (current_avg_spread + 1e-10)
                
                # Small bonus for geometric improvement (scaled to not override fitness)
                diversity_bonus = 0.02 * normalized_gain
                
                # Composite score: fitness + diversity bonus (lower is better)
                composite_score = trial_fitness[i] + diversity_bonus
                
                if composite_score < fitness[i]:
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_fitness[i]
                    improved_mask[i] = True
    
    # Fallback: standard greedy selection for non-improvers and when geometry negligible
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    return new_population, new_fitness, improved_mask
```