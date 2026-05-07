**Idea: Spatial Crowding Replacement with Distance-Aware Selection**

Category A: Geometry / spatial. Use pairwise Euclidean distances between trial vectors and population members to detect spatial crowding and probabilistically reject trials that would reduce geometric spread, even if they improve fitness.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection with spatial crowding detection via pairwise distances."""
    np_pop, dim = population.shape
    improved_mask = trial_fitness < fitness
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    # Compute spatial spread estimate for distance threshold
    # Use max extent as reference scale (robust to outliers)
    pop_range = np.ptp(population, axis=0)  # Per-dimension range
    spatial_threshold = 0.08 * np.sqrt(np.sum(pop_range**2)) + 1e-10
    
    # Compute pairwise distances between each trial and all population members
    # Using broadcasting: trials[:, np.newaxis, :] - population[np.newaxis, :, :]
    # Shape: (np_pop, np_pop, dim)
    diff = trials[:, np.newaxis, :] - population[np.newaxis, :, :]
    trial_to_pop_dist = np.sqrt(np.sum(diff**2, axis=2))  # (np_pop, np_pop)
    
    # For each trial, find minimum distance to any population member (excluding self)
    # Create mask to exclude self-comparison
    identity_mask = ~np.eye(np_pop, dtype=bool)
    trial_to_pop_dist_masked = np.where(identity_mask, trial_to_pop_dist, np.inf)
    min_dist_to_pop = np.min(trial_to_pop_dist_masked, axis=1)  # (np_pop,)
    
    # Process each individual
    for i in range(np_pop):
        if trial_fitness[i] < fitness[i]:
            # Trial improves fitness
            if min_dist_to_pop[i] < spatial_threshold:
                # Too close to existing population - probabilistic accept
                # Higher probability for more unique positions
                accept_prob = min(0.8, min_dist_to_pop[i] / (spatial_threshold + 1e-10))
                if np.random.rand() > accept_prob:
                    # Reject to preserve spatial diversity
                    continue
            # Accept the improvement
            new_population[i] = trials[i]
            new_fitness[i] = trial_fitness[i]
    
    return new_population, new_fitness, improved_mask
```