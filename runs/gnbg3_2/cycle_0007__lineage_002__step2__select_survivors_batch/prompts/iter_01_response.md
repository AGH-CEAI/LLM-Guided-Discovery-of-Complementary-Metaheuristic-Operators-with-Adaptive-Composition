**Idea: k-NN Density Diversity Selection**
Selection based on local spatial sparsity: trials in geometrically underpopulated regions (higher k-NN distance) are accepted to preserve population spread, independent of fitness signals.
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection with k-NN geometric diversity preservation."""
    improved_mask = trial_fitness < fitness

    np_pop, dim = population.shape
    
    # Build combined population for k-NN structure analysis
    combined = np.vstack([trials, population])
    
    # k scales with dimension: enough to capture local structure without noise
    k = min(max(dim // 2, 3), 10)
    k = min(k, np_pop - 1)
    k = max(k, 1)
    
    # Compute full pairwise distance matrix for k-NN structure
    # Shape: (2*np_pop, 2*np_pop)
    dist_matrix = np.linalg.norm(
        combined[:, np.newaxis, :] - combined[np.newaxis, :, :], axis=2
    )
    np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-distance
    
    # k-NN distances for ALL points (sorted)
    kNN_distances = np.sort(dist_matrix, axis=1)[:, :k]
    kNN_mean = np.mean(kNN_distances, axis=1)  # Mean k-NN distance = density proxy
    
    # Split into trial and population indices
    trial_indices = np.arange(np_pop)
    pop_indices = np.arange(np_pop) + np_pop
    
    trial_kNN = kNN_mean[trial_indices]  # Density around each trial
    pop_kNN = kNN_mean[pop_indices]       # Density around each parent
    
    # Geometric diversity criterion: accept if trial is in SPARSER region
    # Higher mean k-NN distance → sparser local neighborhood → preserve spread
    # Small epsilon prevents division issues when pop_kNN is very small
    epsilon = 1e-10
    sparsity_ratio = trial_kNN / (pop_kNN + epsilon)
    
    # Accept if in sparser region (ratio > 1) OR if parent is already very dense
    density_threshold = 1.0
    parent_dense = pop_kNN < np.percentile(pop_kNN, 25)
    accept_diversity = (sparsity_ratio > density_threshold) | parent_dense
    
    # Final acceptance: improved OR geometric diversity benefit
    accept_mask = improved_mask | accept_diversity
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    return new_population, new_fitness, accept_mask
```