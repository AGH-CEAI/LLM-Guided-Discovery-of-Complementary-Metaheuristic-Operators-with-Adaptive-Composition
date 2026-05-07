Looking at the priority unsolved tasks (errors 1e+02 to 5e+04), I need a purely geometric survivor selection mechanism. The current implementation uses eigenvalue spread (Category B/Spectral), which has already been tried. My Category A assignment requires literal geometric computations on the point layout.

**Analysis of failure modes**: The worst tasks (17, 16, 11, 19, 6) have errors 100× to 10,000× larger than the best-performing tasks. This suggests the population is collapsing into clustered regions and losing diversity. Geometric diversity measures can detect this collapse before spectral measures (eigenvalues require sufficient spread to be meaningful).

**Distinctive geometric angle**: k-NN (k-nearest-neighbor) spatial density — compute the average distance to k nearest neighbors for each point. Trials in spatially sparse regions (high ratio of trial-NN-distance to population-NN-distance) get a lower acceptance threshold, preserving geometric spread. This is fundamentally different from eigenvalue-based diversity because it captures *local* density structure rather than *global* anisotropy.

**Idea: k-NN Spatial Density Selection**
One-line description (Category A: Geometry/spatial): Uses k-nearest-neighbor distances to measure local spatial density; trials in sparse regions get relaxed acceptance thresholds to preserve geometric spread.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection with k-NN spatial density modulation."""
    improved_mask = trial_fitness < fitness
    
    np_pop, dim = population.shape
    k = max(3, np_pop // 10)
    
    # Compute average k-NN distance for current population (spatial density)
    pop_dist = population - np.mean(population, axis=0)
    pop_dist_matrix = np.linalg.norm(pop_dist[:, np.newaxis] - pop_dist[np.newaxis, :], axis=2)
    np.fill_diagonal(pop_dist_matrix, np.inf)
    pop_nn_distances = np.mean(np.partition(pop_dist_matrix, k, axis=1)[:, :k], axis=1)
    pop_avg_nn = np.mean(pop_nn_distances)
    
    # Compute average k-NN distance for trials relative to population
    diff = trials - population  # (np_pop, dim)
    trial_dist_to_pop = np.linalg.norm(diff[:, np.newaxis, :] - pop_dist[np.newaxis, :, :], axis=2)
    trial_nn_distances = np.mean(np.partition(trial_dist_to_pop, k, axis=1)[:, :k], axis=1)
    
    # Geometric diversity signal: trials in sparse regions (far from pop members) are valuable
    # Ratio > 1 means trial is farther from neighbors than average → sparse region
    # Use log to dampen extreme values
    with np.errstate(divide='ignore', invalid='ignore'):
        nn_ratio = np.log1p(trial_nn_distances) / np.log1p(pop_avg_nn + 1e-10)
        nn_ratio = np.where(np.isfinite(nn_ratio), nn_ratio, 1.0)
    
    # When population is spatially clustered (low pop_avg_nn), reduce selection pressure
    # to preserve any geometric spread the trials might provide
    spatial_diversity_bonus = np.clip(pop_avg_nn / (dim * 10.0), 0.0, 3.0)
    
    # Scaled improvement relative to population fitness range
    fit_range = np.ptp(fitness)
    fit_range = max(fit_range, 1e-10)
    improvement = fitness - trial_fitness
    scaled_improvement = improvement / fit_range
    
    # Accept if improved OR if geometric contribution (sparse region) is significant
    # Higher nn_ratio → more geometric value → lower threshold
    accept_threshold = 0.01 / (1.0 + 0.5 * spatial_diversity_bonus + 0.3 * (nn_ratio - 1.0))
    accept_mask = improved_mask | (scaled_improvement > accept_threshold)
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    return new_population, new_fitness, accept_mask
```