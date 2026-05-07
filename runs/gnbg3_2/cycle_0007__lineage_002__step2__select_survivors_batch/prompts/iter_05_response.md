**Idea: k-NN Graph Connectivity Survivor Selection**

Leverage graph topology to preserve diversity by modulating selection pressure based on k-NN connectivity. Individuals in sparse graph regions (potential unexplored basins) receive a connectivity bonus that relaxes acceptance thresholds, while those in dense regions face stricter selection. This directly addresses multi-modal tasks where premature convergence to a single basin is the failure mode.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Graph-based survivor selection using k-NN connectivity."""
    k = max(3, self.dim // 4)
    
    # Build k-NN graph for current population (sparse, directed)
    n = len(population)
    pop_dists = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
    np.fill_diagonal(pop_dists, np.inf)
    knn_pop = np.argsort(pop_dists, axis=1)[:, :k]
    pop_degrees = np.array([len(set(knn_pop[i]) | set(np.where(pop_dists[i] < np.sort(pop_dists[i])[k])[0])) for i in range(n)])
    
    # Build k-NN graph for trial population
    trial_dists = np.linalg.norm(trials[:, np.newaxis] - trials, axis=2)
    np.fill_diagonal(trial_dists, np.inf)
    knn_trial = np.argsort(trial_dists, axis=1)[:, :k]
    trial_degrees = np.array([len(set(knn_trial[i]) | set(np.where(trial_dists[i] < np.sort(trial_dists[i])[k])[0])) for i in range(n)])
    
    # Normalize degrees and compute graph-based diversity signal
    avg_degree = k * 2  # Expected undirected degree
    pop_degree_norm = pop_degrees / max(avg_degree, 1e-10)
    trial_degree_norm = trial_degrees / max(avg_degree, 1e-10)
    
    # Connectivity bonus: sparse regions (degree_norm < 1) get bonus
    pop_connectivity_bonus = np.maximum(0, 1.0 - pop_degree_norm)
    trial_connectivity_bonus = np.maximum(0, 1.0 - trial_degree_norm)
    
    # Use max connectivity bonus between current and trial for adaptive threshold
    connectivity_bonus = np.maximum(pop_connectivity_bonus, trial_connectivity_bonus)
    
    # Fitness-based selection
    improved_mask = trial_fitness < fitness
    improvement = fitness - trial_fitness
    
    fit_range = np.ptp(fitness)
    fit_range = max(fit_range, 1e-10)
    scaled_improvement = improvement / fit_range
    
    # Graph-modulated acceptance: sparse regions allow smaller improvements
    base_threshold = 0.01
    accept_threshold = base_threshold * (1.0 + connectivity_bonus)
    accept_mask = improved_mask | (scaled_improvement > accept_threshold)
    
    # Prevent isolating highly-connected individuals (they maintain population structure)
    high_conn_mask = pop_degree_norm > 1.5
    accept_mask[high_conn_mask & ~improved_mask] = False
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    return new_population, new_fitness, accept_mask
```