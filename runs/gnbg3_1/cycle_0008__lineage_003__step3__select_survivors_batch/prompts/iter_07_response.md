**Idea: Spatial Crowding Survivor Selection**

Replace pure elitist selection with a fitness-distance crowding mechanism that preserves population diversity. Each trial competes with its nearest parent in parameter space (not position in combined list), ensuring diverse solutions survive even when fitness is slightly worse. This directly addresses the convergence collapse observed in the worst tasks (errors 10-100+), where the population converges to a single region and loses exploration capability.
```python
def _select_survivors_batch(self):
    """Select survivors via fitness-distance crowding to preserve diversity."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    n_parents = self.NP
    n_trials = len(self.trials)
    
    # Compute pairwise Euclidean distances between trials and parents
    # Shape: (n_trials, n_parents)
    diff = self.trials[:, np.newaxis, :] - self.population[np.newaxis, :, :]  # Broadcasting
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # Euclidean distances
    
    # For each trial, find the nearest parent (crowding partner)
    nearest_parent = np.argmin(dists, axis=1)  # Shape: (n_trials,)
    nearest_dist = np.min(dists, axis=1)  # Shape: (n_trials,)
    
    # Initialize replacement array: -1 means not replaced yet
    replacements = np.full(n_trials, -1, dtype=int)
    
    # Track which parents are claimed
    parent_claimed = np.zeros(n_parents, dtype=bool)
    
    # Sort trials by fitness (best first) to give better solutions priority
    trial_order = np.argsort(self.trial_fitness)
    
    for ti in trial_order:
        pi = nearest_parent[ti]
        if not parent_claimed[pi]:
            # Trial beats or ties parent in fitness - claim this parent slot
            if self.trial_fitness[ti] <= self.fitness[pi] + 1e-15:
                replacements[ti] = pi
                parent_claimed[pi] = True
    
    # Build new population
    new_pop = self.population.copy()
    new_fit = self.fitness.copy()
    
    for ti in range(n_trials):
        if replacements[ti] >= 0:
            pi = replacements[ti]
            new_pop[pi] = self.trials[ti]
            new_fit[pi] = self.trial_fitness[ti]
    
    # Handle unclaimed parents: keep best unclaimed individuals from combined pool
    unclaimed_mask = ~parent_claimed
    unclaimed_indices = np.where(unclaimed_mask)[0]
    
    if len(unclaimed_indices) > 0:
        # Find best individuals from combined pool excluding already-selected
        selected_mask = np.zeros(len(combined_fit), dtype=bool)
        for pi in np.where(parent_claimed)[0]:
            selected_mask[pi] = True
        for ti in np.where(replacements >= 0)[0]:
            selected_mask[n_parents + ti] = True
        
        remaining_pool = np.where(~selected_mask)[0]
        if len(remaining_pool) > 0:
            remaining_fits = combined_fit[remaining_pool]
            best_remaining = np.argsort(remaining_fits)
            fill_count = min(len(unclaimed_indices), len(best_remaining))
            for k in range(fill_count):
                idx = remaining_pool[best_remaining[k]]
                if idx < n_parents:
                    new_pop[unclaimed_indices[k]] = combined_pop[idx]
                    new_fit[unclaimed_indices[k]] = combined_fit[idx]
                # else: trial already placed, skip
    
    self.population = new_pop
    self.fitness = new_fit
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```