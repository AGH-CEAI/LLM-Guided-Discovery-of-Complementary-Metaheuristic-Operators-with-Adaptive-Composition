**Idea: Niching-Based Deterministic Crowding Survivor Selection**

Replace elitist selection with deterministic crowding: match each trial offspring with its most similar parent, keep the better one, and maintain diversity through niche preservation. This prevents premature convergence on deceptive high-dimensional landscapes (Tasks 17, 16, 23) by allowing parallel exploration of multiple basins.

```python
def _select_survivors_batch(self):
    """Select survivors via deterministic crowding with niching to preserve diversity."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    # Determine number of offspring
    n_trials = len(self.trials)
    n_parents = len(self.population)
    
    # Compute pairwise distances between trials and parents
    # Use efficient broadcasting: (n_trials, 1, dim) - (1, n_parents, dim)
    trials_exp = self.trials[:, np.newaxis, :]  # (n_trials, 1, dim)
    parents_exp = self.population[np.newaxis, :, :]  # (1, n_parents, dim)
    dists = np.linalg.norm(trials_exp - parents_exp, axis=2)  # (n_trials, n_parents)
    
    # For each trial, find its most similar parent
    closest_parent = np.argmin(dists, axis=1)  # (n_trials,)
    
    # Build new population through deterministic crowding
    new_pop = []
    new_fit = []
    
    # Add all parents as candidates
    for i in range(n_parents):
        new_pop.append(self.population[i])
        new_fit.append(self.fitness[i])
    
    # For each trial, compare with its closest parent
    for i in range(n_trials):
        parent_idx = closest_parent[i]
        trial_fit = combined_fit[n_parents + i]
        parent_fit = self.fitness[parent_idx]
        
        # Keep whichever has better fitness (minimization)
        if trial_fit < parent_fit:
            new_pop.append(self.trials[i])
            new_fit.append(trial_fit)
        else:
            # Still add parent (already in list, will deduplicate later)
            pass
    
    # If population exceeds NP, use fitness-based pruning with diversity bonus
    if len(new_pop) > self.NP:
        new_pop = np.array(new_pop)
        new_fit = np.array(new_fit)
        
        # Compute diversity contribution of each individual
        pop_stack = np.vstack(new_pop)
        
        # Pairwise distances for diversity
        pdists = np.linalg.norm(pop_stack[:, np.newaxis, :] - pop_stack[np.newaxis, :, :], axis=2)
        
        # Average distance to others (diversity score)
        avg_dist = np.mean(pdists, axis=1)
        
        # Combined score: prioritize fitness but add small diversity bonus
        # Normalize fitness to [0, 1] range for combination
        fit_min, fit_max = np.min(new_fit), np.max(new_fit)
        fit_range = max(fit_max - fit_min, 1e-10)
        fit_norm = (new_fit - fit_min) / fit_range
        
        # Diversity-normalized selection score
        diversity_weight = 0.05
        selection_score = fit_norm - diversity_weight * (avg_dist / (np.max(avg_dist) + 1e-10))
        
        # Select best NP individuals
        best_indices = np.argsort(selection_score)[:self.NP]
        
        new_pop = pop_stack[best_indices]
        new_fit = new_fit[best_indices]
    else:
        new_pop = np.array(new_pop[:self.NP]) if len(new_pop) >= self.NP else np.array(new_pop)
        new_fit = np.array(new_fit[:self.NP]) if len(new_fit) >= self.NP else np.array(new_fit)
    
    # Ensure we have exactly NP individuals
    if len(new_pop) < self.NP:
        # Pad with random individuals from combined population
        deficit = self.NP - len(new_pop)
        combined_sorted = np.argsort(combined_fit)
        padding_indices = combined_sorted[len(new_pop):len(new_pop) + deficit]
        padding_pop = combined_pop[padding_indices]
        padding_fit = combined_fit[padding_indices]
        new_pop = np.vstack([new_pop, padding_pop])
        new_fit = np.concatenate([new_fit, padding_fit])
    
    self.population = new_pop[:self.NP]
    self.fitness = new_fit[:self.NP]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```