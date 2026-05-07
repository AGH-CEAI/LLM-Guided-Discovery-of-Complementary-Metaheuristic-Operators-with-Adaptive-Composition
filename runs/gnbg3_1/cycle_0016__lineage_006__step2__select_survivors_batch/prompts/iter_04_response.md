**Idea: Deterministic Crowding Survivor Selection**
Uses parent-child competition based on spatial proximity, preserving diversity by allowing multiple similar-fitness solutions to survive rather than forcing all NP best solutions into a single basin.
```python
def _select_survivors_batch(self):
    """Select survivors via deterministic crowding: match trials to nearest parents by distance, keep the better of each pair."""
    pop_size = self.NP
    num_trials = min(len(self.trials), pop_size)
    
    if num_trials == 0 or len(self.population) == 0:
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
    else:
        # Compute pairwise distances between trials and population
        # trials: (num_trials, dim), population: (pop_size, dim)
        # dists[i, j] = distance from trial i to population member j
        dists = np.zeros((num_trials, pop_size))
        for i in range(num_trials):
            diffs = self.trials[i] - self.population  # (pop_size, dim)
            dists[i] = np.sum(diffs * diffs, axis=1)  # squared distances
        
        # For each trial, find the nearest parent
        nearest_parent = np.argmin(dists, axis=1)
        
        # Build candidate set: compare each trial to its nearest parent
        # Keep the better one for each pair
        candidates_pop = []
        candidates_fit = []
        used_parents = set()
        
        # Sort trials by fitness (best first) to prioritize improvements
        trial_order = np.argsort(self.trial_fitness[:num_trials])
        
        for ti in trial_order:
            parent_idx = nearest_parent[ti]
            trial_fit = self.trial_fitness[ti]
            parent_fit = self.fitness[parent_idx]
            
            if trial_fit <= parent_fit:
                # Trial is better or equal - use trial
                candidates_pop.append(self.trials[ti])
                candidates_fit.append(trial_fit)
                used_parents.add(parent_idx)
            else:
                # Parent is better - use parent (if not already selected)
                if parent_idx not in used_parents:
                    candidates_pop.append(self.population[parent_idx])
                    candidates_fit.append(parent_fit)
                    used_parents.add(parent_idx)
        
        # Add remaining unused parents (fill up to NP)
        for i in range(pop_size):
            if i not in used_parents:
                candidates_pop.append(self.population[i])
                candidates_fit.append(self.fitness[i])
                if len(candidates_pop) >= pop_size:
                    break
        
        # Trim or pad to exactly NP
        candidates_pop = np.array(candidates_pop)
        candidates_fit = np.array(candidates_fit)
        
        if len(candidates_pop) > pop_size:
            # If more candidates than slots, keep the best NP
            keep_idx = np.argsort(candidates_fit)[:pop_size]
            self.population = candidates_pop[keep_idx]
            self.fitness = candidates_fit[keep_idx]
        elif len(candidates_pop) < pop_size:
            # If fewer candidates, add random individuals to fill
            self.population = list(candidates_pop)
            self.fitness = list(candidates_fit)
            remaining = pop_size - len(candidates_pop)
            rand_indices = np.random.choice(pop_size, remaining, replace=True)
            for idx in rand_indices:
                self.population.append(self.population[idx])
                self.fitness.append(self.fitness[idx])
            self.population = np.array(self.population)
            self.fitness = np.array(self.fitness)
        else:
            self.population = candidates_pop
            self.fitness = candidates_fit
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```