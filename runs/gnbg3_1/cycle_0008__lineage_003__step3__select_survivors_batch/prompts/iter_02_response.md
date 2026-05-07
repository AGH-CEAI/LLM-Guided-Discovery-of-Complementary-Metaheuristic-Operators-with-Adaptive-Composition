**Idea: Deterministic Crowding with Distance-Based Niching**

The current selection is pure elitist (μ,λ)-selection which is too greedy. The worst tasks (errors ~10²-10⁰) are likely multimodal/deceptive where greedy selection causes premature convergence to local optima. This replacement uses deterministic crowding: each trial competes with its nearest parent in solution space, preserving diversity by keeping both when similar but different fitness.

```python
def _select_survivors_batch(self):
    """Select survivors via deterministic crowding with distance-based niching."""
    n_parents = self.NP
    n_trials = len(self.trials)
    
    if n_trials == 0:
        return
    
    if n_trials < n_parents:
        self.population = self.population[:n_trials]
        self.fitness = self.fitness[:n_trials]
        self.generation += 1
        return
    
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    # Keep top 20% of parents unconditionally (elitist preservation)
    n_elite = max(1, n_parents // 5)
    elite_indices = set(np.argsort(self.fitness)[:n_elite].tolist())
    
    # Deterministic crowding: pair each trial with nearest parent
    crowding_winners = []
    used_parents = set(elite_indices)
    trial_indices = np.arange(n_trials)
    np.random.shuffle(trial_indices)
    
    for t_offset in trial_indices:
        trial_idx = n_parents + t_offset
        if len(crowding_winners) >= n_parents - n_elite:
            break
        
        # Find nearest unused parent using chunked distance computation
        trial_vec = combined_pop[trial_idx]
        available_parents = [i for i in range(n_parents) if i not in used_parents]
        
        if not available_parents:
            continue
        
        min_dist = np.inf
        nearest_parent = available_parents[0]
        
        chunk_size = 32
        for c_start in range(0, len(available_parents), chunk_size):
            c_end = min(c_start + chunk_size, len(available_parents))
            chunk_parents = combined_pop[available_parents[c_start:c_end]]
            dists = np.einsum('ij,ij->i', chunk_parents, chunk_parents) - 2.0 * np.dot(chunk_parents, trial_vec)
            dists = dists + np.dot(trial_vec, trial_vec)
            local_min_idx = np.argmin(dists)
            if dists[local_min_idx] < min_dist:
                min_dist = dists[local_min_idx]
                nearest_parent = available_parents[c_start + local_min_idx]
        
        # Pairwise competition: compare fitness of trial vs. nearest parent
        trial_fit = combined_fit[trial_idx]
        parent_fit = combined_fit[nearest_parent]
        
        if trial_fit <= parent_fit:
            crowding_winners.append(trial_idx)
        else:
            crowding_winners.append(nearest_parent)
            used_parents.add(nearest_parent)
    
    # Combine elite + crowding winners
    all_winners = list(elite_indices) + crowding_winners
    
    # Fill remaining slots with best individuals if needed
    if len(all_winners) < n_parents:
        remaining = [i for i in range(n_parents + n_trials) if i not in set(all_winners)]
        remaining.sort(key=lambda i: combined_fit[i])
        all_winners.extend(remaining[:n_parents - len(all_winners)])
    
    all_winners = all_winners[:n_parents]
    
    self.population = combined_pop[all_winners]
    self.fitness = combined_fit[all_winners]
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```