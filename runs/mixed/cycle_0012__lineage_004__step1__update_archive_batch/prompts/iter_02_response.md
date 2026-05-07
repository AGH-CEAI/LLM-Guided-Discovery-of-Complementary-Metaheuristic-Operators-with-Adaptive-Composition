**Idea: Fitness-Diversity Tiered Archive**
A fundamentally different archive strategy that maintains two tiers: (1) elite tier with best-fitness solutions for exploitation, and (2) diversity tier selected by maximum distance from all other archive members for exploration. This directly addresses multimodal tasks (12, 8, 21) where escaping local optima requires diverse mutation sources.

```python
def _update_archive_batch(self, archive, new_solutions, new_fitness):
    if len(new_solutions) == 0:
        return archive
    
    combined = np.vstack([archive, new_solutions])
    n_combined = len(combined)
    
    # Compute fitness for all: archive members get -inf (unknown), new solutions get actual fitness
    all_fitness = np.concatenate([
        np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
        new_fitness
    ])
    
    # Remove near-duplicates
    unique_mask = self._find_unique_solutions(combined)
    combined = combined[unique_mask]
    all_fitness = all_fitness[unique_mask]
    
    n_unique = len(combined)
    if n_unique == 0:
        return archive
    
    # Tier 1: Keep best solutions by actual fitness (new solutions)
    new_mask = np.arange(len(combined)) >= len(archive)
    new_combined = combined[new_mask]
    new_fitness_filtered = all_fitness[new_mask]
    
    n_elite = min(self.archive_max // 2, len(new_combined))
    elite_indices = np.argsort(new_fitness_filtered)[:n_elite]
    elite_solutions = new_combined[elite_indices]
    
    # Tier 2: Diversity tier - farthest solutions from elite for exploration
    n_diversity = self.archive_max - n_elite
    remaining = combined[~new_mask] if len(archive) > 0 else combined
    
    if len(remaining) == 0:
        return elite_solutions
    
    # Greedy diversity selection: iteratively add farthest from selected
    selected = list(elite_solutions)
    remaining_list = list(remaining)
    diversity_solutions = []
    
    for _ in range(min(n_diversity, len(remaining_list))):
        if len(remaining_list) == 0:
            break
        # Find point farthest from all currently selected
        dists = np.array([min(np.linalg.norm(p - s) for s in selected) 
                         if selected else np.linalg.norm(p) 
                         for p in remaining_list])
        farthest_idx = np.argmax(dists)
        diversity_solutions.append(remaining_list.pop(farthest_idx))
    
    if len(diversity_solutions) == 0:
        return elite_solutions
    
    return np.vstack([elite_solutions, np.array(diversity_solutions)])
```