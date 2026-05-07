**Idea: Crowding-based Archive with Distance-aware Diversity Maintenance**

Replace fitness-only truncation with crowding-based selection that maintains both solution quality and spatial diversity. This addresses the worst unsolved tasks (12, 8, 21, 14, 23, 20) which appear stuck in local optima — the archive is likely discarding diverse but slightly suboptimal solutions that are critical for escaping these basins.

```python
def _update_archive_batch(self, archive, new_solutions, new_fitness):
    if len(new_solutions) == 0:
        return archive
    
    combined = np.vstack([archive, new_solutions])
    combined_fit = np.concatenate([
        np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
        new_fitness
    ])
    
    n = len(combined)
    if n == 0:
        return combined
    
    # Use stricter uniqueness (1e-5) to preserve more distinct solutions
    unique_mask = self._find_unique_solutions(combined, tol=1e-5)
    combined = combined[unique_mask]
    combined_fit = combined_fit[unique_mask]
    
    n_unique = len(combined)
    if n_unique == 0:
        return combined[:0] if self.archive_max == 0 else combined
    
    # If within limit, keep all unique solutions (prefer diversity over fitness pruning)
    if n_unique <= self.archive_max:
        return combined
    
    # Crowding-based selection: iteratively remove least diverse member
    # that is also not among the absolute best (preserve elites)
    selected = np.arange(n_unique)
    n_to_remove = n_unique - self.archive_max
    
    # Sort by fitness, mark top 10% as protected elites
    sorted_indices = np.argsort(combined_fit)
    elite_count = max(1, self.archive_max // 10)
    elite_indices = set(sorted_indices[:elite_count])
    
    for _ in range(n_to_remove):
        if len(selected) <= self.archive_max:
            break
        
        # Find non-elite candidate with minimum crowding distance
        min_crowding = np.inf
        victim_idx = -1
        
        for idx in selected:
            if idx in elite_indices:
                continue
            
            # Compute min distance to all other selected solutions
            if len(selected) > 1:
                other_indices = [i for i in selected if i != idx]
                diffs = combined[other_indices] - combined[idx]
                distances = np.sqrt(np.sum(diffs * diffs, axis=1))
                min_dist = np.min(distances)
            else:
                min_dist = np.inf
            
            # Prefer removing solutions that are both poor in fitness AND crowded
            fit_rank = np.sum(combined_fit[idx] > combined_fit[selected])
            crowding_score = min_dist / (fit_rank + 1)
            
            if crowding_score < min_crowding:
                min_crowding = crowding_score
                victim_idx = idx
        
        if victim_idx >= 0:
            selected = np.array([i for i in selected if i != victim_idx])
        else:
            # Fallback: remove worst fitness among non-elites
            worst_non_elite = None
            worst_fitness = -np.inf
            for idx in selected:
                if idx not in elite_indices and combined_fit[idx] > worst_fitness:
                    worst_fitness = combined_fit[idx]
                    worst_non_elite = idx
            if worst_non_elite is not None:
                selected = np.array([i for i in selected if i != worst_non_elite])
            break
    
    return combined[selected]
```