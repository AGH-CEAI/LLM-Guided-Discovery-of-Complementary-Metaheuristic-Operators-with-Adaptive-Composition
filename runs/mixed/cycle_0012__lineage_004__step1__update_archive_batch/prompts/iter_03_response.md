**Idea: Diversity-First Archive with Fitness-Distance Pareto Selection**
Replaces fitness-only selection with a two-objective ranking (fitness + distance to archive), prioritizing unique solutions that help escape local optima on multi-modal tasks.

```python
def _update_archive_batch(self, archive, new_solutions, new_fitness):
    if len(new_solutions) == 0:
        return archive
    
    combined = np.vstack([archive, new_solutions])
    combined_fit = np.concatenate([
        np.full(len(archive), -np.inf) if len(archive) > 0 else np.array([]),
        new_fitness
    ])
    
    unique_mask = self._find_unique_solutions(combined)
    combined = combined[unique_mask]
    combined_fit = combined_fit[unique_mask]
    
    if len(combined) <= self.archive_max:
        return combined
    
    # Fitness-distance Pareto selection: prefer solutions that are
    # both good AND different from existing archive members
    n_keep = self.archive_max
    sorted_indices = np.argsort(combined_fit)
    
    selected = []
    selected_vectors = []
    
    for idx in sorted_indices:
        if len(selected) >= n_keep:
            break
        
        if len(selected_vectors) == 0:
            selected.append(idx)
            selected_vectors.append(combined[idx])
            continue
        
        # Compute min distance to already-selected solutions
        diffs = combined[idx] - np.array(selected_vectors)
        min_dist = np.min(np.sqrt(np.sum(diffs**2, axis=1)))
        
        # Keep if sufficiently different (diversity threshold scales with dim)
        dist_threshold = 1e-2 * self.dim + 1e-3
        if min_dist >= dist_threshold or len(selected) < n_keep // 3:
            selected.append(idx)
            selected_vectors.append(combined[idx])
    
    # If still not full, add best remaining by fitness
    if len(selected) < n_keep:
        remaining = [i for i in sorted_indices if i not in selected]
        needed = n_keep - len(selected)
        selected.extend(remaining[:needed])
    
    return combined[selected]
```