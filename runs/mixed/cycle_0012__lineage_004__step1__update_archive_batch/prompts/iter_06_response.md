**Idea: Fitness-Distance Clustering Archive**

Replace pure fitness-ranking with a clustering-based selection that preserves diverse basins of attraction. For each candidate, compute both Euclidean distance to centroid AND fitness rank, then greedily select archive members that maximize coverage in (distance, fitness) space. This helps escape local optima (Tasks 20, 21 stuck at 5.0) and explore multimodal landscapes (Tasks 12, 8 with errors ~1000x larger).

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
    
    # Fitness-distance clustering selection
    n_keep = self.archive_max
    n_combined = len(combined)
    
    # Compute normalized fitness (lower is better, 0=best)
    fit_min, fit_max = combined_fit.min(), combined_fit.max()
    fit_range = fit_max - fit_min + 1e-20
    norm_fitness = (combined_fit - fit_min) / fit_range
    
    # Compute centroid of combined population
    centroid = np.mean(combined, axis=0)
    
    # Compute distance from centroid (normalized)
    distances = np.linalg.norm(combined - centroid, axis=1)
    dist_max = distances.max() + 1e-20
    norm_dist = distances / dist_max
    
    # Combined score: balance fitness (lower is better) and diversity (higher distance is better)
    # Use sqrt to soften dominance of either factor
    scores = np.sqrt(norm_fitness ** 2 + (1 - norm_dist) ** 2)
    
    # Greedy selection: pick best score, then iteratively pick furthest in (fitness, dist) space
    selected_indices = []
    remaining = np.arange(n_combined)
    
    # Start with best fitness individual
    best_idx = np.argmin(combined_fit)
    selected_indices.append(best_idx)
    remaining = np.delete(remaining, np.where(remaining == best_idx)[0])
    
    # Greedily add remaining, prioritizing those farthest from already selected
    while len(selected_indices) < n_keep and len(remaining) > 0:
        # Compute min distance from each remaining to any selected
        selected_arr = np.array(selected_indices)
        min_dists = np.zeros(len(remaining))
        for i, rem_idx in enumerate(remaining):
            rem_point = combined[rem_idx]
            rem_fit = combined_fit[rem_idx]
            # Distance in solution space
            sol_dists = np.linalg.norm(combined[selected_arr] - rem_point, axis=1)
            # Distance in fitness space (normalized)
            fit_diffs = np.abs(combined_fit[selected_arr] - rem_fit) / (fit_range + 1e-20)
            # Combined metric
            min_dists[i] = np.sqrt(np.min(sol_dists) ** 2 + np.min(fit_diffs) ** 2)
        
        # Pick the one maximizing distance (exploration) weighted by score (quality)
        # Blend: prefer diverse AND reasonably good
        weighted = min_dists * (1.0 + scores[remaining])
        pick_pos = np.argmax(weighted)
        pick_idx = remaining[pick_pos]
        selected_indices.append(pick_idx)
        remaining = np.delete(remaining, pick_pos)
    
    return combined[np.array(selected_indices)]
```