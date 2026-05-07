**Idea: Spatial-Diversity Archive with Replacement and Capacity-Aware Pruning**

A fundamentally different strategy that prioritizes **solution diversity** over pure fitness ranking. Instead of partitioning archive slots by source (new vs. old), this approach compares every new solution against all archive members — replacing the *most similar worse* archive member when a new solution is both better-fitness AND spatially distant (novel). When over capacity, it removes the most similar pair rather than the worst-fitness, ensuring the archive maintains broad genotypic coverage. This directly addresses the worst tasks (12, 8) where the algorithm appears stuck in isolated basins ~11 decades from target — the current fitness-only archive cannot inject diversity needed to escape.

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
    
    if len(combined) > self.archive_max:
        top_keep = min(self.archive_max // 2, len(new_solutions))
        arch_keep = self.archive_max - top_keep
        
        new_sorted = np.argsort(new_fitness)[:top_keep]
        arch_sorted = np.argsort(combined_fit[:len(archive)])[-arch_keep:] if len(archive) > 0 else []
        
        selected = np.concatenate([new_sorted, arch_sorted])
        return combined[selected]
    
    return combined
```