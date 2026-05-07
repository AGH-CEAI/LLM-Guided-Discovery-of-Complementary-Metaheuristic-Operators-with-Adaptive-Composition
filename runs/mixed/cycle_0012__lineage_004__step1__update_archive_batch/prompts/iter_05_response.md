**Idea: Fitness-Distance Ranked Diversity Archive**

Replaces pure fitness-based selection with a niching approach that explicitly rewards solutions from underexplored regions. Computes crowding distance in solution space and ranks candidates by balancing fitness quality with distance from existing archive members. This forces the archive to maintain basin coverage rather than collapsing to a single region—critical for the multimodal tasks (12, 8, 21) currently stuck at large errors.

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
    
    n_new = len(new_solutions)
    n_arch = len(combined) - n_new
    
    if n_arch > 0:
        arch_fitness = combined_fit[:n_arch]
        arch_best_mask = np.zeros(len(combined), dtype=bool)
        arch_best_count = min(self.archive_max // 4, n_arch)
        arch_best_indices = np.argsort(arch_fitness)[-arch_best_count:]
        arch_best_mask[arch_best_indices] = True
    
    if n_new > 0:
        new_solutions_trimmed = combined[n_arch:]
        new_fitness_trimmed = combined_fit[n_arch:]
        
        if n_arch > 0 and arch_best_count > 0:
            arch_best_sols = combined[arch_best_indices]
            distances = np.linalg.norm(
                new_solutions_trimmed[:, np.newaxis, :] - arch_best_sols[np.newaxis, :, :],
                axis=2
            )
            min_distances = np.maximum(np.min(distances, axis=1), 1e-10)
            fitness_range = max(np.max(new_fitness_trimmed) - np.min(new_fitness_trimmed), 1e-10)
            norm_fitness = (new_fitness_trimmed - np.min(new_fitness_trimmed)) / fitness_range
            diversity_scores = 0.4 * norm_fitness + 0.6 * (min_distances / (min_distances + 1.0))
        else:
            fitness_range = max(np.max(new_fitness_trimmed) - np.min(new_fitness_trimmed), 1e-10)
            diversity_scores = (new_fitness_trimmed - np.min(new_fitness_trimmed)) / fitness_range
        
        new_best_count = self.archive_max - arch_best_count
        if n_new <= new_best_count:
            new_best_mask = np.zeros(len(combined), dtype=bool)
            new_best_mask[n_arch:] = True
        else:
            new_best_indices = np.argsort(diversity_scores)[-new_best_count:]
            new_best_mask = np.zeros(len(combined), dtype=bool)
            new_best_mask[n_arch + new_best_indices] = True
    
    if n_arch > 0:
        selected_mask = arch_best_mask
    else:
        selected_mask = np.zeros(len(combined), dtype=bool)
    
    if n_new > 0:
        selected_mask = np.logical_or(selected_mask, new_best_mask)
    
    return combined[selected_mask]
```