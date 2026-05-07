Looking at the unsolved tasks, I see two distinct failure modes:
1. **Tasks 12, 8** (errors ~10³): Extremely stuck in poor local optima - the algorithm is failing to explore effectively
2. **Tasks 21, 14, 20, 23** (errors ~10⁰): Multi-modal problems where current archive keeps similar solutions and loses diversity
3. **Tasks 19, 22** (errors ~10⁻⁴ to 10⁻⁶): Almost solved - need finer-grained diversity

**Root cause**: The current archive only tracks fitness and keeps the best solutions, but for multimodal/deceptive tasks, this causes rapid loss of diversity. Once stuck in a local basin, the archive has no coverage of other promising regions.

**Key insight**: Replace fitness-only ranking with **fitness-diversity dual ranking** - select solutions that are both high-quality AND spatially distributed across the search space. This ensures the archive maintains exploration potential for escaping local optima.

**Idea: Fitness-Distance Composite Ranking Archive**

```python
def _update_archive_batch(self, archive, new_solutions, new_fitness):
    if len(new_solutions) == 0:
        return archive
    
    # Combine archive and new solutions
    if len(archive) > 0:
        combined = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([np.full(len(archive), -np.inf), new_fitness])
    else:
        combined = new_solutions
        combined_fit = new_fitness.copy()
    
    # Remove near-duplicates (keep the better one)
    unique_mask = self._find_unique_solutions(combined)
    combined = combined[unique_mask]
    combined_fit = combined_fit[unique_mask]
    
    n = len(combined)
    if n <= self.archive_max:
        return combined
    
    # Compute minimum distance to nearest neighbor for diversity metric
    n_combined = len(combined)
    min_dists = np.full(n_combined, np.inf)
    
    # Vectorized distance computation (batch for efficiency)
    chunk_size = 500
    for i in range(0, n_combined, chunk_size):
        end_i = min(i + chunk_size, n_combined)
        chunk = combined[i:end_i]  # (chunk_size, dim)
        # Compute distances to all solutions
        diffs = combined[np.newaxis, :, :] - chunk[:, np.newaxis, :]  # (chunk, n, dim)
        dists = np.linalg.norm(diffs, axis=2)  # (chunk, n)
        np.fill_diagonal(dists[i:i+chunk_size, :][:end_i-i], np.inf)  # Mask self-distances
        min_dists[i:end_i] = np.min(dists, axis=1)
    
    # Normalize fitness (lower is better, so negate for proper scoring)
    fit_min, fit_max = np.min(combined_fit), np.max(combined_fit)
    fit_range = fit_max - fit_min + 1e-15
    fitness_score = 1.0 - (combined_fit - fit_min) / fit_range  # Higher = better
    
    # Normalize diversity (higher distance = better diversity)
    dist_max = np.max(min_dists) + 1e-15
    diversity_score = min_dists / dist_max
    
    # Composite: balance fitness (70%) and diversity (30%)
    # For hard problems, diversity helps escape local optima
    composite = 0.7 * fitness_score + 0.3 * diversity_score
    
    # Select top archive_max by composite score
    top_indices = np.argsort(composite)[-self.archive_max:]
    
    return combined[top_indices]
```