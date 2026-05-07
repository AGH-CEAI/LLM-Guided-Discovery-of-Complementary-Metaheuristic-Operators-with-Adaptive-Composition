Looking at the unsolved tasks, I see two critical failure patterns:
1. **Tasks 12 & 8** (errors ~2000-2500): Stuck at local optima — the archive likely contains solutions too clustered around poor regions
2. **Tasks 20, 21** (stuck at exactly 5.0): Deceptive local optima — all variants converge to same bad basin

The current archive update keeps the best solutions by fitness, which causes the archive to collapse into a single region. When this archive is used for mutation (as `r3`), it provides no diverse exploration vectors. For multimodal/deceptive tasks, we need an archive that *deliberately preserves diverse, spatially-separated solutions* even if they're not the absolute best.

**Idea: Crowding-Distance Archive Selection**
Instead of selecting by fitness alone, use crowding distance in objective space to protect isolated solutions. This is inspired by NSGA-II's diversity preservation — solutions far from their nearest neighbor in fitness space are protected because they're in sparse regions of the search landscape, meaning they're more likely to offer diverse mutation directions.

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
        
        # Crowding distance in objective space: protect solutions in sparse regions
        n = len(combined)
        crowding = np.zeros(n)
        
        for i in range(n):
            distances = np.abs(combined_fit - combined_fit[i])
            distances[i] = np.inf
            nearest_idx = np.argmin(distances)
            crowding[i] = distances[nearest_idx] if np.isfinite(distances[nearest_idx]) else 0.0
        
        # Normalize to [0, 1] with safe epsilon
        fit_range = combined_fit.max() - combined_fit.min()
        fit_norm = np.zeros(n) if fit_range <= 1e-15 else (combined_fit - combined_fit.min()) / fit_range
        
        crow_range = crowding.max() - crowding.min()
        crow_norm = np.zeros(n) if crow_range <= 1e-15 else (crowding - crowding.min()) / crow_range
        
        # Fitness-diversity tradeoff score (higher = better)
        score = 0.5 * (1.0 - fit_norm) + 0.5 * crow_norm
        
        keep_count = min(self.archive_max, n)
        keep_indices = np.argsort(score)[-keep_count:]
        
        return combined[keep_indices]
```