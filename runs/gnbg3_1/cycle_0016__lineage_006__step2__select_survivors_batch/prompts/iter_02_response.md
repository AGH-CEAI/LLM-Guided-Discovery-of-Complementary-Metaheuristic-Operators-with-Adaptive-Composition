**Idea: Fitness-Diversity Balanced Survivor Selection**
Replace pure elitist (μ+λ) selection with a two-phase approach that preserves top fitness while injecting diversity. Select NP/2 best by fitness, then fill remaining slots with diverse representatives from the remaining pool using distance-based sampling. This directly counteracts the premature convergence that causes errors ~10-80 on the worst tasks.
```python
def _select_survivors_batch(self):
    """Select survivors via fitness-diversity balanced (mu, lambda)-selection."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    n_combined = len(combined_fit)
    if n_combined < self.NP:
        self.population = combined_pop
        self.fitness = combined_fit
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
        return
    
    sorted_indices = np.argsort(combined_fit)
    
    # Phase 1: Select top NP/2 by fitness (preserve elite)
    elite_count = max(1, self.NP // 2)
    selected = sorted_indices[:elite_count]
    selected_set = set(selected)
    
    # Phase 2: Fill remaining with diverse candidates from remaining pool
    remaining_indices = [i for i in sorted_indices[elite_count:] if i not in selected_set]
    remaining_count = self.NP - elite_count
    
    if remaining_count > 0 and len(remaining_indices) > 0:
        remaining_pop = combined_pop[remaining_indices]
        remaining_fit = combined_fit[remaining_indices]
        
        # Distance-based diversity: pick centroid candidate first, then farthest from selected
        centroid = np.mean(remaining_pop, axis=0)
        dists = np.linalg.norm(remaining_pop - centroid, axis=1)
        diversity_candidates = np.argsort(dists)[::-1]
        
        for cand_idx in diversity_candidates:
            if len(selected) >= self.NP:
                break
            actual_idx = remaining_indices[cand_idx]
            if actual_idx not in selected_set:
                selected = np.append(selected, actual_idx)
                selected_set.add(actual_idx)
    
    self.population = combined_pop[selected]
    self.fitness = combined_fit[selected]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```