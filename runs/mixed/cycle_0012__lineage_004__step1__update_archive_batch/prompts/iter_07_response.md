Looking at the unsolved tasks, I see that Tasks 12 (error=2.56e+03) and 8 (error=2.03e+03) are catastrophically failing, while others are stuck at moderate errors (5-50). The current archive strategy is too fitness-focused and loses spatial diversity, causing the algorithm to get trapped in poor local optima on multimodal/ill-conditioned problems.

**Idea: Spatial Diversity Archive with Fitness-Diversity Pareto Selection**

Instead of simple fitness-based truncation, maintain diversity by partitioning the search space and selecting solutions that cover different regions, while still preferring better fitness. This ensures the archive provides truly diverse mutation targets for breaking out of local optima.

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
    
    n_total = len(combined)
    if n_total == 0:
        return np.array([]).reshape(0, self.dim)
    
    if n_total <= self.archive_max:
        return combined
    
    max_archive = self.archive_max
    if n_total <= max_archive * 2:
        top_indices = np.argsort(combined_fit)[:max_archive]
        return combined[top_indices]
    
    n_bins = min(max_archive, 20)
    bin_bounds = np.linspace(self.lower, self.upper, n_bins + 1)
    
    bin_assignments = np.zeros(n_total, dtype=int)
    for i in range(self.dim):
        bin_assignments += np.digitize(combined[:, i], bin_bounds) * (n_bins ** i)
    
    unique_bins = np.unique(bin_assignments)
    n_bins_used = len(unique_bins)
    
    fitness_scores = combined_fit.copy()
    fitness_min, fitness_max = np.min(fitness_scores), np.max(fitness_scores)
    if fitness_max > fitness_min:
        fitness_norm = (fitness_scores - fitness_min) / (fitness_max - fitness_min + 1e-12)
    else:
        fitness_norm = np.zeros_like(fitness_scores)
    
    diversity_scores = np.zeros(n_total)
    for i in range(n_total):
        distances = np.linalg.norm(combined - combined[i], axis=1)
        distances[i] = np.inf
        k = min(5, n_total - 1)
        knn_distances = np.partition(distances, k)[:k]
        diversity_scores[i] = np.mean(knn_distances)
    
    div_min, div_max = np.min(diversity_scores), np.max(diversity_scores)
    if div_max > div_min:
        diversity_norm = (diversity_scores - div_min) / (div_max - div_min + 1e-12)
    else:
        diversity_norm = np.zeros_like(diversity_scores)
    
    lambda_param = 0.7
    composite_scores = lambda_param * fitness_norm + (1 - lambda_param) * diversity_norm
    
    selected_indices = []
    bin_to_best = {}
    
    for idx in np.argsort(composite_scores):
        bin_id = bin_assignments[idx]
        if bin_id not in bin_to_best:
            bin_to_best[bin_id] = idx
            selected_indices.append(idx)
            if len(selected_indices) >= max_archive:
                break
    
    remaining_slots = max_archive - len(selected_indices)
    if remaining_slots > 0:
        remaining_candidates = [i for i in np.argsort(composite_scores) 
                                if i not in selected_indices][:remaining_slots]
        selected_indices.extend(remaining_candidates)
    
    return combined[selected_indices]
```