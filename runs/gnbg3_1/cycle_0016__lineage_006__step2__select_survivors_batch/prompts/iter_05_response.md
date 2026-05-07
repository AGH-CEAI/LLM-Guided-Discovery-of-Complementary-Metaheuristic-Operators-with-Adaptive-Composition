Looking at the problem:
- **Worst tasks (17, 16, 23, 20, 11, 12, 19, 18)**: Errors stuck at 10-40, indicating severe local optima trapping
- **Root cause**: Elitist selection greedily keeps only best-fit individuals, destroying diversity and preventing escape from local basins
- **Key insight**: All 4 prior variants modified covariance adaptation, not survivor selection — a fundamentally different selection pressure is needed

The current `(μ, λ)` selection is purely fitness-based, which causes premature convergence on multimodal/deceptive tasks. I propose **diversity-maintaining survivor selection** that greedily builds a diverse subset from the combined population, explicitly preventing the population from collapsing onto local optima.

**Idea: Greedy Spatial Diversity Selection**
Select survivors by greedily choosing individuals that maximize minimum distance to already-selected survivors, while still favoring fitter individuals. This maintains exploration pressure to escape local optima on the hardest tasks.
```python
def _select_survivors_batch(self):
    """Select survivors maintaining spatial diversity to escape local optima."""
    combined_pop = np.vstack([self.population, self.trials])
    combined_fit = np.concatenate([self.fitness, self.trial_fitness])
    
    # Sort by fitness
    sorted_indices = np.argsort(combined_fit)
    sorted_pop = combined_pop[sorted_indices]
    sorted_fit = combined_fit[sorted_indices]
    
    n_candidates = min(2 * self.NP, len(sorted_pop))
    n_to_select = min(self.NP, n_candidates)
    
    if n_to_select >= n_candidates:
        self.population = sorted_pop[:self.NP]
        self.fitness = sorted_fit[:self.NP]
    else:
        # Compute pairwise distances among top candidates
        candidate_pop = sorted_pop[:n_candidates]
        candidate_fit = sorted_fit[:n_candidates]
        
        # Compute normalized distances (clip to avoid overflow)
        lb, ub = self.lb, self.ub
        range_scale = np.maximum(ub - lb, 1e-10)
        
        # Efficient batch distance computation
        norm_pop = (candidate_pop - lb) / range_scale
        diffs = norm_pop[:, np.newaxis, :] - norm_pop[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Greedy diverse selection
        selected = [0]  # Start with best individual
        remaining = set(range(1, n_candidates))
        
        # Precompute diversity weight based on generation
        diversity_weight = np.clip(1.0 / (1.0 + np.log1p(self.generation + 1)), 0.1, 1.0)
        
        for _ in range(n_to_select - 1):
            if not remaining:
                break
                
            best_idx = None
            best_score = -np.inf
            
            for idx in remaining:
                # Minimum distance to already selected
                min_dist = min(dist_matrix[idx, selected])
                
                # Fitness rank (0 = best)
                fitness_bonus = 1.0 / (1.0 + sorted_fit[sorted_indices[idx]])
                
                # Score: balance diversity (min_dist) with fitness
                score = diversity_weight * min_dist + (1.0 - diversity_weight) * fitness_bonus
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        self.population = candidate_pop[selected]
        self.fitness = candidate_fit[selected]
    
    self.old_mean = self.mean.copy()
    self.mean = np.mean(self.population, axis=0)
    self.generation += 1
```