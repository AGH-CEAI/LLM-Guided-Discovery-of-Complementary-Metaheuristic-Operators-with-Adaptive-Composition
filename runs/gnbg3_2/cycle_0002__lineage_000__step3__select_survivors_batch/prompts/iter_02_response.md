**Idea: SVD Condition-Number-Guided Survivor Selection**

Use singular value decomposition of the population matrix to compute its condition number. When conditioning is poor (high condition number), the population has collapsed to a low-dimensional subspace. Select survivors that project strongly onto minor singular vectors to restore effective dimensionality, preventing premature convergence on the worst-performing tasks.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Spectral selection: use SVD condition number to preserve exploration subspace."""
    improved_mask = trial_fitness < fitness
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    # Apply greedy selection first
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    # Compute condition number via SVD of population matrix
    centered = new_population - np.mean(new_population, axis=0)
    
    try:
        # SVD gives us the effective dimensionality of the search distribution
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Condition number: ratio of largest to smallest singular value
        # High condition number = population collapsed to a subspace
        eps = 1e-12
        cond_number = s[0] / max(s[-1], eps)
        
        # Threshold for detecting subspace collapse (ill-conditioning)
        cond_threshold = 50.0
        
        if cond_number > cond_threshold and len(trials) > 0:
            # Identify minor singular directions (where population has little variance)
            minor_threshold_idx = max(0, len(s) // 3)  # Bottom third of singular values
            minor_directions = Vt[minor_threshold_idx:]  # Shape: (k, dim), k = num minor dirs
            
            if minor_directions.shape[0] > 0:
                # Project all trial vectors onto minor singular directions
                trial_projections = trials @ minor_directions.T  # (np, k)
                trial_minor_energy = np.sum(trial_projections ** 2, axis=1)  # (np,)
                
                # Also compute current population projections for comparison
                current_projections = new_population @ minor_directions.T
                current_minor_energy = np.sum(current_projections ** 2, axis=1)
                
                # Find trials that add MORE minor-direction energy than their replaced counterparts
                improvement_in_minor = trial_minor_energy - current_minor_energy
                
                # Select up to 2 candidates that maximally restore minor-direction spread
                # Only consider those that were NOT already selected (to add new diversity)
                not_improved = ~improved_mask
                if np.any(not_improved):
                    candidates = np.where(not_improved)[0]
                    candidate_improvement = improvement_in_minor[candidates]
                    
                    # Select top candidates that expand the minor subspace
                    n_select = min(2, len(candidates))
                    top_indices = candidates[np.argsort(candidate_improvement)[-n_select:]]
                    
                    # Only replace if they genuinely expand the subspace
                    for idx in top_indices:
                        if improvement_in_minor[idx] > 0.01 * np.mean(s):
                            new_population[idx] = trials[idx]
                            new_fitness[idx] = trial_fitness[idx]
    
    except Exception:
        # Fallback: just return greedy selection on any numerical issue
        pass
    
    return new_population, new_fitness, improved_mask
```