**Idea: Pairwise Distance Geometric Survivor Selection**
Greedy selection with geometric diversity maintenance using pairwise distances: accept fitness improvements unconditionally, then for non-improving trials, compute average inter-individual distances and accept trials that maintain greater geometric separation from their corresponding population members.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection with geometric diversity maintenance via pairwise distances."""
    improved_mask = trial_fitness < fitness
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    # Accept fitness improvements unconditionally
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    # For non-improving trials, apply geometric diversity selection
    not_improved_mask = ~improved_mask
    if np.any(not_improved_mask):
        # Compute average pairwise distances within population
        # This establishes the characteristic geometric scale
        all_pairwise = np.linalg.norm(
            population[:, np.newaxis, :] - population[np.newaxis, :, :],
            axis=2
        )
        np.fill_diagonal(all_pairwise, np.inf)
        avg_dist_to_others = np.mean(all_pairwise, axis=1)
        
        # For non-improving trials, compute distance to corresponding population members
        trial_to_member_dist = np.linalg.norm(
            trials[not_improved_mask] - population[not_improved_mask],
            axis=1
        )
        
        # Accept non-improving trials that maintain better geometric separation
        # Accept if: distance(trial, member) > avg distance(member, others)
        geometric_acceptance = trial_to_member_dist > avg_dist_to_others[not_improved_mask]
        
        indices_to_update = np.where(not_improved_mask)[0][geometric_acceptance]
        new_population[indices_to_update] = trials[indices_to_update]
        new_fitness[indices_to_update] = trial_fitness[indices_to_update]
    
    return new_population, new_fitness, improved_mask
```