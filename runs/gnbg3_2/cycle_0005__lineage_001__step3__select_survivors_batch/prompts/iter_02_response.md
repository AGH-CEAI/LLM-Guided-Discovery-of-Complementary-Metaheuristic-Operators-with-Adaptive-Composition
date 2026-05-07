**Idea: Spectral-Effective-Rank Survivor Selection**
Uses SVD-based effective rank of the population covariance matrix to detect when the population is collapsing into a low-dimensional subspace, then diversifies selection to expand the effective dimensionality. Specifically targets the worst unsolved tasks (17, 16, 11, 6) which likely suffer from ill-conditioning and subspace collapse.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Greedy selection with spectral diversity maintenance via SVD effective rank."""
    improved_mask = trial_fitness < fitness
    
    # Compute SVD-based effective rank of population (spectral measure of diversity)
    pop_centered = population - np.mean(population, axis=0)
    _, s_values, _ = np.linalg.svd(pop_centered, full_matrices=False)
    total_var = np.sum(s_values) + 1e-15
    variance_fractions = s_values / total_var
    effective_rank = np.exp(-np.sum(variance_fractions * np.log(variance_fractions + 1e-15)))
    
    # If population is collapsing (low effective rank), be diversity-seeking
    if effective_rank < self.dim * 0.3:
        for i in range(len(population)):
            if not improved_mask[i]:
                # Compute effective rank if this trial were added
                candidate_pop = np.vstack([pop_centered, trials[i] - np.mean(population, axis=0)])
                _, s_vals_cand, _ = np.linalg.svd(candidate_pop, full_matrices=False)
                total_var_cand = np.sum(s_vals_cand) + 1e-15
                var_frac_cand = s_vals_cand / total_var_cand
                cand_eff_rank = np.exp(-np.sum(var_frac_cand * np.log(var_frac_cand + 1e-15)))
                
                # Accept if it substantially increases effective dimensionality
                if cand_eff_rank > effective_rank + 0.1:
                    improved_mask[i] = True
    
    # Standard greedy selection for improved individuals
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    return new_population, new_fitness, improved_mask
```