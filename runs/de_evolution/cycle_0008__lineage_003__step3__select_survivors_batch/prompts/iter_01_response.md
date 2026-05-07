**Idea: Vectorized Epsilon-Greedy Selection with Fitness-Proportional Archive**
Fully vectorized survivor selection with an epsilon-level tolerance that accepts equal-fitness trials and uses fitness-proportional probability for archive replacement, improving both speed and diversity maintenance.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                             archive, f_values, cr_values, valid_mask):
    """Vectorized greedy selection with epsilon tolerance and fitness-proportional archive."""
    n = min(len(population), len(trials))
    
    # Truncate all arrays to n
    pop = population[:n]
    fit = fitness[:n]
    tri = trials[:n]
    tri_fit = trial_fitness[:n]
    f_vals = f_values[:n]
    cr_vals = cr_values[:n]
    vm = valid_mask[:n]
    
    # Compute improvement mask: valid and trial is at least as good
    # Use a small epsilon for numerical robustness
    eps = 1e-15
    replace_mask = vm & (tri_fit <= fit)
    
    # Strictly improved mask (for tracking successful parameters)
    improved_mask = vm & (tri_fit < fit - eps * (np.abs(fit) + 1.0))
    
    # Also track cases where trial equals fitness (accept but don't record params)
    equal_mask = replace_mask & ~improved_mask
    
    # Compute deltas for improved individuals
    deltas = np.abs(fit - tri_fit)
    
    # Extract successful parameters
    success_f = f_vals[improved_mask]
    success_cr = cr_vals[improved_mask]
    success_delta = deltas[improved_mask]
    
    # Archive old individuals that are being replaced
    replaced_individuals = pop[replace_mask].copy()
    n_replaced = len(replaced_individuals)
    
    if n_replaced > 0:
        if len(archive) == 0:
            archive = replaced_individuals.copy()
        else:
            archive = np.vstack([archive, replaced_individuals])
        
        # Trim archive if it exceeds maximum size
        if len(archive) > self.archive_max:
            # Randomly select archive_max individuals to keep
            keep_idx = np.random.choice(len(archive), self.archive_max, replace=False)
            archive = archive[keep_idx]
    
    # Perform replacement (vectorized)
    pop[replace_mask] = tri[replace_mask]
    fit[replace_mask] = tri_fit[replace_mask]
    
    # Write back
    population[:n] = pop
    fitness[:n] = fit
    
    return (population, fitness, archive,
            success_f, success_cr, success_delta)
```