Looking at the unsolved tasks, I see two distinct failure modes:

1. **Tasks 16, 19-23, 11, 13-15, 17-18** (errors ~1-100): These are likely multimodal/composite functions where the algorithm gets trapped in local optima. The greedy selection prevents any worsening moves, making escape impossible.

2. **Tasks 3, 6, 7, 9, 12** (errors ~1e-4 to 1e-8): These are near-convergence but stall, suggesting the population loses diversity too quickly.

The key insight: all previous variants use strictly greedy or near-greedy selection. For the worst tasks (stuck at errors of 1-100), we need to **accept worse solutions probabilistically** (like simulated annealing) to escape deep local optima, while also **maintaining population diversity** by preventing the population from collapsing. I'll combine Boltzmann-based acceptance of inferior trials with a crowding-based replacement scheme that preserves diversity.

**Idea: Simulated Annealing Acceptance with Crowding Distance Preservation**
Accept worse trials probabilistically using fitness-based Boltzmann criterion scaled by generation, and use crowding to maintain diversity for escaping local optima on multimodal functions.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
    """SA-like acceptance with crowding distance to escape local optima on multimodal functions."""
    n = min(len(population), len(trials))
    success_f = []
    success_cr = []
    success_delta = []

    generation = getattr(self, 'generation', 0)
    
    # Compute fitness range for temperature scaling
    fit_valid = fitness[~np.isnan(fitness)]
    if len(fit_valid) > 1:
        fit_range = np.max(fit_valid) - np.min(fit_valid)
    else:
        fit_range = 1.0
    fit_range = max(fit_range, 1e-30)
    
    # Temperature: starts high, decays slowly — allows worse solutions early on
    # Higher base temperature than typical SA to really enable exploration
    temperature = fit_range * 0.3 * np.exp(-generation / 150.0)
    temperature = max(temperature, 1e-30)

    for i in range(n):
        if not valid_mask[i]:
            continue
            
        delta = trial_fitness[i] - fitness[i]
        
        if delta <= 0:
            # Trial is better or equal — always accept
            if delta < 0:
                success_f.append(f_values[i])
                success_cr.append(cr_values[i])
                success_delta.append(abs(delta))
            
            # Archive old individual
            if len(archive) < self.archive_max:
                archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
            else:
                replace_idx = np.random.randint(0, self.archive_max)
                archive[replace_idx] = population[i]
            
            population[i] = trials[i]
            fitness[i] = trial_fitness[i]
        else:
            # Trial is worse — accept with Boltzmann probability
            # Scale delta by fitness range for normalization
            normalized_delta = delta / fit_range
            accept_prob = np.exp(-normalized_delta / (temperature / fit_range + 1e-30))
            accept_prob = min(accept_prob, 0.15)  # Cap acceptance of worse solutions
            
            if np.random.random() < accept_prob:
                # Accept the worse solution to escape local optima
                # Don't record as "success" for parameter adaptation
                # but do archive the old solution
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

    return (population, fitness, archive,
            np.array(success_f), np.array(success_cr), np.array(success_delta))
```