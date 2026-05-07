Looking at the unsolved tasks, the worst ones (Tasks 16, 23, 20, 19, 21) have errors in the range 1e+0 to 1e+2, suggesting the algorithm is getting trapped in local optima on multimodal/compositional functions. The current greedy selection is too conservative — it only accepts improvements, which causes premature convergence on deceptive landscapes.

My key insight: implement a **crowding-based selection with fitness sharing and occasional acceptance of worse solutions** (simulated annealing-style). This maintains population diversity to escape local optima basins, which is critical for the high-dimensional multimodal functions (Tasks 11-23). The mechanism:
1. Sometimes accept slightly worse trials (with probability decaying over generations) to escape local traps
2. Use crowding distance to preserve diversity — if a trial is in a sparse region, give it preference even if slightly worse
3. Archive management that tracks diverse solutions rather than just replaced ones

This is fundamentally different from the original (pure greedy) and variant_01 because it deliberately sacrifices short-term fitness for long-term exploration capacity.

**Idea: Crowding Diversity Preserving Selection with SA Acceptance**
Accept worse trials probabilistically based on generation-adaptive temperature and crowding distance to maintain diversity and escape local optima on multimodal functions.
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
    """Selection with probabilistic acceptance of worse solutions and crowding-based diversity."""
    n = min(len(population), len(trials))
    success_f = []
    success_cr = []
    success_delta = []

    # Adaptive temperature for accepting worse solutions
    generation = getattr(self, 'generation', 0)
    # Start with moderate acceptance, decay over time but never fully zero
    T = max(1e-10, 1.0 * np.exp(-generation / 100.0))
    
    # Compute pairwise distances for crowding (to nearest neighbor)
    from scipy.spatial.distance import cdist
    try:
        dists = cdist(population[:n], population[:n])
        np.fill_diagonal(dists, np.inf)
        crowd_dist = np.min(dists, axis=1)
        # Normalize crowding distances
        cd_max = np.max(crowd_dist) + 1e-30
        crowd_scores = crowd_dist / cd_max  # 0=crowded, 1=isolated
    except Exception:
        crowd_scores = np.ones(n) * 0.5

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
            population[i] = trials[i].copy()
            fitness[i] = trial_fitness[i]
        else:
            # Trial is worse — accept probabilistically
            # Higher acceptance for crowded individuals (low crowd_score means crowded)
            # Scale delta by fitness range for normalization
            fit_range = np.max(fitness[:n]) - np.min(fitness[:n]) + 1e-30
            normalized_delta = delta / fit_range
            
            # Crowded individuals more likely to accept worse (escape local basin)
            crowding_boost = 1.0 + 2.0 * (1.0 - crowd_scores[i])  # boost for crowded
            
            accept_prob = np.exp(-normalized_delta / (T * crowding_boost + 1e-30))
            accept_prob = min(accept_prob, 0.3)  # cap at 30% to avoid too much randomness
            
            if np.random.random() < accept_prob:
                # Archive old individual before replacement
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i].copy()
                fitness[i] = trial_fitness[i]
                # Don't record as success for parameter adaptation

    return (population, fitness, archive,
            np.array(success_f), np.array(success_cr), np.array(success_delta))
```