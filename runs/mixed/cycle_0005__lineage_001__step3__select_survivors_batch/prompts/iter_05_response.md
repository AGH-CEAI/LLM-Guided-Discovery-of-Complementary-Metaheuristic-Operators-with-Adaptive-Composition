**Idea: Fitness-Diversity Hybrid Survivor Selection**

One-line description: Replace pure greedy selection with a hybrid that balances fitness improvement against population diversity contribution, using adaptive weighting that increases diversity pressure when stagnant.
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    # Compute diversity contribution: average distance to other population members
    def diversity_contribution(solution, pop):
        if len(pop) < 2:
            return 0.0
        dists = np.linalg.norm(pop - solution, axis=1)
        return float(np.mean(dists))

    # Build candidate pool: current vs trial for each individual
    candidates_pop = np.vstack([population, trials])
    candidates_fit = np.concatenate([fitness, trial_fitness])
    is_trial = np.concatenate([np.zeros(len(population), dtype=bool),
                               np.ones(len(trials), dtype=bool)])

    # Compute diversity contributions for all candidates
    div_contribs = np.array([diversity_contribution(candidates_pop[i], candidates_pop)
                             for i in range(len(candidates_pop))])

    # Normalize diversity contributions
    div_min, div_max = np.min(div_contribs), np.max(div_contribs)
    if div_max > div_min + 1e-10:
        div_normalized = (div_contribs - div_min) / (div_max - div_min)
    else:
        div_normalized = np.ones_like(div_contribs) * 0.5

    # Compute fitness scores (lower fitness = better)
    fit_min, fit_max = np.min(candidates_fit), np.max(candidates_fit)
    if fit_max > fit_min + 1e-10:
        fitness_score = 1.0 - (candidates_fit - fit_min) / (fit_max - fit_min)
    else:
        fitness_score = np.ones_like(candidates_fit) * 0.5

    # Adaptive diversity weight: increases when stagnant
    diversity_weight = min(0.4, 0.1 + 0.02 * self.stagnation_count)

    # Combined score: balance fitness vs diversity
    combined_score = (1.0 - diversity_weight) * fitness_score + diversity_weight * div_normalized

    # Select: for each pair (original, trial), pick the one with higher combined score
    n = len(population)
    new_population = np.empty_like(population)
    new_fitness = np.empty_like(fitness)

    for i in range(n):
        orig_idx = i
        trial_idx = n + i
        if combined_score[trial_idx] > combined_score[orig_idx]:
            new_population[i] = candidates_pop[trial_idx]
            new_fitness[i] = candidates_fit[trial_idx]
        else:
            new_population[i] = candidates_pop[orig_idx]
            new_fitness[i] = candidates_fit[orig_idx]

    return new_population, new_fitness
```