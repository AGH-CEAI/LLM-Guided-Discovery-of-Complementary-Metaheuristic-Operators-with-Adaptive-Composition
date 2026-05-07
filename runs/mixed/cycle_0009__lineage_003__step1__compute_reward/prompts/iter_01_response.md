**Idea: Rank-Normalized Success Rate with Relative Improvement**
Uses rank-based ordering and success rate, normalized by current fitness magnitude for scale-invariance across diverse problem types.
```python
def _compute_reward(self, fitness, trial_fitness):
    """Compute rank-based reward with success rate and relative improvement"""
    # Guard against invalid values
    fitness = np.clip(fitness, -1e50, 1e50)
    trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

    improvements = fitness - trial_fitness
    n_improved = int(np.sum(improvements > 0))
    n_total = len(fitness)

    if n_total == 0:
        return 0.0

    # Success rate component (primary signal)
    success_rate = n_improved / float(n_total)

    # Relative improvement: normalize by current best fitness for scale invariance
    current_best = float(np.min(fitness))
    if abs(current_best) > 1e-10 and np.isfinite(current_best):
        # Relative improvement of best individual
        best_idx = int(np.argmin(fitness))
        rel_improvement = improvements[best_idx] / (abs(current_best) + 1.0)
    else:
        rel_improvement = 0.0

    # Rank-based component: use ordinal position change
    # Lower rank = better fitness
    rank_fitness = np.argsort(np.argsort(fitness))
    rank_trial = np.argsort(np.argsort(trial_fitness))
    rank_improvement = np.mean(rank_fitness - rank_trial) / float(n_total)

    # Combine components: success rate is primary, augmented by relative and rank signals
    reward = 0.6 * success_rate + 0.25 * np.clip(rel_improvement, -1.0, 1.0) + 0.15 * np.clip(rank_improvement, -1.0, 1.0)

    # Penalize complete failure more strongly
    if n_improved == 0:
        reward = -0.1

    return float(np.clip(reward, -1.0, 1.0))
```