Looking at the data, I see a clear pattern: the WORST unsolved tasks (12, 16, 8, 21, 23, 13, 14, 20) are stuck at error levels of 10^0 to 10^3, many at exact values like 5.0 or 50.0. This strongly suggests **deceptive local optima traps** where the algorithm is stuck in basins it cannot escape.

**Critical observation:** The current reward computes `total_improvement / n_improved` — an **average** metric. For tasks where the population is trapped in a local optimum, individual improvements may be small and noisy, causing the reward signal to be misleading. This causes Thompson Sampling to select suboptimal operators.

**New strategy: Best-Individual-Focused + Stagnation-Aware Reward**
- Focus on the **single best individual's improvement** (not average) — this is the signal that matters for escaping traps
- Add a **rank-based component** for robustness (less sensitive to outlier values)
- Apply an **explicit stagnation penalty** when the best doesn't improve — this directly addresses the deceptive trap problem

This is fundamentally different because it shifts from population-average to **best-individual** focus, which is the correct signal for operator selection when trapped.

```python
def _compute_reward(self, fitness, trial_fitness):
    """Compute reward based on best individual improvement + rank-based robustness"""
    # Guard against invalid values
    fitness = np.clip(fitness, -1e50, 1e50)
    trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

    # Track improvement of best individual (primary signal for escaping traps)
    best_f = float(np.min(fitness))
    best_trial_f = float(np.min(trial_fitness))
    best_improvement = best_f - best_trial_f

    # Rank-based component: more robust to outliers than raw values
    fitness_sorted_idx = np.argsort(fitness)
    trial_sorted_idx = np.argsort(trial_fitness)
    best_rank = float(np.argmin(fitness_sorted_idx == np.argmin(fitness)))
    trial_best_rank = float(np.argmin(trial_sorted_idx == np.argmin(trial_fitness)))
    
    # Normalize ranks to [0, 1] range
    max_rank = max(float(len(fitness) - 1), 1.0)
    rank_component = 0.3 * (1.0 - (trial_best_rank / max_rank))

    # Stagnation detection: if best didn't improve, apply strong penalty
    if best_improvement <= 0:
        # Stagnation penalty (more aggressive to break out of traps)
        reward = -0.15
    else:
        # Progressive reward based on best improvement magnitude
        # Scale by population size for dimension-invariance
        scale_factor = max(float(len(fitness)), 1.0)
        improvement_component = 0.7 * min(float(best_improvement) / (scale_factor + 1e-10), 1.0)
        reward = improvement_component + rank_component

    return float(np.clip(reward, -1.0, 1.0))
```