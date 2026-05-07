Looking at the data, I see a clear bimodal failure pattern:
- **High-error tasks (12, 16, 8, 21, 23, 14, 20, 13, 5, 4, 1)**: Stuck at error 10^-1 to 10^+3 — the algorithm is making almost no progress toward the global optimum
- **Near-target tasks (18, 10, 15, 17, 3)**: Already at ~1e-08 to ~3e-08 — minor refinements needed

The current reward function computes `total_improvement / n_improved`, which is an absolute improvement metric. This is fundamentally flawed because:
1. For high-error tasks, a tiny absolute improvement (e.g., 0.01 on error 100) gets the same reward as a huge improvement
2. It doesn't distinguish between progress toward the global optimum vs. local improvements in the wrong basin
3. It doesn't adapt to the convergence stage

**Idea: Rank-Normalized Adaptive Reward with Target-Distance Scaling**

This approach fundamentally changes the reward computation by:
1. **Rank-normalizing**: Converting absolute fitness to percentile ranks within the population, making the reward invariant to fitness landscape scale
2. **Target-distance scaling**: Amplifying rewards when far from target (encourage exploration), fine-grained rewards when near target (encourage exploitation)
3. **Adaptive blending**: Combines rank quality with directional improvement signal, with weights shifting based on convergence stage

```python
def _compute_reward(self, fitness, trial_fitness):
    """Compute reward using rank-normalized adaptive signal with target-distance scaling"""
    # Guard against invalid values
    fitness = np.clip(fitness, -1e50, 1e50)
    trial_fitness = np.clip(trial_fitness, -1e50, 1e50)
    
    # Combine current and trial fitness for ranking
    n = len(fitness)
    combined = np.concatenate([fitness, trial_fitness])
    
    # Compute percentile ranks (0 = worst, 1 = best)
    # Using argsort-based ranking for numerical stability
    order = np.argsort(combined, kind='quicksort')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(2 * n) / float(2 * n - 1)
    
    trial_ranks = ranks[n:]  # Last n are trial_fitness
    orig_ranks = ranks[:n]   # First n are fitness
    
    # Rank improvement: how much better is trial vs original?
    rank_improvement = trial_ranks - orig_ranks
    
    # Per-individual improvement magnitude (normalized by population spread)
    pop_min = np.min(fitness)
    pop_max = np.max(fitness)
    pop_range = pop_max - pop_min + 1e-10
    
    # Normalized improvement: positive if trial is better
    norm_improvement = (fitness - trial_fitness) / pop_range
    
    # Combined signal: rank improvement + normalized improvement
    signal = 0.5 * rank_improvement + 0.5 * np.clip(norm_improvement, -1.0, 1.0)
    
    # Compute population-level convergence indicator
    pop_mean = float(np.mean(fitness))
    pop_std = float(np.std(fitness)) + 1e-10
    
    # Adaptive scaling based on distance to target (1e-08)
    # When FAR from target (pop_mean > 1.0): amplify rewards, encourage exploration
    # When NEAR target (pop_mean < 0.01): fine-grained rewards, encourage exploitation
    if pop_mean > 1.0:
        # Far from target: aggressive reward scaling
        scale_factor = 2.0 + np.log1p(pop_mean) * 0.2
    elif pop_mean < 0.01:
        # Near target: fine-grained rewards
        scale_factor = 0.5
    else:
        # Transition zone: linear scaling
        scale_factor = 1.0 + (1.0 - np.log10(pop_mean + 1e-10)) * 0.5
    
    # Apply scaling and compute final reward
    reward = signal * scale_factor
    
    # Penalize stagnation: if no improvements, give small negative reward
    n_improved = int(np.sum(trial_fitness < fitness))
    if n_improved == 0:
        reward = reward - 0.1
    
    return float(np.clip(reward, -1.0, 1.0))
```