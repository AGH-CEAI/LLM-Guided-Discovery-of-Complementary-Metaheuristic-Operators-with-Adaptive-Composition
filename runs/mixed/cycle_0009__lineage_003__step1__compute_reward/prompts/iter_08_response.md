Looking at the unsolved tasks, the worst failures (Tasks 12, 16, 8, 21, 23, 14, 20, 13 with errors 5-650) are stuck in local optima. The current reward function only measures immediate improvement magnitude, which fails to signal when the algorithm needs to switch to exploration-heavy operators. I need a reward that detects stagnation and incentivizes operators that can escape local optima.

**Idea: Stagnation-Detecting Bimodal Reward**
When stagnant, reward EXPLORATION (diversity of trial solutions from population). When improving, reward EXPLOITATION (fitness gain). This bimodal signal helps Thompson Sampling select the right operator for each problem state.

```python
def _compute_reward(self, fitness, trial_fitness):
    """Compute reward with stagnation-aware bimodal strategy"""
    # Guard against invalid values
    fitness = np.clip(fitness, -1e50, 1e50)
    trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

    improvements = fitness - trial_fitness
    total_improvement = float(np.sum(improvements[improvements > 0]))
    n_improved = int(np.sum(improvements > 0))

    # Compute best fitness improvement rate
    best_before = float(np.min(fitness))
    best_after = float(np.min(trial_fitness))
    best_delta = best_before - best_after  # positive = improvement

    # Compute diversity of trial solutions (exploration signal)
    # Use mean distance of trials from population centroid
    pop_centroid = np.mean(fitness)  # weighted by fitness similarity
    trial_centroid = np.mean(trial_fitness)
    # Diversity = spread of trial fitness values
    trial_spread = float(np.std(trial_fitness[np.isfinite(trial_fitness)])) if np.any(np.isfinite(trial_fitness)) else 0.0
    pop_spread = float(np.std(fitness[np.isfinite(fitness)])) if np.any(np.isfinite(fitness)) else 1.0
    diversity_ratio = trial_spread / (pop_spread + 1e-10)

    # Compute population-level diversity (geometric diversity of solutions)
    finite_trials = trial_fitness[np.isfinite(trial_fitness)]
    if len(finite_trials) > 1:
        diversity_score = float(np.std(finite_trials)) / (float(np.abs(np.mean(finite_trials))) + 1e-10)
    else:
        diversity_score = 0.0

    # Detect stagnation: best solution not improving
    is_stagnant = (best_delta < 1e-6 * (abs(best_before) + 1.0)) or (best_delta <= 0)

    if is_stagnant:
        # Bimodal: when stagnant, reward EXPLORATION
        # Higher diversity = better at escaping local optima
        reward = float(np.clip(0.3 * diversity_ratio + 0.7 * diversity_score, -0.5, 1.0))
    else:
        # Bimodal: when improving, reward EXPLOITATION
        if n_improved == 0:
            reward = -0.05
        else:
            # Normalize by population spread to reward bold improvements
            norm_improvement = total_improvement / max(float(n_improved), 1.0)
            pop_scale = pop_spread + abs(best_before) * 0.1 + 1.0
            reward = float(np.clip(norm_improvement / pop_scale, -0.1, 1.0))

    return float(np.clip(reward, -1.0, 1.0))
```