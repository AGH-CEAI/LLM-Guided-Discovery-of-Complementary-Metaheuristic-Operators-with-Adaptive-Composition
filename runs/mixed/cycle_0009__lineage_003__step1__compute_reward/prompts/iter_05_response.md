**Idea: Log-Scaled Consistency Reward**

A fundamentally different reward strategy: use logarithmic improvement scaling, penalize non-improvement, and reward per-individual consistency rather than total improvement magnitude. This targets the worst unsolved tasks (12, 16, 8, 21, 23) which are stuck at errors 10^1–10^3 — these likely need operators that make *consistent* small-to-medium improvements rather than operators that occasionally make huge improvements on easy tasks.

```python
def _compute_reward(self, fitness, trial_fitness):
    """Log-scaled consistency reward: penalize stagnation, reward per-individual improvement"""
    # Guard against invalid values
    fitness = np.clip(fitness, -1e50, 1e50)
    trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

    improvements = fitness - trial_fitness
    
    # Compute per-individual log-scaled improvement rewards
    reward_components = []
    
    for i in range(len(improvements)):
        imp = improvements[i]
        if imp > 1e-10:
            # Log-scaled reward: compress huge improvements, boost consistent small ones
            # Using log(1 + imp) gives diminishing returns for large imp
            # Adding small baseline ensures tiny improvements still register
            log_reward = np.log1p(imp) / 20.0
            reward_components.append(log_reward)
        elif imp < -1e-10:
            # Penalize degradation (but not as harshly as -0.01 flat penalty)
            reward_components.append(-0.05 * np.log1p(-imp) / 20.0)
        else:
            # No change: small penalty to encourage seeking improvements
            reward_components.append(-0.005)
    
    if len(reward_components) == 0:
        return -0.1
    
    # Return mean of per-individual rewards (rewards consistency over total magnitude)
    reward = float(np.mean(reward_components))
    
    return float(np.clip(reward, -1.0, 1.0))
```