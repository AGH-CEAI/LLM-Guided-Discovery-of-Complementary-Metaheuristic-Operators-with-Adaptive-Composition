Looking at the priority unsolved tasks, I need to understand why operators fail on the hardest problems (Tasks 12, 16, 8 with errors >100) versus near-target tasks.

**Analysis:**
- Tasks 12, 16, 8 (error ~300-600): Highly multi-modal/deceptive landscapes where operators make tiny progress relative to the massive fitness range
- Current reward: `total_improvement / n_improved` — rewards **magnitude of improvement**, so operators that make tiny progress on hard tasks get near-zero reward, even though breaking through a plateau is as important (or more) than making large gains on easy tasks
- The adaptive selector needs to **value progress on hard problems equally** to progress on easy ones

**Idea: Logarithmic Relative Improvement Reward**
This fundamentally differs by: (1) using log-scaling to compress huge fitness values so small gains on hard problems are valued, (2) rewarding **any positive progress** equally (not just large improvements), (3) adding a consistency bonus to reward operators that reliably make progress across individuals.

```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on relative improvement quality (log-scaled)"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        n_improved = int(np.sum(improvements > 0))
        total_improved = float(n_improved)

        if n_improved == 0:
            return -0.1

        # Compute relative improvement (fractional gain)
        relative_imp = improvements[improvements > 0] / (np.abs(fitness[improvements > 0]) + 1e-10)
        
        # Log-scaled reward: small gains on hard problems valued similarly to large gains on easy ones
        log_rewards = np.log1p(relative_imp)
        avg_log_reward = float(np.mean(log_rewards))

        # Consistency bonus: reward operators that improve many individuals
        consistency_ratio = total_improved / max(float(len(fitness)), 1.0)
        consistency_bonus = 0.15 * consistency_ratio

        # Combined reward: log-scale captures difficulty-normalized progress
        reward = avg_log_reward + consistency_bonus

        return float(np.clip(reward, -0.5, 1.0))
```