**Idea: Relative Progress Reward with Hard-Task Prioritization**

This variant uses logarithmic scaling of relative improvement (improvement/current_error) to handle the huge error disparities across tasks. It prioritizes the hardest unsolved tasks (errors ~10^2-10^3) by giving higher rewards for proportional progress, while still rewarding fine-grained improvement near the target. The stagnation penalty scales with error magnitude, strongly discouraging giving up on hard tasks.
```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on relative improvement, prioritizing hard unsolved tasks"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            # Penalty scales with error magnitude - don't give up on hard tasks
            current_error = float(np.mean(np.abs(fitness)))
            penalty = -0.01 * (1.0 + np.log1p(max(current_error, 1.0)))
            return float(np.clip(penalty, -1.0, 0.0))

        total_improvement = float(np.sum(improvements[improvements > 0]))
        current_error = float(np.mean(np.abs(trial_fitness)))

        # Logarithmic relative improvement for large errors (hard unsolved tasks)
        # This gives higher rewards for proportional progress on Tasks 12, 16, 8, etc.
        if current_error > 1e-02:
            # Log scale: 10x improvement gets ~2.3 reward, 100x gets ~4.6
            reward = np.log1p(total_improvement) / (np.log1p(current_error) + 1e-10)
            reward = float(np.clip(reward, -0.5, 2.0))
        else:
            # Linear scale near target: reward fine-grained progress on Tasks 3, 15, 18
            reward = total_improvement / max(current_error, 1e-15)
            reward = float(np.clip(reward, -0.5, 3.0))

        return float(np.clip(reward, -1.0, 1.0))
```