**Idea: Logarithmic Decade-Closing Reward**

Replaces linear improvement magnitude with log-scale reward that prioritizes closing decades of error on the hardest (worst) tasks. This fundamentally changes the reward landscape: a 10x improvement on a task with error 100 gets the same reward as a 10x improvement on a task with error 1e-6, but far more reward than a 1000x improvement on an already-solved task.

```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on logarithmic improvement toward target (1e-08)"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            return -0.01

        # Log-scale reward: reward is proportional to decades closed toward target
        # This prioritizes hard tasks (error ~1e+2) over easy tasks (error ~1e-6)
        target = 1e-8
        log_fitness = np.log10(np.maximum(np.abs(fitness), target) + 1e-100)
        log_trial = np.log10(np.maximum(np.abs(trial_fitness), target) + 1e-100)

        # Log improvement: positive when closing distance to target
        log_improvements = log_trial - log_fitness
        log_improvements[improvements <= 0] = 0.0

        # Sum of decades closed, weighted by population fraction
        total_log_improvement = float(np.sum(log_improvements))
        n_dims = float(len(fitness))

        # Scale: ~2 decades of improvement → reward = 1.0
        reward = total_log_improvement / (2.0 * n_dims / max(n_improved, 1))

        return float(np.clip(reward, -1.0, 1.0))
```