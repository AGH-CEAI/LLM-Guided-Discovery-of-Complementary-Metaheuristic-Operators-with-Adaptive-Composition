**Idea: Log-Scaled Stagnation-Penalized Reward**

This variant uses logarithmic scaling of improvements to handle the enormous fitness ranges (1e-08 to 1e+03) on unsolved tasks, and applies a stronger stagnation penalty when no improvements occur. This fundamentally differs from the original's linear average by: (1) compressing extreme fitness differences into comparable reward signals via log transformation, (2) differentiating small vs. large improvements with different scaling, and (3) penalizing stagnation proportionally to how far the optimizer is from the target.

```python
def _compute_reward(self, fitness, trial_fitness):
        """Compute reward using log-scaled improvements with stagnation penalty"""
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            # Stagnation penalty proportional to distance from target
            best_f = float(np.min(fitness))
            if np.isinf(best_f):
                reward = -1.0
            else:
                log_target = -8.0  # log10(1e-08)
                log_best = np.log10(max(abs(best_f), 1e-100))
                distance = max(0.0, log_best - log_target)
                reward = -min(1.0, distance / 30.0)
        else:
            # Log-scale small improvements to handle wide fitness ranges
            log_improvement = np.log1p(max(total_improvement, 0.0))
            small_thresh = 1.0
            if total_improvement <= small_thresh:
                reward = np.clip(log_improvement / 5.0, -0.5, 0.5)
            else:
                reward = 0.5 + np.clip((total_improvement - small_thresh) / float(self.NP), 0.0, 0.5)
            reward = reward / max(float(n_improved), 1.0)

        return float(np.clip(reward, -1.0, 1.0))
```