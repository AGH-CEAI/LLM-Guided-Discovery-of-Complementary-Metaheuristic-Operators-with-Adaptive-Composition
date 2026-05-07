**Idea: Momentum-Enhanced Step Size Adaptation with Diversity Modulation**

This approach fundamentally differs from the simple success-rate method by: (1) tracking cumulative improvement magnitude (not just count), (2) using exponential moving average momentum for smoother adaptation, and (3) modulating damping based on population diversity to escape local optima on the worst tasks (17, 16, 19, etc.).

```python
def _adapt_step_size(self):
    """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
    recent_improved = 0
    recent_total = 0
    total_improvement = 0.0

    for i in range(self.NP):
        if i < len(self.trial_fitness) and i < len(self.fitness):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
                diff = self.fitness[i] - self.trial_fitness[i]
                total_improvement += max(diff, 0.0)
            recent_total += 1

    if recent_total == 0:
        recent_total = max(1, self.NP // 4)

    success_rate = recent_improved / recent_total
    success_rate = np.clip(success_rate, 0.0, 1.0)

    avg_improvement = total_improvement / max(recent_improved, 1)
    improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

    if not hasattr(self, 'improvement_ema'):
        self.improvement_ema = improvement_magnitude
    self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

    target_rate = 0.25
    adaptation = (success_rate - target_rate) / target_rate
    adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
    adaptation = np.clip(adaptation, -0.8, 0.8)

    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

    diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
    diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

    damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
    damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

    self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```