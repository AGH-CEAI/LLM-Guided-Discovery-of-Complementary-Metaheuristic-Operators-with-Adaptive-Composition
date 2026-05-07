**Idea: Momentum-Weighted Step Size with Diversity Correction**

Instead of simple success counting, this approach tracks improvement momentum via exponential moving average of fitness gains, applies non-linear scaling for aggressive response to stagnation, and includes a population diversity correction to prevent premature convergence. This targets the worst tasks (17, 16, 19, etc.) that appear stuck at large errors, likely due to step size being either too aggressive or too conservative without proper momentum feedback.

```python
def _adapt_step_size(self):
        """Adapt step-size using momentum-weighted improvement with diversity correction."""
        recent_improved = 0
        recent_total = 0
        improvement_sum = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    improvement_sum += self.fitness[i] - self.trial_fitness[i]
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        # Momentum-based improvement tracking (EMA of fitness gain)
        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = 1.0
        
        avg_improvement = improvement_sum / max(recent_improved, 1)
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * avg_improvement
        
        # Diversity-based correction to prevent premature convergence
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.01, 1.0)
        
        # Target rate with non-linear diversity modulation
        target_rate = 0.25 * (1.0 + 0.5 * np.log1p(1.0 / (diversity_ratio + 1e-10)))
        target_rate = np.clip(target_rate, 0.05, 0.6)
        
        # Non-linear adaptation: steeper response when far from target
        rate_ratio = success_rate / max(target_rate, 1e-10)
        if rate_ratio < 0.5:
            # Stagnation detected: use non-linear boost to escape
            adaptation = -0.5 * (1.0 + np.log1p(0.5 / (rate_ratio + 1e-10)))
        elif rate_ratio > 1.5:
            # Very high success: can be more aggressive
            adaptation = 0.5 * np.log1p(rate_ratio)
        else:
            # Normal regime
            adaptation = (success_rate - target_rate) / target_rate
        
        adaptation = np.clip(adaptation, -0.8, 0.8)
        
        # Momentum-weighted damping (slower when improvement momentum is low)
        momentum_factor = np.clip(np.log1p(self.improvement_ema * 1e6) / 10.0, 0.2, 2.0)
        
        # Dimension and diversity-aware damping
        damping_adaptive = self.damping * momentum_factor * (0.3 + 0.7 * np.log1p(self.dim) / diversity_ratio)
        damping_adaptive = np.clip(damping_adaptive, 0.05, 150.0)
        
        # Adaptive learning rate (faster early, slower late)
        generation_factor = np.exp(-self.generation / (50 + self.dim * 2))
        cs_adaptive = self.cs * (1.0 + generation_factor)
        
        self.sigma *= np.exp(adaptation * cs_adaptive / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```