Looking at the uncovered tasks (14, 15, 16, 17), I need to understand why they fail. The current `_check_stagnation` uses only the global best fitness with a fixed 50-generation threshold. For large-scale/deceptive tasks, this triggers reinitialization too aggressively, disrupting the search before convergence.

**Idea: Trend-adaptive stagnation with CV diversity and progressive randomization**

This approach uses multiple signals: (1) exponential moving average of fitness to detect improvement trends rather than single-generation drops, (2) coefficient of variation of population fitness to detect diversity collapse, and (3) adaptive thresholds based on problem scale and distance from optimum.

```python
def _check_stagnation(self, fitness):
        current_best = float(np.min(fitness))
        
        # Initialize trend tracking
        if not hasattr(self, 'ema_best'):
            self.ema_best = float(current_best)
            self.ema_alpha = 0.3
            self.stagnation_score = 0.0
        
        # Update exponential moving average of best fitness
        prev_ema = self.ema_best
        self.ema_best = self.ema_alpha * current_best + (1 - self.ema_alpha) * self.ema_best
        
        # Compute population diversity via coefficient of variation
        fit_mean = float(np.mean(fitness))
        fit_std = float(np.std(fitness))
        cv = fit_std / (abs(fit_mean) + 1e-10) if abs(fit_mean) > 1e-10 else 0.0
        
        # Adaptive threshold based on problem scale
        if self.ema_best > 1e3:
            base_threshold = 1e-2
        elif self.ema_best > 1e1:
            base_threshold = 1e-4
        else:
            base_threshold = 1e-6
        
        # Near optimum: require more consistent improvement
        if self.ema_best < 1.0:
            threshold = max(base_threshold, 1e-8)
        else:
            threshold = base_threshold
        
        # Check improvement: EMA is improving or significant direct improvement
        ema_improving = prev_ema - self.ema_best > threshold * max(1.0, abs(prev_ema))
        direct_improving = current_best < self.last_best_fitness - threshold * max(1.0, abs(self.last_best_fitness))
        
        if ema_improving or direct_improving:
            self.generation_without_improvement = 0
            self.stagnation_score = max(0.0, self.stagnation_score - 0.5)
            self.last_best_fitness = current_best
            return False
        else:
            self.generation_without_improvement += 1
            # Accumulate stagnation score; diversity collapse accelerates it
            self.stagnation_score += 1.0 + max(0.0, 0.5 - cv)
            
            # Stagnation threshold adapts: higher for large-scale or near-optimum problems
            dim_factor = max(1.0, self.dim / 50.0)
            near_opt_factor = 2.0 if self.ema_best < 1.0 else 1.0
            stagnation_threshold = 50.0 * dim_factor * near_opt_factor
            
            if self.stagnation_score > stagnation_threshold:
                self.ema_best = float(current_best)
                self.stagnation_score = 0.0
                self.generation_without_improvement = 0
                return True
            
            if self.generation_without_improvement > 100:
                self.ema_best = float(current_best)
                self.stagnation_score = 0.0
                return True
            
            return False
```