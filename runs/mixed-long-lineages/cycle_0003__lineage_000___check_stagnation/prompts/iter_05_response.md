Looking at the uncovered tasks (11, 14, 15, 16, 17), ALL variants are making them worse than the original. This suggests the stagnation/reinitialization mechanism may be too aggressive, destroying good solutions. The current simple counter-based approach doesn't account for the magnitude or pattern of recent improvements.

**Idea: Momentum-Based Stagnation Detection**
Uses exponential moving average of improvement rates to detect true stagnation vs. slow progress, with adaptive thresholds that scale with problem difficulty.

```python
def _check_stagnation(self, fitness):
        current_best = float(np.min(fitness))
        improvement = self.last_best_fitness - current_best
        
        # Track improvement momentum with EMA
        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = 0.0
            self.recent_improvements = []
        
        alpha = 0.3  # EMA smoothing
        if improvement > 1e-10:
            self.improvement_ema = alpha * improvement + (1 - alpha) * self.improvement_ema
            self.generation_without_improvement = 0
        else:
            self.generation_without_improvement += 1
        
        self.last_best_fitness = current_best
        
        # Adaptive threshold based on current fitness magnitude
        scale = max(abs(current_best), 1.0)
        threshold = max(50, int(100 / (1 + self.improvement_ema / scale)))
        
        # Only trigger if no improvement AND momentum is dead AND sufficient generations
        if self.generation_without_improvement > threshold:
            # Check if recent momentum was truly zero (not just slow)
            if self.improvement_ema < scale * 1e-8:
                return True
        
        return False
```