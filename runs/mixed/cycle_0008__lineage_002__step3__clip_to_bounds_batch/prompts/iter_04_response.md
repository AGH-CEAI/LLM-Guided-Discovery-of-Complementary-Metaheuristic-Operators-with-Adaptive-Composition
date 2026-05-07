**Idea: Reflective Bounce with Stagnation-Adaptive Damping**

Instead of hard-clipping, use damped reflection at bounds to escape local optima where the algorithm appears stuck at specific error plateaus (Tasks 20, 21 at exactly 5.0). This preserves directional momentum and enables larger exploratory jumps when stagnation is high.

```python
def _clip_to_bounds_batch(self, population):
    reflected = np.copy(population)
    # Adaptive damping: less damping when stagnant (allow bigger jumps to escape)
    stagnation_factor = min(1.0 + 0.5 * self.stagnation_count, 2.5)
    damping = 1.0 / stagnation_factor
    # Upper bound: reflect with damped bounce
    upper_bounce = self.upper - (population - self.upper) * damping
    reflected = np.where(population > self.upper, upper_bounce, reflected)
    # Lower bound: reflect with damped bounce
    lower_bounce = self.lower + (self.lower - population) * damping
    reflected = np.where(population < self.lower, lower_bounce, reflected)
    # Safety clip for extreme outliers
    return np.clip(reflected, self.lower, self.upper)
```