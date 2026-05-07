**Idea: Reflective Boundary with Velocity Inversion**

Instead of hard-clipping (which kills momentum and causes particles to "stick" at boundaries), use reflective boundaries that invert velocity when particles cross bounds. This preserves exploration momentum and is fundamentally different from the simple clipping used by all 7 prior variants. The worst unsolved tasks (12, 8, 13, 21) likely suffer from particles getting trapped at boundaries and losing exploration capability.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflective boundary handling - invert velocity at bounds to preserve momentum."""
    reflected = pop.copy()
    # Below lower bound: reflect back inward
    below_lower = reflected < self.lower
    reflected[below_lower] = 2.0 * self.lower - reflected[below_lower]
    # Above upper bound: reflect back inward
    above_upper = reflected > self.upper
    reflected[above_upper] = 2.0 * self.upper - reflected[above_upper]
    # Final hard clip for any double-reflections (extreme cases)
    return np.clip(reflected, self.lower, self.upper)
```