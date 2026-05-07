**Idea: Reflective Boundary Handling with Adaptive Damping**

A fundamentally different approach to boundary handling: instead of hard clipping, use velocity reflection with damping when particles exceed bounds. This preserves momentum and encourages boundary exploration, helping particles escape from corners and local optima that trap the optimizer on the worst tasks (12, 8, 13, 21) which show errors stuck at 1e+2 to 1e+3.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflective boundary handling with velocity damping for better exploration."""
    result = np.clip(pop, self.lower, self.upper)
    # Identify particles that hit boundaries (need reflection)
    below = pop < self.lower
    above = pop > self.upper
    # Apply reflection: particles that went out bounce back with damped velocity
    # This preserves momentum and encourages boundary region exploration
    if np.any(below) or np.any(above):
        reflected = result.copy()
        # Reflect with 0.7 damping factor to gradually reduce boundary exploration
        reflected[below] = self.lower + 0.7 * (result[below] - self.lower + 1e-10)
        reflected[above] = self.upper - 0.7 * (self.upper - result[above] + 1e-10)
        result = reflected
    return result
```