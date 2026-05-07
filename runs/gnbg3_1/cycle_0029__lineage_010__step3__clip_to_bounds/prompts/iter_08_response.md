**Idea: Reflection-Based Boundary Handling**

Instead of hard-clipping at bounds (which creates discontinuities and loses directional information), use reflection to bounce values back into the feasible region. This preserves search momentum and prevents artificial "walls" that trap the algorithm—critical for the high-error tasks (16, 17, 11, etc.) where the optimizer appears stuck at local optima.

```python
def _clip_to_bounds(self, x):
    """Reflect out-of-bounds values back into the feasible region."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < self.lb, 2.0 * self.lb - x, x)
    x = np.where(x > self.ub, 2.0 * self.ub - x, x)
    x = np.clip(x, self.lb, self.ub)
    return x
```