**Idea: Variance-Weighted Reflective Boundary Handling**

Instead of hard-clipping, reflect out-of-bounds points back into bounds while preserving directional information. This prevents destroying gradient-like signals when solutions hit boundaries—a key failure mode for the worst tasks (16, 19, 20, 23) with errors >10^1.

```python
def _clip_to_bounds_batch(self, population):
    """Reflect out-of-bounds candidates back into bounds while preserving directional info."""
    result = np.copy(population)
    
    # Handle lower bound violations
    lb_violations = result < self.lb
    if np.any(lb_violations):
        # Reflect: distance beyond lb becomes distance from lb on other side
        result[lb_violations] = 2.0 * self.lb - result[lb_violations]
    
    # Handle upper bound violations
    ub_violations = result > self.ub
    if np.any(ub_violations):
        # Reflect: distance beyond ub becomes distance from ub on other side
        result[ub_violations] = 2.0 * self.ub - result[ub_violations]
    
    # Final safety clip for any remaining edge cases
    return np.clip(result, self.lb, self.ub)
```