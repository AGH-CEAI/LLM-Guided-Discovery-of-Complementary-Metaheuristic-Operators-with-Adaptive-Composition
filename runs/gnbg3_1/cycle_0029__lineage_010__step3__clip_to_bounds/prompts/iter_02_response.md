Looking at the worst unsolved tasks with errors in the 10^1 to 10^2 range, these are highly multi-modal/deceptive problems where the optimizer gets trapped in local optima. The current hard clipping (`np.clip`) creates artificial discontinuities that can trap the search near boundaries and prevent escape from local optima.

**Idea: Reflective Boundary Handling**
Instead of hard clipping, use smooth reflection at boundaries. This preserves search direction and momentum, allowing the optimizer to "bounce" off boundaries rather than getting stuck against hard walls. This is fundamentally different from simple clipping and can help escape local optima near boundaries.

```python
def _clip_to_bounds(self, x):
    """Clip solution to bounds using smooth reflection."""
    x = np.asarray(x, dtype=np.float64)
    lb = np.asarray(self.lb, dtype=np.float64)
    ub = np.asarray(self.ub, dtype=np.float64)
    
    range_size = ub - lb
    
    below = x < lb
    above = x > ub
    
    x = np.where(below, 2.0 * lb - x, x)
    x = np.where(above, 2.0 * ub - x, x)
    
    x = np.clip(x, lb, ub)
    
    return x
```