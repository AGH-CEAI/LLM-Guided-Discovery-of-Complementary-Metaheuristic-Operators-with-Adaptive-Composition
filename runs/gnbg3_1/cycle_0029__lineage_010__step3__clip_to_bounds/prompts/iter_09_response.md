**Idea: Cyclic Boundary Wrapping**
Use modular/cyclic wrapping instead of hard clipping. When a solution exceeds bounds, it wraps to the opposite side of the search space, preserving directional exploration momentum and enabling escape from local optima near boundaries.

```python
def _clip_to_bounds(self, x):
    """Clip solution to bounds using cyclic wrapping."""
    bound_range = self.ub - self.lb
    x_clipped = x.copy() if isinstance(x, np.ndarray) else np.array(x, dtype=np.float64)
    
    for d in range(self.dim):
        if x_clipped[..., d].ndim == 0:
            val = float(x_clipped[d])
            if val < self.lb[d]:
                x_clipped[d] = self.ub[d] - ((self.lb[d] - val) % bound_range[d])
            elif val > self.ub[d]:
                x_clipped[d] = self.lb[d] + ((val - self.ub[d]) % bound_range[d])
        else:
            mask_below = x_clipped[..., d] < self.lb[d]
            mask_above = x_clipped[..., d] > self.ub[d]
            if np.any(mask_below):
                x_clipped[mask_below, d] = self.ub[d] - ((self.lb[d] - x_clipped[mask_below, d]) % bound_range[d])
            if np.any(mask_above):
                x_clipped[mask_above, d] = self.lb[d] + ((x_clipped[mask_above, d] - self.ub[d]) % bound_range[d])
    
    return x_clipped
```