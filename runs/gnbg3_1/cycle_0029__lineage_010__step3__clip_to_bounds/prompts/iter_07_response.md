**Idea: Reflective Boundary Handling**

Instead of hard clipping, use reflection at bounds to preserve directional information and maintain search momentum. This is fundamentally different from simple truncation and helps escape local optima near boundaries by keeping the search trajectory intact.

```python
def _clip_to_bounds(self, x):
    """Reflective boundary handling to preserve search direction."""
    x_clipped = np.clip(x, self.lb, self.ub)
    
    # Compute how far each dimension exceeded bounds
    below_mask = x < self.lb
    above_mask = x > self.ub
    
    # For out-of-bounds dimensions, reflect the excess back
    if np.any(below_mask):
        excess = self.lb[below_mask] - x[below_mask]
        x_clipped[below_mask] = self.lb[below_mask] + excess * 0.5
    
    if np.any(above_mask):
        excess = x[above_mask] - self.ub[above_mask]
        x_clipped[above_mask] = self.ub[above_mask] - excess * 0.5
    
    # Ensure final values are within bounds (handle extreme cases)
    return np.clip(x_clipped, self.lb, self.ub)
```