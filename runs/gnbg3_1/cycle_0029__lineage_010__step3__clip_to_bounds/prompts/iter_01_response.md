**Idea: Reflection Boundary Handling**
Reflect values that exceed bounds back into the valid range, preserving search direction unlike hard clipping.

```python
def _clip_to_bounds(self, x):
    """Reflect values outside bounds back into the valid range."""
    x = np.asarray(x, dtype=np.float64)
    lb, ub = self.lb, self.ub
    
    # Handle values below lower bound
    below = x < lb
    if np.any(below):
        excess = lb - x[below]
        x = np.where(below, lb + np.mod(excess, 2.0 * (ub - lb)), x)
    
    # Handle values above upper bound
    above = x > ub
    if np.any(above):
        excess = x[above] - ub
        x = np.where(above, ub - np.mod(excess, 2.0 * (ub - lb)), x)
    
    # Final safety clip for numerical stability
    return np.clip(x, lb, ub)
```