**Idea: Boundary Reflection**
Instead of hard clipping, reflect particles back from boundaries to maintain momentum and exploration. This prevents particles from getting "stuck" at bounds and helps preserve directional information.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflect particles back from bounds to maintain exploration momentum."""
    reflected = pop.copy()
    
    # Reflect values below lower bound
    below_mask = reflected < self.lower
    if np.any(below_mask):
        excess = self.lower - reflected[below_mask]
        reflected[below_mask] = self.lower + excess
    
    # Reflect values above upper bound
    above_mask = reflected > self.upper
    if np.any(above_mask):
        excess = reflected[above_mask] - self.upper
        reflected[above_mask] = self.upper - excess
    
    # Final safety clip for any remaining edge cases
    return np.clip(reflected, self.lower, self.upper)
```