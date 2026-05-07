Looking at the data, the worst tasks (12, 8, 13, 21, 23, 16) have errors 10+ orders of magnitude above target, indicating severe stagnation in poor local optima. The current hard clipping destroys velocity information when particles hit boundaries, causing them to stop and lose momentum—this is catastrophic for escaping boundary-adjacent local optima.

**Idea: Reflective Boundary with Velocity Preservation**
Instead of hard clipping, reflect particles at boundaries while preserving velocity direction. This maintains search momentum and helps particles escape boundary-trapped local optima by bouncing them back into the search space naturally.

```python
def _clip_to_bounds_batch(self, pop):
    """Clip population to search bounds with velocity reflection."""
    reflected = np.copy(pop)
    
    # Reflect values below lower bound
    below_mask = reflected < self.lower
    if np.any(below_mask):
        reflected[below_mask] = 2 * self.lower - reflected[below_mask]
        reflected = np.clip(reflected, self.lower, self.upper)
    
    # Reflect values above upper bound
    above_mask = reflected > self.upper
    if np.any(above_mask):
        reflected[above_mask] = 2 * self.upper - reflected[above_mask]
        reflected = np.clip(reflected, self.lower, self.upper)
    
    return reflected
```