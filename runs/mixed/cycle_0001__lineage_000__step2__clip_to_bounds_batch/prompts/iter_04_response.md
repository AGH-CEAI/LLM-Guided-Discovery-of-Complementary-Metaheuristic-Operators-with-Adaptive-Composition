**Idea: Reflective Boundary Handling**

Instead of hard clipping which destroys velocity information and creates discontinuities, use **symmetric reflection at boundaries**. This preserves directional momentum, prevents particles from getting stuck at edges, and enables continued exploration near constraints—critical for the high-error tasks (12, 8, 13) that appear to suffer from premature boundary saturation.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflective boundary handling - particles bounce off bounds instead of hard clipping."""
    range_size = self.upper - self.lower
    result = np.copy(pop)
    
    # Reflect above upper bound
    above_mask = result > self.upper
    if np.any(above_mask):
        excess = result[above_mask] - self.upper
        reflected = self.upper - (excess % (2 * range_size))
        reflected = np.where(reflected < self.lower, 
                            2 * self.lower - reflected, reflected)
        result[above_mask] = np.clip(reflected, self.lower, self.upper)
    
    # Reflect below lower bound
    below_mask = result < self.lower
    if np.any(below_mask):
        excess = self.lower - result[below_mask]
        reflected = self.lower + (excess % (2 * range_size))
        reflected = np.where(reflected > self.upper, 
                            2 * self.upper - reflected, reflected)
        result[below_mask] = np.clip(reflected, self.lower, self.upper)
    
    return result
```