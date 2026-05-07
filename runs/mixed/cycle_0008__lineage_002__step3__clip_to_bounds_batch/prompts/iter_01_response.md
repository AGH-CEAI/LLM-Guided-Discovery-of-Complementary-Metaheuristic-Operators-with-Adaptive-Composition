**Idea: Boundary Reflection**
Reflects out-of-bounds values symmetrically across boundaries instead of hard clipping, preserving mutation direction and improving exploration near constraints.

```python
def _clip_to_bounds_batch(self, population):
    result = np.copy(population)
    
    # Reflect values above upper bound
    above_upper = result > self.upper
    if np.any(above_upper):
        reflected = 2.0 * self.upper - result[above_upper]
        result[above_upper] = np.where(reflected >= self.lower, reflected,
                                        2.0 * self.lower - reflected)
    
    # Reflect values below lower bound
    below_lower = result < self.lower
    if np.any(below_lower):
        reflected = 2.0 * self.lower - result[below_lower]
        result[below_lower] = np.where(reflected <= self.upper, reflected,
                                        2.0 * self.upper - reflected)
    
    return result
```