**Idea: Boundary Reflection**
A reflection-based boundary handling strategy that mirrors out-of-bounds candidates back into the valid search space, preserving distribution characteristics better than hard clipping.

```python
def _clip_to_bounds_batch(self, population):
    """Reflect candidates that exceed bounds back into the search space."""
    # Identify out-of-bounds regions
    below_lb = population < self.lb
    above_ub = population > self.ub
    
    reflected = population.copy()
    
    # Reflect below-lb values: x' = lb + (lb - x) = 2*lb - x
    reflected[below_lb] = 2.0 * self.lb - population[below_lb]
    
    # Reflect above-ub values: x' = ub - (x - ub) = 2*ub - x
    reflected[above_ub] = 2.0 * self.ub - population[above_ub]
    
    # Final safety clip for any remaining edge cases
    return np.clip(reflected, self.lb, self.ub)
```