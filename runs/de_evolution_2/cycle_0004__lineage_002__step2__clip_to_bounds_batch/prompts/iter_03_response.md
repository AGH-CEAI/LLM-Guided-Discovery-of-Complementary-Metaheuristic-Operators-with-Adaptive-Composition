**Idea: Boundary Reflection with Distribution Preservation**

Hard clipping destroys the correlation structure of the CMA-ES covariance matrix by truncating outliers to the boundary, which is catastrophic for tasks 10, 16, 18, 19, 20, 21, 23 (errors 4-400). This function uses **reflection** instead of truncation: points exceeding bounds bounce back into the feasible region while preserving their directional information and distance from the mean. This maintains the population distribution's integrity, allowing CMA-ES to continue proper covariance adaptation even when sampling generates extreme candidates.

```python
def _clip_to_bounds_batch(self, population):
    """Clip candidates to bounds using reflection instead of hard truncation.
    
    Reflection preserves population distribution structure by bouncing
    out-of-bound points back into the feasible region, maintaining the
    correlation information that hard clipping destroys.
    """
    clipped = np.clip(population, self.lb, self.ub)
    
    # Detect out-of-bounds values (where clipping changed the population)
    out_of_bounds = (population != clipped)
    
    if not np.any(out_of_bounds):
        return clipped
    
    # Reflection distances: how far past each boundary
    below_mask = population < self.lb
    above_mask = population > self.ub
    
    # Reflect below-bound points: bounce off lower boundary
    lb_excess = self.lb - population[below_mask]
    reflected_below = self.lb + lb_excess
    
    # Reflect above-bound points: bounce off upper boundary  
    ub_excess = population[above_mask] - self.ub
    reflected_above = self.ub - ub_excess
    
    # Apply reflections
    result = clipped.copy()
    result[below_mask] = reflected_below
    result[above_mask] = reflected_above
    
    # Ensure numerical safety: final clip in case of double-reflection edge cases
    return np.clip(result, self.lb, self.ub)
```