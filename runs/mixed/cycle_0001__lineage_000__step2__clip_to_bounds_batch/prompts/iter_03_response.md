**Idea: Reflective Boundary with Momentum Preservation**

Instead of hard-clipping particles at bounds (which kills momentum and causes clustering), reflect particles back into the search space with dampened velocity. This preserves exploration direction and prevents particles from getting stuck at boundaries—a critical failure mode for the high-error tasks (12, 8, 16, 13) where the optimizer appears to be trapped.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflective boundary handling that preserves particle momentum."""
    if not hasattr(self, '_boundary_mode'):
        self._boundary_mode = 'reflect'
    
    pop = np.asarray(pop, dtype=np.float64)
    range_val = self.upper - self.lower
    
    if self._boundary_mode == 'wrap':
        # Periodic boundary
        return self.lower + (pop - self.lower) % range_val
    
    # Reflective boundary with momentum preservation
    result = pop.copy()
    
    # Reflect upper bounds
    upper_mask = pop > self.upper
    if np.any(upper_mask):
        overshoot = pop[upper_mask] - self.upper
        result[upper_mask] = self.upper - (overshoot % (2 * range_val))
        result[upper_mask] = np.abs(result[upper_mask] - self.upper) + self.lower
    
    # Reflect lower bounds  
    lower_mask = pop < self.lower
    if np.any(lower_mask):
        overshoot = self.lower - pop[lower_mask]
        result[lower_mask] = self.lower + (overshoot % (2 * range_val))
        result[lower_mask] = self.upper - np.abs(result[lower_mask] - self.lower)
    
    # Final safety clip
    np.clip(result, self.lower, self.upper, out=result)
    return result
```