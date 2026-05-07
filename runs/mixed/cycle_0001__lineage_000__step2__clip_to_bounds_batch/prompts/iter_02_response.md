Looking at the unsolved tasks, I see massive errors (hundreds to thousands) concentrated in Tasks 12, 16, 8, 13, 2, 21, 23. These suggest the optimizer is getting stuck—likely with particles accumulating at boundaries and losing diversity. The current hard clipping (`np.clip`) causes particles to "pile up" at bounds, destroying velocity information and exploration.

**Idea: Velocity Reflection at Boundaries**
Instead of hard clipping which kills velocity, reflect particles at boundaries to preserve momentum and maintain exploration diversity. This prevents the catastrophic diversity loss that appears to be causing the worst tasks to diverge.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflect particles at boundaries to preserve velocity and exploration."""
    pop = np.asarray(pop, dtype=np.float64)
    lower = np.asarray(self.lower, dtype=np.float64)
    upper = np.asarray(self.upper, dtype=np.float64)
    range_arr = upper - lower
    
    # First, clip to prevent extreme values
    pop = np.clip(pop, lower - range_arr * 0.5, upper + range_arr * 0.5)
    
    # Reflect particles that went below lower bound
    below_mask = pop < lower
    if np.any(below_mask):
        offset = lower - pop
        reflections = np.floor(offset / range_arr) + 1
        pop = lower + (offset - reflections * range_arr)
        pop = np.where(np.abs(offset - reflections * range_arr) < 
                       np.abs(offset - (reflections - 1) * range_arr),
                       pop, lower - (offset - (reflections - 1) * range_arr))
    
    # Reflect particles that went above upper bound
    above_mask = pop > upper
    if np.any(above_mask):
        offset = pop - upper
        reflections = np.floor(offset / range_arr) + 1
        pop = upper - (offset - reflections * range_arr)
        pop = np.where(np.abs(offset - reflections * range_arr) < 
                       np.abs(offset - (reflections - 1) * range_arr),
                       pop, upper + (offset - (reflections - 1) * range_arr))
    
    return np.clip(pop, lower, upper)
```