**Idea: Reflective Boundary with Velocity Reversal**

Instead of hard clipping (which kills momentum at boundaries), use reflective boundaries that bounce particles back with reversed velocity. This preserves directional momentum and helps particles escape local optima near boundaries—a critical failure mode for the worst tasks (12, 8, 13, 21, 23, 16) where errors exceed 1e+01.

```python
def _clip_to_bounds_batch(self, pop):
    """Reflective boundary handling: bounce particles back with velocity reversal."""
    # Calculate overflow past upper bound and reflect
    upper_overflow = pop - self.upper
    upper_mask = upper_overflow > 0
    pop = np.where(upper_mask, 2 * self.upper - pop, pop)
    
    # Calculate underflow past lower bound and reflect  
    lower_underflow = self.lower - pop
    lower_mask = lower_underflow > 0
    pop = np.where(lower_mask, 2 * self.lower - pop, pop)
    
    # Final safety clip for any remaining out-of-bounds values
    return np.clip(pop, self.lower, self.upper)
```