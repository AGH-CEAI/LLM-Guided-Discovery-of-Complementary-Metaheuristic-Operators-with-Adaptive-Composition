Looking at the task errors, the worst unsolved tasks (16, 12, 8, 21, 23) have errors stuck at 10^1 to 10^3, suggesting fundamental convergence failures. The current hard clipping (`np.clip`) destroys directional information when particles exceed bounds, effectively resetting exploration momentum.

**Idea: Boundary Reflection with Momentum Preservation**

A reflection strategy bounces particles back into bounds while preserving their directional trajectory. This is fundamentally different from hard clipping because:
1. It maintains exploration momentum near boundaries
2. For rotated/non-separable problems (likely the hardest tasks), preserving direction helps escape poor regions
3. It provides a softer "landing" at boundaries rather than abrupt truncation

```python
def _clip_to_bounds_batch(self, population):
    # Reflection-based boundary handling: bounce back instead of hard clip
    # Preserves directional information for continued exploration
    result = np.copy(population)
    
    # Handle upper bound violations with reflection
    upper_mask = result > self.upper
    if np.any(upper_mask):
        overflow = result[upper_mask] - self.upper
        result[upper_mask] = self.upper - overflow
    
    # Handle lower bound violations with reflection
    lower_mask = result < self.lower
    if np.any(lower_mask):
        underflow = self.lower - result[lower_mask]
        result[lower_mask] = self.lower + underflow
    
    # Final safety clip for any remaining boundary violations
    return np.clip(result, self.lower, self.upper)
```