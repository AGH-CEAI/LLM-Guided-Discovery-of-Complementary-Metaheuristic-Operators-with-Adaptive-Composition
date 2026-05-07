**Idea: Adaptive Probabilistic Boundary Reflection**
Uses sigmoid-scaled reflection instead of hard clipping, with boundary softness increasing during stagnation to help escape local optima on the worst unsolved tasks (16, 12, 8) that are stuck at large errors (~1e+02).

```python
def _clip_to_bounds_batch(self, population):
    result = np.copy(population)
    
    # Compute how far each dimension exceeds bounds
    below_mask = result < self.lower
    above_mask = result > self.upper
    
    # Calculate distance beyond bounds
    below_dist = np.where(below_mask, self.lower - result, 0.0)
    above_dist = np.where(above_mask, result - self.upper, 0.0)
    
    # Adaptive boundary softness: increase during stagnation
    stagnation_factor = min(self.stagnation_count / 30.0, 1.0)
    generation_factor = min(self.generation / 100.0, 0.3)
    boundary_softness = 0.5 + 0.5 * stagnation_factor + generation_factor
    
    # Probabilistic reflection with adaptive probability
    for _ in range(3):  # Max 3 reflection bounces
        any_outside = np.any(below_mask) or np.any(above_mask)
        if not any_outside:
            break
        
        # Reflect below bounds
        reflect_below = below_mask & (np.random.rand(*result.shape) < boundary_softness)
        result[reflect_below] = 2 * self.lower[reflect_below] - result[reflect_below]
        
        # Reflect above bounds
        reflect_above = above_mask & (np.random.rand(*result.shape) < boundary_softness)
        result[reflect_above] = 2 * self.upper[reflect_above] - result[reflect_above]
        
        # Re-evaluate masks after reflection
        below_mask = result < self.lower
        above_mask = result > self.upper
    
    # Final hard clip as safety net
    result = np.clip(result, self.lower, self.upper)
    return result
```