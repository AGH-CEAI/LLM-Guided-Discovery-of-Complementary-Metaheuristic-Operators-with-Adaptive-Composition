**Idea: Reflective Boundary Handling**

Instead of hard-clipping particles at bounds (which kills velocity and causes discontinuity), use reflective boundaries that preserve momentum direction. This helps particles escape local optima near boundaries and explore more effectively on multimodal tasks (Tasks 12, 8, 13, 21, 23, 16).

```python
def _clip_to_bounds_batch(self, pop):
    """Reflective boundary handling - bounce off bounds preserving velocity."""
    lb, ub = self.lower, self.upper
    width = ub - lb
    
    # Compute normalized position within one period
    normalized = (pop - lb) / width
    
    # Handle below lower bound: reflect into valid range
    below = normalized < 0.0
    if np.any(below):
        pop = pop.copy()
        # Count complete periods and get fractional part
        normalized_below = normalized[below]
        fractional = normalized_below - np.floor(normalized_below)
        # Odd number of reflections -> mirror to upper side
        n_reflections = np.floor(-normalized_below)
        is_odd_reflection = (n_reflections.astype(int) % 2 == 1)
        # Map to valid range: even reflections stay near lb, odd near ub
        pop[below] = np.where(is_odd_reflection,
                             ub + fractional * width,
                             lb - fractional * width)
    
    # Handle above upper bound: reflect into valid range  
    above = normalized > 1.0
    if np.any(above):
        pop = pop.copy()
        normalized_above = normalized[above]
        fractional = normalized_above - np.floor(normalized_above)
        n_reflections = np.floor(normalized_above - 1.0)
        is_odd_reflection = (n_reflections.astype(int) % 2 == 1)
        pop[above] = np.where(is_odd_reflection,
                             lb + fractional * width,
                             ub - fractional * width)
    
    # Final safety clip for any edge cases
    return np.clip(pop, lb, ub)
```