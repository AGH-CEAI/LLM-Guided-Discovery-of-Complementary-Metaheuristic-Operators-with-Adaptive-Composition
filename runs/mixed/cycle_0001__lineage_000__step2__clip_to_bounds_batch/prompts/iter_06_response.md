Looking at the unsolved tasks, I see two distinct failure patterns:
1. **Severe stagnation** (Tasks 12, 8, 13, 21, 16, 23): errors 30-2000, suggesting particles get trapped at boundaries or in local optima
2. **Moderate stagnation** (Tasks 2, 20, 14, etc.): errors 5-8, partial convergence but stuck

The current hard clipping (`np.clip`) causes particles that reach boundaries to **accumulate at edges**, destroying diversity. This is especially problematic for tasks where the global optimum lies near boundaries or where periodic structure exists.

**Idea: Periodic Toroidal Wrapping**

Instead of clamping particles to boundaries, wrap them toroidally. This is fundamentally different because:
- Particles never stop at boundaries — they continue exploring the "other side"
- Creates natural escape from local optima near boundaries
- Maintains full population diversity throughout all dimensions
- Handles periodic fitness landscapes naturally

```python
def _clip_to_bounds_batch(self, pop):
    """Wrap population toroidally within search bounds using periodic boundary conditions."""
    range_size = self.upper - self.lower
    # Wrap each dimension using modulo arithmetic
    wrapped = (pop - self.lower) % range_size + self.lower
    return wrapped
```