**Idea: Reflective Boundary Handling**

Instead of hard clipping (which causes population members to cluster at boundaries, killing diversity), use reflection at bounds to preserve directional information and maintain exploration. This is critical for the worst unsolved tasks (12, 16, 8, 21) which likely have boundary traps or need diverse exploration to escape local optima.

```python
def _clip_to_bounds_batch(self, population):
    range_width = self.upper - self.lower
    clipped = np.clip(population, self.lower, self.upper)
    overflow = population - clipped
    reflected = clipped - overflow
    reflected = np.where(population > self.upper,
                        self.upper - ((population - self.upper) % (2 * range_width + 1e-10)),
                        reflected)
    reflected = np.where(population < self.lower,
                        self.lower + ((self.lower - population) % (2 * range_width + 1e-10)),
                        reflected)
    return np.clip(reflected, self.lower, self.upper)
```