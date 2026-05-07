**Idea: Reflection with Adaptive Damping**

Rather than hard clipping, use boundary reflection with momentum damping. When solutions exceed bounds, reflect them back while dampening the reflected component. This preserves directional exploration momentum and prevents solutions from getting "stuck" near boundaries — a common failure mode in multimodal tasks with large errors where the optimizer can't escape local basins.

```python
def _clip_to_bounds_batch(self, population):
    reflected = np.copy(population)
    # Reflect values above upper bound
    above_upper = population > self.upper
    if np.any(above_upper):
        excess = population[above_upper] - self.upper
        reflected[above_upper] = self.upper - 0.7 * excess
    # Reflect values below lower bound
    below_lower = population < self.lower
    if np.any(below_lower):
        deficit = self.lower - population[below_lower]
        reflected[below_lower] = self.lower + 0.7 * deficit
    return reflected
```