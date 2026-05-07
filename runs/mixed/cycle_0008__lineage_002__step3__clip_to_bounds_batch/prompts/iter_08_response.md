**Idea: Reflective Boundary Wrapping**

Instead of hard clipping, use reflection at bounds with small random perturbations. This prevents population "sticking" at boundaries and encourages continued exploration in tasks where boundary traps cause the massive errors (Tasks 16, 12, 8, 21 all stuck at 10^2-10^3 error).

```python
def _clip_to_bounds_batch(self, population):
    clipped = np.clip(population, self.lower, self.upper)
    # Detect boundary hits and apply reflective perturbation
    hit_lower = population < self.lower
    hit_upper = population > self.upper
    if not (np.any(hit_lower) or np.any(hit_upper)):
        return clipped
    # Reflect with small random jitter to escape boundary traps
    jitter = np.random.randn(*population.shape) * 0.01 * (self.upper - self.lower)
    reflected = np.where(hit_lower, self.lower + (self.lower - population[hit_lower]) + jitter[hit_lower], clipped)
    reflected = np.where(hit_upper, self.upper - (population[hit_upper] - self.upper) + jitter[hit_upper], reflected)
    return np.clip(reflected, self.lower, self.upper)
```