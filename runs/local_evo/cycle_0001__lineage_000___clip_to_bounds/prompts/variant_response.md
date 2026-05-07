**Idea 1: Exponential Crossover Clip**  
Use exponential decay to pull out‑of‑bounds values toward the nearest bound.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    rng = high - low
    rng = rng if rng != 0 else 1e-12
    above = candidates > high
    below = candidates < low
    d_above = candidates - high
    d_below = low - candidates
    repaired = np.where(above, high - d_above * np.exp(-d_above / rng), candidates)
    repaired = np.where(below, low + d_below * np.exp(-d_below / rng), repaired)
    return np.clip(repaired, low, high)
```

**Idea 2: Arithmetic Blend (BLX‑α) Repair**  
Blend each out‑of‑bounds coordinate with the nearest bound using a random weight in [0.5, 1.5].  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    above = candidates > high
    below = candidates < low
    w = self._rng.uniform(0.5, 1.5, size=candidates.shape)
    repaired = np.where(above, high - w * (candidates - high), candidates)
    repaired = np.where(below, low + w * (low - candidates), repaired)
    return np.clip(repaired, low, high)
```

**Idea 3: Sawtooth Reflection (Segment‑based)**  
Reflect values off the bounds in a periodic “sawtooth” fashion so they bounce within the interval.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    segment = high - low
    if segment == 0:
        return np.full_like(candidates, low)
    period = 2 * segment
    shifted = candidates - low
    t = shifted % period
    reflected = np.where(t <= segment, low + t, high - (t - segment))
    return np.clip(reflected, low, high)
```

**Idea 4: Eigenvector‑rotated Projection**  
Rotate the population by the covariance eigenvectors, clip in the rotated space, then rotate back.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    if self.covariance is not None:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            V = eigenvectors
            centered = candidates - self.mean
            y = centered @ V
            sigma = np.sqrt(eigenvalues)
            y_clipped = np.clip(y, -sigma, sigma)
            repaired = y_clipped @ V.T + self.mean
            return np.clip(repaired, low, high)
        except np.linalg.LinAlgError:
            pass
    return np.clip(candidates, low, high)
```

**Idea 5: Mirror Bounce Reflection**  
Iteratively reflect out‑of‑bounds points across the nearest bound until they fall inside.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    repaired = candidates.copy()
    max_iter = 10
    for _ in range(max_iter):
        above = repaired > high
        below = repaired < low
        if not (np.any(above) or np.any(below)):
            break
        diff = repaired[above] - high
        repaired[above] = high - diff
        diff = low - repaired[below]
        repaired[below] = low + diff
    return np.clip(repaired, low, high)
```

**Idea 6: Stochastic Truncated Normal Repair**  
Replace out‑of‑bounds coordinates with samples from a truncated normal centered on the bound.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    repaired = candidates.copy()
    if self.covariance is not None:
        std = np.sqrt(np.diag(self.covariance))
    else:
        std = np.full(self.dim, self.step_size)
    std = np.maximum(std, 1e-12)
    noise = self._rng.normal(0, std, size=candidates.shape)
    above = repaired > high
    below = repaired < low
    if np.any(above):
        repaired[above] = np.clip(high + noise[above], low, high)
    if np.any(below):
        repaired[below] = np.clip(low - noise[below], low, high)
    return repaired
```

**Idea 7: Logit Sigmoid Mapping**  
Map the entire real line to the bounds via a logistic‑type transformation.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    midpoint = (high + low) / 2.0
    rng = high - low
    if rng == 0:
        return np.full_like(candidates, low)
    z = (candidates - midpoint) / (rng / 2.0)
    compressed = np.tanh(z)
    repaired = low + (compressed + 1.0) * 0.5 * rng
    return np.clip(repaired, low, high)
```

**Idea 8: Adaptive Shrinkage Repair**  
Shrink out‑of‑bounds distances using a rational factor that depends on the overshoot.  

```python
def _clip_to_bounds(self, candidates):
    low, high = self.bounds
    rng = high - low
    if rng == 0:
        return np.full_like(candidates, low)
    above = candidates > high
    below = candidates < low
    d_above = candidates - high
    d_below = low - candidates
    shrink_above = 1.0 / (1.0 + d_above / rng)
    shrink_below = 1.0 / (1.0 + d_below / rng)
    repaired = np.where(above, high - d_above * shrink_above, candidates)
    repaired = np.where(below, low + d_below * shrink_below, repaired)
    return np.clip(repaired, low, high)
```