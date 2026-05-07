**Idea: Axis-Aligned Spread Ratio Inertia**

Adapt inertia weight using the ratio of geometric mean to arithmetic mean of axis-aligned spreads, capturing the isotropy of the swarm's spatial distribution without using fitness or probability distributions.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight based on axis-aligned geometric spread ratio."""
    # Compute axis-aligned bounding box
    min_pos = np.min(self.population, axis=0)
    max_pos = np.max(self.population, axis=0)
    spread = max_pos - min_pos + 1e-10
    
    # Geometric mean vs arithmetic mean ratio of spreads
    # Low ratio -> anisotropic (elongated) swarm -> increase exploration (higher inertia)
    # High ratio -> isotropic (spherical) swarm -> increase exploitation (lower inertia)
    geo_mean = np.exp(np.mean(np.log(spread)))
    arith_mean = np.mean(spread)
    spread_ratio = geo_mean / (arith_mean + 1e-10)
    
    # Normalize ratio: for dim=10, random uniform spread_ratio ~0.9
    # Tight clustering gives spread_ratio ~0.01
    spread_ratio_normalized = np.clip(spread_ratio / 0.9, 0.0, 1.0)
    
    # Invert: low ratio -> high inertia (exploration)
    # Generational decay pushes toward exploitation over time
    target_inertia = (1.0 - spread_ratio_normalized) * 0.55 + 0.4
    decay = 0.729 - 0.15 * (self.generation / 1000)
    target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)
    
    # Smooth convergence toward target
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```