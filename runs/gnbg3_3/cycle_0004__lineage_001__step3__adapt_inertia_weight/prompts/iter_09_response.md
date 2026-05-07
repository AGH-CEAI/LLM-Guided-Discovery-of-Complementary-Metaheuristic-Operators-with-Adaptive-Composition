**Idea: Pairwise-Diameter Inertia**
Adapt inertia using population diameter vs mean pairwise distance — a purely geometric measure of swarm compactness that directly captures how clustered or dispersed particles are in space.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight based on pairwise diameter vs mean inter-particle distance."""
    # Population diameter: max pairwise Euclidean distance
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, -np.inf)
    diameter = np.sqrt(np.max(sq_dists)) + 1e-10

    # Mean pairwise distance as reference scale
    upper_sq = np.triu(sq_dists, k=1)
    mean_pairwise_sq = np.sum(upper_sq) / max(1, self.np * (self.np - 1) / 2)
    mean_pairwise = np.sqrt(max(mean_pairwise_sq, 0)) + 1e-10

    # Geometric ratio: diameter / mean_pairwise
    # High ratio -> elongated/fragmented swarm -> high inertia for exploration
    # Low ratio -> compact swarm -> low inertia for exploitation
    spread_ratio = diameter / (mean_pairwise + 1e-10)
    spread_ratio_normalized = np.clip(spread_ratio / 5.0, 0.0, 1.0)

    target_inertia = spread_ratio_normalized * 0.55 + 0.4
    decay = 0.729 - 0.15 * (self.generation / 1000)
    target_inertia = np.clip(target_inertia * 0.5 + decay * 0.5, 0.4, 0.95)

    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```