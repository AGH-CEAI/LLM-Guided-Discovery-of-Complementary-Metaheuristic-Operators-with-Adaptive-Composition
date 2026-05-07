**Idea: Temporal-Topology Hybrid Neighborhood Adaptation**

Combine two distinct mechanisms: (1) temporal convergence signals (EMA improvement rate + stagnation counter) and (2) topology fragmentation signals (k-NN connected components + fragmentation ratio). The weighting is data-driven: topology signal dominates when population is fragmented (multiple disconnected components), temporal signal dominates when well-connected. This directly targets worst tasks (17, 16, 6, 11) where premature fragmentation traps the swarm in disconnected local optima — larger neighborhood bridges components, smaller neighborhood intensifies exploitation when converging.

```python
def _adapt_neighborhood_size(self):
    """Hybrid: combine temporal convergence signals with topology fragmentation detection."""
    # === MECHANISM 1: TEMPORAL CONVERGENCE SIGNALS ===
    if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
        improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
    else:
        improvement = 0.0
    self._prev_global_best_fitness = self.global_best_fitness

    if not hasattr(self, '_ema_improvement'):
        self._ema_improvement = 0.0
    self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

    # Stagnation signal
    stagnation_ratio = min(self.stagnation_counter / 100.0, 1.0)

    # Temporal signal: high improvement + low stagnation → exploit (smaller neighborhood)
    # Low improvement + high stagnation → explore (larger neighborhood)
    temporal_signal = stagnation_ratio - min(self._ema_improvement, 1.0)
    temporal_signal = np.clip(temporal_signal, -1.0, 1.0)

    # === MECHANISM 2: TOPOLOGY FRAGMENTATION SIGNALS ===
    k = min(5, self.np - 1)
    sq_dists = np.sum(
        (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2
    )
    np.fill_diagonal(sq_dists, np.inf)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

    # Connected component analysis via BFS
    visited = np.zeros(self.np, dtype=bool)
    components = []
    for start in range(self.np):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in knn_indices[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        components.append(component)

    component_sizes = np.array([len(c) for c in components])
    n_components = len(components)

    # Fragmentation ratio: how uneven is the distribution?
    max_component_size = np.max(component_sizes)
    fragmentation_ratio = 1.0 - (max_component_size / (self.np + 1e-10))

    # Topology signal: more components + higher fragmentation → explore (larger neighborhood)
    topology_signal = (n_components - 1) / max(self.np // 10, 1) + fragmentation_ratio
    topology_signal = np.clip(topology_signal, 0.0, 2.0)

    # === PRINCIPLED WEIGHTING SWITCHING ===
    is_fragmented = n_components > 1 and fragmentation_ratio > 0.3

    if is_fragmented:
        topology_weight = 0.7
        temporal_weight = 0.3
    else:
        topology_weight = 0.3
        temporal_weight = 0.7

    # === COMBINE MECHANISMS ===
    combined_signal = (
        temporal_weight * (0.5 + 0.5 * temporal_signal) +
        topology_weight * topology_signal
    )

    # Map combined signal to neighborhood size
    # signal > 1 → need more exploration → larger neighborhood
    # signal < 1 → need more exploitation → smaller neighborhood
    target_ratio = combined_signal
    target_size = int(self.np * np.clip(target_ratio, 0.05, 0.5))
    target_size = max(1, min(self.np // 2, target_size))

    # Gradual adjustment
    if target_size > self.neighborhood_size:
        self.neighborhood_size = min(target_size, self.neighborhood_size + 2)
    else:
        self.neighborhood_size = max(target_size, self.neighborhood_size - 2)

    self.neighborhood_size = max(1, min(self.np // 2, self.neighborhood_size))
```