**Idea: Adaptive Non-Linear Scaling with Multi-Vector Exploration**

This strategy tackles the worst unsolved tasks (16, 12, 8, 21, 23) which are stuck at error >5e+00 by combining: (1) non-linear power-law F scaling that grows super-exponentially with stagnation to break out of deep local optima, (2) dimension-level mixing with random swapping to escape deceptive basins, and (3) multi-vector composite (best+centroid+rand) with stagnation-adaptive weights to balance aggressive exploration vs. exploitation.

```python
def _mutate_original(self, population, fitness):
    """Adaptive Non-Linear Scaling with Multi-Vector Exploration"""
    sorted_idx = np.argsort(fitness)
    n_best = max(1, int(self.NP * self.p_best_rate))
    best_indices = sorted_idx[:n_best]
    p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
    p_best = population[p_best_idx]

    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]

    # Compute centroid of top performers for diversity
    top_percentile = max(1, self.NP // 5)
    top_indices = sorted_idx[:top_percentile]
    centroid = np.mean(population[top_indices], axis=0)

    # Adaptive non-linear F scaling: grows super-exponentially with stagnation
    stagnation = self.stagnation_count
    if stagnation > 3:
        # Power-law growth: much more aggressive than linear escalation
        power_factor = (stagnation - 3) ** 1.5
        base_F = min(self.F * (1.0 + 0.3 * power_factor), 3.0)
        # Add heavy-tailed randomness for explosive exploration
        F_i = np.clip(base_F + np.abs(np.random.standard_t(df=3, size=self.NP)) * 0.5, 0.0, 3.0)
    else:
        # Normal operation: bounded random F
        F_i = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)

    # Three candidate donors for composite mutation
    donor_best = population + F_i[:, np.newaxis] * (p_best - population)
    donor_centroid = population + F_i[:, np.newaxis] * (centroid - population)
    donor_rand = population + F_i[:, np.newaxis] * (population[r1] - population[r2])

    # Stagnation-adaptive weights: shift toward exploration when stuck
    progress = min(self.generation / max(1, 100), 1.0)
    w_best_base = 0.6 - 0.3 * progress
    w_centroid_base = 0.25 + 0.2 * progress
    w_rand_base = 0.15 + 0.1 * progress

    if stagnation > 5:
        # When stagnant: dramatically boost exploration components
        boost = min(stagnation / 50.0, 0.5)
        w_best_base = max(0.1, w_best_base - boost)
        w_centroid_base = min(0.6, w_centroid_base + boost * 0.6)
        w_rand_base = min(0.4, w_rand_base + boost * 0.4)

    # Normalize weights
    total_w = w_best_base + w_centroid_base + w_rand_base
    w_best = w_best_base / total_w
    w_centroid = w_centroid_base / total_w
    w_rand = w_rand_base / total_w

    base = (w_best[:, np.newaxis] * donor_best +
            w_centroid[:, np.newaxis] * donor_centroid +
            w_rand[:, np.newaxis] * donor_rand)

    # Random dimension mixing for escaping deceptive local optima
    if stagnation > 2:
        mix_mask = np.random.rand(self.NP, self.dim) < min(0.3, stagnation * 0.01)
        random_sources = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
        base = np.where(mix_mask, random_sources, base)

    # Cultural memory influence (only when mature)
    if self.generation > 10 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        culture_strength = min(0.2, stagnation * 0.005)
        donors = (1 - culture_strength) * base + culture_strength * (population + cultural_contribution)
    else:
        donors = base

    return np.clip(donors, self.lower, self.upper)
```