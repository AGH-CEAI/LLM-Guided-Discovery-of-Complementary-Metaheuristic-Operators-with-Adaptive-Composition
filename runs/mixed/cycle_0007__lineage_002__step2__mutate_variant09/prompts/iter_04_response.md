Looking at the error patterns, I notice that Tasks 20 and 21 are **stuck at exactly 5.0** across all variants—this is a strong signal of convergence to a specific local attractor. The worst tasks (12, 8, 16) have errors >500, indicating fundamental failure to make progress. All existing variants use **single-differential mutation vectors**, which limits exploration directions and makes it easy to get trapped.

**Idea: Multi-Donor Composite Mutation with Diversity-Weighted Selection**

This variant generates THREE candidate donors per individual using different directional strategies (best-directed, centroid-directed, random-difference), then composites them using diversity as the selection proxy (no extra fitness evaluations needed). This fundamentally differs from single-vector approaches by exploring multiple promising directions simultaneously and adaptively weighting them.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Multi-Donor Composite Mutation with Diversity-Weighted Selection"""
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

    # Compute centroid of top 20% as exploration target
    top_percentile = max(1, self.NP // 5)
    top_indices = sorted_idx[:top_percentile]
    centroid = np.mean(population[top_indices], axis=0)

    # Generate THREE candidate donors per individual
    # Candidate 1: Best-directed (exploitation)
    donor_best = population + self.F * (p_best - population)
    # Candidate 2: Centroid-directed (diversification)
    donor_centroid = population + self.F * (centroid - population)
    # Candidate 3: Standard rand/diff (random exploration)
    donor_rand = population + self.F * (population[r1] - population[r2])

    # Diversity-weighted composite (no extra fitness evals needed)
    # Higher diversity contribution = more different from parent = better explorer
    dist_best = np.linalg.norm(donor_best - population, axis=1) + 1e-10
    dist_centroid = np.linalg.norm(donor_centroid - population, axis=1) + 1e-10
    dist_rand = np.linalg.norm(donor_rand - population, axis=1) + 1e-10

    # Normalize diversity scores
    total_dist = dist_best + dist_centroid + dist_rand + 1e-10
    w_best = dist_best / total_dist
    w_centroid = dist_centroid / total_dist
    w_rand = dist_rand / total_dist

    # Adaptive weight adjustment based on stagnation
    if self.stagnation_count > 10:
        # When stuck: boost centroid exploration, reduce best-pull
        w_best = np.clip(w_best * 0.5, 0.1, 0.6)
        w_centroid = np.clip(w_centroid * 1.5, 0.2, 0.6)
        w_rand = np.clip(w_rand * 1.2, 0.1, 0.4)
        # Renormalize
        total_w = w_best + w_centroid + w_rand
        w_best /= total_w
        w_centroid /= total_w
        w_rand /= total_w

    # Composite donors
    donors = (w_best[:, np.newaxis] * donor_best +
              w_centroid[:, np.newaxis] * donor_centroid +
              w_rand[:, np.newaxis] * donor_rand)

    return np.clip(donors, self.lower, self.upper)
```