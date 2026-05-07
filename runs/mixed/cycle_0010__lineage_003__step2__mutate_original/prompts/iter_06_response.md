**Idea: Opposition-Guided Escape with Adaptive Basin Hopping**
Uses dynamic opposition-based learning to escape deceptive local optima. When stagnation is high, computes opposition vectors (reflection around centroid) and probabilistically selects between mutation direction and opposition direction. This enables large-scale jumps across basins of attraction for the worst unsolved tasks.

```python
def _mutate_original(self, population, fitness):
    """Opposition-Guided Escape: basin-hopping via dynamic opposition vectors"""
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

    # Compute centroid for opposition reference
    centroid = np.mean(population, axis=0)

    # Adaptive F: larger when stagnant to escape basins
    if self.stagnation_count > 10:
        F_escape = min(self.F * (1.0 + 0.5 * np.log1p(self.stagnation_count)), 2.5)
    else:
        F_escape = self.F

    # Standard mutation direction (exploitation)
    mutation_dir = p_best - population + F_escape * (population[r1] - population[r2])

    # Opposition direction: reflect around centroid (exploration)
    # Opposition of x is x_opp = centroid + (centroid - x) = 2*centroid - x
    opposition_vector = 2.0 * centroid - population
    opposition_dir = opposition_vector - population + F_escape * (population[r2] - population[r1])

    # Adaptive selection: more opposition when stagnant
    # Start conservative, increase opposition probability with stagnation
    base_opp_prob = 0.1
    stagnation_boost = min(0.6, 0.05 * self.stagnation_count)
    opp_prob = np.clip(base_opp_prob + stagnation_boost, 0.0, 0.7)

    use_opposition = np.random.rand(self.NP) < opp_prob

    # Compute donors for both directions
    donor_exploit = population + mutation_dir
    donor_explore = population + opposition_dir

    # Select based on adaptive probability
    donors = np.where(use_opposition[:, np.newaxis], donor_explore, donor_exploit)

    # Small perturbation when very stagnant (adds randomization for escape)
    if self.stagnation_count > 20:
        perturbation_scale = min(0.5 * np.log1p(self.stagnation_count - 20), 5.0)
        random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
        donors = donors + random_perturbation

    return np.clip(donors, self.lower, self.upper)
```