**Idea: Mean-Reversion Opposition DE**

The worst unsolved tasks (12, 8, 16, 21, 14, 20, 23, 13) have errors in the range 1e+00 to 1e+03, indicating the population is trapped in poor local optima. All current variants rely on `p_best` or `global_best` as the primary mutation target — but when the best individuals themselves are far from global optimality, this creates a "follow-the-leader" trap. The solution: break free using **mean-reversion** (pull toward population centroid to counter premature convergence) combined with **opposition-based learning** (reflect through search-space center to escape to the opposite side).

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Mean-Reversion Opposition DE — escape local optima via centroid pull and opposition"""
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

    # Compute population centroid (geometric center of search)
    centroid = np.mean(population, axis=0)

    # Compute search-space center and opposition vector
    search_center = 0.5 * (self.lower + self.upper)
    opposition_vector = 2.0 * search_center - population

    # Adaptive stagnation detection
    is_stagnant = self.stagnation_count > 10
    stagnation_factor = min(self.stagnation_count / 50.0, 1.0) if is_stagnant else 0.0

    # Mean-reversion weight: increases when stagnant to pull away from converged cluster
    w_mean = 0.1 + 0.4 * stagnation_factor

    # Opposition weight: only significant when stagnant (escape to opposite side)
    w_opp = 0.3 * stagnation_factor

    # Standard DE components (always present for exploitation)
    w_best = 0.5
    w_diff = 0.5

    # Build composite donor
    base_direction = w_best * (p_best - population) + w_diff * (population[r1] - population[r2])
    mean_direction = w_mean * (centroid - population)
    opp_direction = w_opp * (opposition_vector - population)

    # Combine with adaptive F
    F_base = self.F
    F_escape = np.clip(F_base * (1.0 + stagnation_factor * 0.5), 0.5, 2.0)

    donors = population + F_escape * (base_direction + mean_direction + opp_direction)

    # Clamp to bounds and return
    return np.clip(donors, self.lower, self.upper)
```