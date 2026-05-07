**Idea: Opposition-Based Hyper-Escape**
Uses opposition-based learning and hyper-random perturbation to escape local optima traps when stagnation occurs — fundamentally different from gradient-following DE mutations.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Opposition-based hyper-escape with intelligent direction weighting"""
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

    # Standard DE component
    base_donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

    # Hyper-escape when stagnant or when fitness is very far from target
    current_best = np.min(fitness)
    is_stagnant = self.stagnation_count > 8
    is_far_stuck = current_best > 1e1  # Tasks with error > 10 need special handling

    if is_stagnant or is_far_stuck:
        # Opposition-based mutation: x_opp = lower + upper - x
        # This reflects solutions across the center of the search space
        opp_best = self.lower + self.upper - self.best_solution

        # Compute direction from current population toward opposition of best
        direction_to_opp = opp_best - population

        # Compute normalized direction toward best solution
        direction_to_best = self.best_solution - population
        dist_to_best = np.linalg.norm(direction_to_best, axis=1, keepdims=True) + 1e-10
        norm_direction_best = direction_to_best / dist_to_best

        # Compute normalized direction from best toward opposition
        direction_opp = opp_best - self.best_solution
        dist_opp = np.linalg.norm(direction_opp) + 1e-10
        norm_direction_opp = direction_opp / dist_opp

        if is_stagnant:
            # Heavy hyper-mutation: jump to opposite region with large perturbation
            hyper_F = min(2.0, self.F * (1.0 + 0.3 * self.stagnation_count))
            escape_component = hyper_F * direction_to_opp

            # Add random hyper-perturbation
            random_scale = min(20.0, 5.0 * np.log1p(self.stagnation_count))
            random_perturbation = np.random.randn(self.NP, self.dim) * random_scale

            # Blend: 60% toward opposition, 30% random, 10% standard
            donors = 0.6 * (population + escape_component) + 0.3 * random_perturbation + 0.1 * base_donors
        else:
            # Far from target but not stagnant: intelligent exploration
            # Weight between going toward best vs exploring opposite region
            exploration_weight = min(0.4, current_best / 1e4)

            # Direction: blend best-seeking with opposition-based exploration
            blended_direction = (1 - exploration_weight) * norm_direction_best + exploration_weight * norm_direction_opp

            # Scale by distance and current F
            scale = self.F * (1.0 + np.log1p(current_best + 1))
            donors = population + scale * blended_direction

            # Add diversity from random individuals
            donors = donors + 0.3 * self.F * (population[r1] - population[r2])

        # Additional opposition jump for worst individuals when very stuck
        if is_far_stuck and current_best > 1e2:
            n_bad = max(1, self.NP // 4)
            bad_indices = sorted_idx[-n_bad:]
            opp_bad = self.lower + self.upper - population[bad_indices]
            donors[bad_indices] = 0.7 * opp_bad + 0.3 * np.random.uniform(self.lower, self.upper, (n_bad, self.dim))
    else:
        # Normal operation: intelligent direction weighting
        if np.isfinite(self.best_fitness) and self.generation > 5:
            denom = abs(self.best_fitness) + 1e-30
            improvement_ratio = (current_best - self.best_fitness) / denom

            # Weight between historical best and current p_best
            w_hist = np.clip(0.7 + max(0, -improvement_ratio) * 0.2, 0.5, 0.9)
            w_curr = 1.0 - w_hist

            # Cultural memory contribution
            if np.any(self.cultural_weights > 0):
                weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
                cultural_centroid = np.dot(weights_norm, self.cultural_memory)
                target = w_hist * self.best_solution + w_curr * p_best + 0.1 * cultural_centroid
            else:
                target = w_hist * self.best_solution + w_curr * p_best

            donors = population + self.F * (target - population) + self.F * (population[r1] - population[r2])
        else:
            donors = base_donors

    return np.clip(donors, self.lower, self.upper)
```