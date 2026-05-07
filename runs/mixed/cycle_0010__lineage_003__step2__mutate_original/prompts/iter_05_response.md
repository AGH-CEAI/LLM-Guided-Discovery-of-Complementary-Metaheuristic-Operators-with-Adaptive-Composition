**Idea: Opposition-Guided DE with Adaptive Reflection Probability**

This variant uses opposition-based learning (OBL) as a fundamentally different exploration mechanism. For the worst unsolved tasks (errors ~1e+2 to 1e+3), the standard difference-vector approach gets trapped in poor local optima. OBL provides a complementary exploration direction by explicitly reflecting solutions across the search space center, combined with quasi-opposition sampling for additional diversity. The adaptive reflection probability starts high (aggressive exploration) and decays, while dimension-wise random walks add multi-directional perturbation.

```python
def _mutate_original(self, population, fitness):
    """Opposition-Guided DE with Adaptive Reflection and Quasi-Opposition"""
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

    # Standard base mutation
    base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

    # Opposition-based learning: reflect solutions across search space center
    # Adaptive reflection probability (high at start, decays over generations)
    decay_rate = 0.01
    reflect_prob = max(0.05, 0.8 * np.exp(-decay_rate * self.generation))
    
    # Search space center
    center = (self.lower + self.upper) / 2.0
    
    # Generate opposite candidates (full reflection)
    opposite_full = 2.0 * center - base
    
    # Generate quasi-opposition candidates (between center and opposite)
    quasi_opposite = np.random.uniform(center, opposite_full)
    
    # Generate quasi-reflected (between center and base)
    quasi_reflected = np.random.uniform(center, base)
    
    # Dimension-wise random walk for additional exploration
    dim_perturb = np.random.randn(self.NP, self.dim) * (0.1 + 0.3 * np.exp(-0.005 * self.generation))
    
    # Decide which candidates to use per individual
    rand_mask = np.random.rand(self.NP, self.dim)
    
    # Apply different strategies to different dimensions for diversity
    donors = base.copy()
    reflect_mask = np.random.rand(self.NP) < reflect_prob
    n_reflect = int(np.sum(reflect_mask))
    
    if n_reflect > 0:
        reflect_indices = np.where(reflect_mask)[0]
        # Half use full opposition, half use quasi-opposition
        half = n_reflect // 2
        full_opp_indices = reflect_indices[:half]
        quasi_opp_indices = reflect_indices[half:]
        
        donors[full_opp_indices] = opposite_full[full_opp_indices]
        donors[quasi_opp_indices] = quasi_opposite[quasi_opp_indices]
    
    # Apply dimension-wise perturbation to non-reflected individuals
    non_reflect_indices = np.where(~reflect_mask)[0]
    if len(non_reflect_indices) > 0:
        donors[non_reflect_indices] = donors[non_reflect_indices] + dim_perturb[non_reflect_indices]
        # Mix with quasi-reflected for some individuals
        mix_mask = np.random.rand(len(non_reflect_indices)) < 0.3
        mix_indices = non_reflect_indices[mix_mask]
        if len(mix_indices) > 0:
            donors[mix_indices] = 0.7 * donors[mix_indices] + 0.3 * quasi_reflected[mix_indices]

    # Blend with cultural memory if available
    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_contribution = np.dot(weights_norm, self.cultural_memory)
        similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                               for i in range(self.NP)])
        sim_std = np.std(similarity) + 1e-10
        weights_culture = np.exp(-similarity / sim_std)
        weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
        cultural_offset = cultural_contribution * 0.15
        donors = (1 - 0.15) * donors + 0.15 * (population + cultural_offset)

    return np.clip(donors, self.lower, self.upper)
```