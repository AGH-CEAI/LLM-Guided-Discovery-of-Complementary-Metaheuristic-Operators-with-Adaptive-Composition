**Idea: Rank-Adaptive Velocity with Success-Driven Perturbation**

Uses fitness-rank signals (Spearman correlation, fitness percentiles, success history) to modulate velocity components. Top-ranked particles get amplified cognitive attraction; stagnant particles receive exploratory boosts. Targets worst tasks (17, 16, 6) where premature convergence traps the swarm in poor local optima.

```python
def _velocity_update_base(self):
    """Rank-based velocity update using fitness signals (Category D).
    
    Key mechanisms:
    - Spearman correlation between current and personal-best fitness to detect
      rank stagnation (different from value stagnation)
    - Fitness percentile modulates cognitive component strength
    - Success history drives per-particle exploration boost
    - Rank-distance correlation informs social component scaling
    """
    cognitive, social = self._adaptive_coefficients()
    
    # --- Fitness Rank Signals (Category D only) ---
    pop_size = self.np
    
    # 1. Fitness percentile: where does each particle rank in the swarm?
    current_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, pop_size - 1)
    
    # 2. Personal-best rank (how good was this particle's best historically?)
    pb_ranks = np.argsort(np.argsort(self.personal_best_fitness)) / max(1, pop_size - 1)
    
    # 3. Spearman-like rank correlation between current and personal-best fitness
    #    Low correlation = particle's rank has shifted = exploration happening
    #    High correlation = particle stuck at same rank = stagnation signal
    if pop_size > 2 and np.std(self.current_fitness) > 1e-12 and np.std(self.personal_best_fitness) > 1e-12:
        rank_corr = np.corrcoef(current_ranks, pb_ranks)[0, 1]
        rank_corr = np.clip(rank_corr if not np.isnan(rank_corr) else 0.0, -1.0, 1.0)
    else:
        rank_corr = 0.0
    
    # Stagnation signal: particle's rank hasn't moved much → needs boost
    rank_stagnation = np.abs(current_ranks - pb_ranks)
    
    # 4. Success history per particle (fitness-based, not distance-based)
    if not hasattr(self, '_rank_success_count'):
        self._rank_success_count = np.zeros(pop_size)
    
    improved = self.personal_best_fitness > self.current_fitness
    self._rank_success_count[improved] += 1
    self._rank_success_count[~improved] *= 0.85  # decay
    success_norm = self._rank_success_count / (np.max(self._rank_success_count) + 1.0)
    
    # 5. Fitness-distance rank correlation (Kendall-style signal)
    #    High positive = better fitness → closer to best (smooth landscape)
    #    Low/negative = better fitness → farther from best (rugged/deceptive)
    if self.global_best is not None and pop_size > 2:
        dists_to_best = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists_to_best)) / max(1, pop_size - 1)
        # Kendall-like: count concordant pairs
        concordant = np.sum((current_ranks - pb_ranks) * (dist_ranks - 0.5) > 0)
        fitness_dist_corr = (2 * concordant - pop_size * (pop_size - 1) / 2) / (pop_size * (pop_size - 1) / 2)
        fitness_dist_corr = np.clip(fitness_dist_corr if not np.isnan(fitness_dist_corr) else 0.0, -1.0, 1.0)
    else:
        fitness_dist_corr = 0.0
    
    # --- Rank-adaptive coefficient scaling ---
    # Top fitness percentile → stronger cognitive pull (exploit good solutions)
    # Bottom fitness percentile → weaker cognitive, rely more on social/global
    cognitive_scale = 0.5 + 1.5 * (1.0 - current_ranks)  # top rank = 2.0, bottom = 0.5
    
    # Social component modulated by fitness-distance correlation
    # Rugged landscape (low corr) → stronger social to explore globally
    # Smooth landscape (high corr) → trust local search
    social_scale = 0.5 + 1.5 * (0.5 + 0.5 * fitness_dist_corr)
    
    # Per-particle adaptive scaling
    r1 = np.random.uniform(0, 1, (pop_size, self.dim))
    r2 = np.random.uniform(0, 1, (pop_size, self.dim))
    
    cognitive_component = cognitive * cognitive_scale[:, np.newaxis] * r1 * (self.personal_best - self.population)
    social_component = social * social_scale[:, np.newaxis] * r2 * (self.local_best - self.population)
    
    # --- Success-driven exploration boost ---
    # Particles with low success count get velocity amplification
    exploration_boost = 1.0 + 0.5 * (1.0 - success_norm)
    exploration_boost = np.clip(exploration_boost, 0.5, 2.0)
    
    # --- Rank-stagnation perturbation ---
    # Particles whose rank has been stuck get extra random velocity
    rank_stagnation_boost = 1.0 + 1.0 * (1.0 - rank_stagnation)
    rank_stagnation_boost = np.clip(rank_stagnation_boost, 1.0, 2.5)
    
    # --- DE mutation component with rank-based intensity ---
    mutation_mask = np.random.uniform(0, 1, (pop_size, self.dim))
    # More mutation for stagnant/unsuccessful particles
    mutation_threshold = 0.15 * (1.0 - self.generation / 5000) * (2.0 - success_norm)
    mutation_threshold = np.clip(mutation_threshold, 0.01, 0.2)
    mutation_active = mutation_mask < mutation_threshold[:, np.newaxis]
    
    mutation_vectors = np.zeros((pop_size, self.dim))
    for i in range(pop_size):
        indices = np.random.choice(
            [j for j in range(pop_size) if j != i], 3, replace=False
        )
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    mutation_intensity = 0.4 * (1.0 + 0.5 * (1.0 - success_norm))
    mutation_component = np.where(
        mutation_active,
        mutation_intensity[:, np.newaxis] * (mutation_vectors - self.population),
        0.0
    )
    
    # --- Global velocity update ---
    new_velocity = (
        self.inertia_weight * self.velocity * exploration_boost[:, np.newaxis] * rank_stagnation_boost[:, np.newaxis] +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```