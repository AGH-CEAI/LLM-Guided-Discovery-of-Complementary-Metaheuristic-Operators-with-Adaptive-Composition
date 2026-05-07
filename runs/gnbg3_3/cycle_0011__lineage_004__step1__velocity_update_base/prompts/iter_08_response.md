Looking at the worst unsolved tasks (17, 16, 6, 11, 19, 9, 23, 18), the common failure mode appears to be premature convergence and loss of exploration. The current velocity update uses static cognitive/social coefficients and time-decaying DE mutation, but lacks dynamic balance between exploration and exploitation signals.

**Idea: Momentum-Guided Rank-Weighted Velocity**
Combine TWO distinct mechanisms in the velocity update:
1. **Temporal momentum tracking**: Exponentially-weighted velocity direction consistency to detect/prevent oscillations and stagnation
2. **Fitness landscape rank-weighting**: Use Spearman rank correlation between particle fitness and distance to global best to modulate social attraction strength

Switching logic: When momentum EMA is low (oscillating/stuck), increase inertia and reduce social attraction; when momentum EMA is high (steady), trust rank-weighted social signals. This is data-driven based on actual velocity behavior.

```python
def _velocity_update_base(self):
    """Hybrid velocity update: momentum tracking + fitness-rank-weighted social attraction."""
    # === MECHANISM 1: TEMPORAL MOMENTUM TRACKING ===
    # Track velocity direction consistency via EMA of velocity magnitude
    velocity_magnitude = np.linalg.norm(self.velocity, axis=1, keepdims=True) + 1e-10
    if not hasattr(self, '_ema_velocity_mag'):
        self._ema_velocity_mag = velocity_magnitude.mean()
    self._ema_velocity_mag = 0.7 * self._ema_velocity_mag + 0.3 * velocity_magnitude.mean()
    
    # Normalized momentum signal: current mag vs EMA (detects oscillation/damping)
    momentum_ratio = velocity_magnitude.mean() / (self._ema_velocity_mag + 1e-10)
    momentum_signal = np.clip(momentum_ratio, 0.1, 3.0)
    
    # === MECHANISM 2: FITNESS-LANDSCAPE RANK-WEIGHTED SOCIAL ATTRACTION ===
    # Spearman-like rank correlation: fitness rank vs distance to global best
    if self.global_best is not None and self.np > 2:
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        dists_to_best = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists_to_best)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            rank_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            rank_corr = 0.0
    else:
        rank_corr = 0.0
    
    # Rank-weighted social coefficient: better-fitness particles attract more
    # High rank_corr (>0.5) = fitness correlates with distance → social attraction helps
    # Low rank_corr (<0.2) = rugged/noisy landscape → reduce social, increase exploration
    social_weight_factor = np.clip(2.0 * rank_corr, 0.3, 1.5)
    
    # === PRINCIPLED SWITCHING LOGIC ===
    # Data-driven: momentum_ratio drives exploration/exploitation balance
    is_oscillating = momentum_signal < 0.6  # Velocity collapsing = stuck/oscillating
    is_expanding = momentum_signal > 1.4    # Velocity growing = exploring
    
    if is_oscillating:
        # Dampen oscillations: increase inertia, reduce social attraction
        inertia_boost = 1.3
        social_dampen = 0.6
    elif is_expanding:
        # Steady exploration: moderate inertia, trust rank-weighted social
        inertia_boost = 1.0
        social_dampen = 1.0
    else:
        # Balanced: slight inertia increase, full rank-weighted social
        inertia_boost = 1.1
        social_dampen = 0.9
    
    # === COEFFICIENT COMPUTATION ===
    cognitive, social_base = self._adaptive_coefficients()
    social = social_base * social_weight_factor * social_dampen
    
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # === MUTATION WITH MOMENTUM-MODULATED INTENSITY ===
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    # More mutation when oscillating (stuck) or expanding (exploring)
    mutation_intensity = 0.1 * (1.0 + 0.5 * abs(momentum_signal - 1.0))
    mutation_intensity *= (1.0 - self.generation / 5000)
    mutation_active = mutation_mask < mutation_intensity
    
    mutation_vectors = np.zeros((self.np, self.dim))
    for i in range(self.np):
        indices = np.random.choice(
            [j for j in range(self.np) if j != i], 3, replace=False
        )
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    mutation_component = np.where(
        mutation_active,
        0.3 * (mutation_vectors - self.population),
        0.0
    )
    
    # === COMBINE WITH MOMENTUM-MODULATED INERTIA ===
    effective_inertia = self.inertia_weight * inertia_boost
    effective_inertia = np.clip(effective_inertia, 0.3, 1.1)
    
    new_velocity = (
        effective_inertia * self.velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```