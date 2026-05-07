**Idea: Temporal Momentum & Diversity-Drift Velocity Update**

I will track exponential moving averages of velocity direction and swarm diversity across generations to detect convergence regimes. The key insight from the worst tasks (errors 1e+02 to 1e+04) is that the swarm is getting stuck in local optima or losing momentum — temporal analysis of velocity drift and diversity decay will let me inject exploration bursts exactly when the swarm starts oscillating or stagnating.

```python
def _velocity_update_batch(self):
    """Update velocities using temporal momentum and diversity-drift detection."""
    # Initialize temporal state if not present
    if not hasattr(self, 'velocity_ema'):
        self.velocity_ema = np.zeros((self.np, self.dim))
        self.diversity_history = []
        self.velocity_magnitude_ema = 1.0
        self.prev_diversity = None
        self.ema_alpha = 0.3  # Smoothing factor for temporal tracking
    
    # Track current velocity magnitude for momentum analysis
    current_vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True) + 1e-10
    
    # Update exponential moving average of velocity magnitude
    self.velocity_magnitude_ema = (
        self.ema_alpha * np.mean(current_vel_mag) +
        (1 - self.ema_alpha) * self.velocity_magnitude_ema
    )
    
    # Update EMA of velocity vectors (temporal smoothed velocity)
    self.velocity_ema = (
        self.ema_alpha * self.velocity +
        (1 - self.ema_alpha) * self.velocity_ema
    )
    
    # Compute diversity and track its rate of change (temporal signal)
    current_diversity = self._compute_diversity()
    self.diversity_history.append(current_diversity)
    if len(self.diversity_history) > 20:
        self.diversity_history.pop(0)
    
    # Compute diversity rate of change (diversity drift)
    diversity_drift = 0.0
    if len(self.diversity_history) >= 5:
        recent_window = self.diversity_history[-5:]
        diversity_drift = (recent_window[-1] - recent_window[0]) / (len(recent_window) + 1e-10)
    
    # Detect velocity momentum (autocorrelation of velocity direction)
    vel_dot_ema = np.sum(self.velocity * self.velocity_ema, axis=1, keepdims=True)
    vel_mag_product = current_vel_mag * (np.linalg.norm(self.velocity_ema, axis=1, keepdims=True) + 1e-10)
    momentum_signal = np.clip(vel_dot_ema / (vel_mag_product + 1e-10), -1, 1)
    avg_momentum = np.mean(momentum_signal)
    
    # Compute adaptive inertia weight using temporal signals
    base_inertia = self.inertia_weight
    
    # Diversity-drift correction: positive drift = converging, need more exploration
    if diversity_drift < -1e-4:  # Diversity decreasing rapidly
        inertia_adjustment = 1.15  # Boost exploration
    elif diversity_drift > 1e-4:  # Diversity increasing
        inertia_adjustment = 0.85  # Boost exploitation
    else:
        inertia_adjustment = 1.0
    
    # Momentum correction: low momentum = oscillating or stuck, need boost
    if avg_momentum < 0.3:
        momentum_correction = 1.2  # Increase exploration when stuck
    elif avg_momentum > 0.8:
        momentum_correction = 0.9  # Reduce when moving consistently
    else:
        momentum_correction = 1.0
    
    # Apply temporal corrections to inertia
    temporal_inertia = base_inertia * inertia_adjustment * momentum_correction
    temporal_inertia = np.clip(temporal_inertia, 0.2, 1.1)
    
    # Adaptive cognitive/social coefficients with temporal boost
    improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
    cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
    social = self.social_base * (2.0 - improvement_ratio)
    
    # Temporal momentum term: dampen oscillation, maintain direction
    momentum_factor = 0.15 * (1.0 - avg_momentum)
    temporal_momentum = momentum_factor * self.velocity_ema
    
    # Generate random matrices
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Social component
    social_component = social * r2 * (self.local_best - self.population)
    
    # DE mutation with temporal threshold adjustment
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    # Increase mutation when stuck or oscillating
    mutation_boost = 1.0 + max(0, 0.5 - avg_momentum) + min(0.5, self.stagnation_counter / 200)
    mutation_threshold = 0.1 * mutation_boost * (1.0 - self.generation / 5000)
    mutation_active = mutation_mask < mutation_threshold
    
    # Generate mutation vectors
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
    
    # Velocity update with temporal momentum
    new_velocity = (
        temporal_inertia * self.velocity +
        cognitive_component +
        social_component +
        temporal_momentum +
        mutation_component
    )
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```