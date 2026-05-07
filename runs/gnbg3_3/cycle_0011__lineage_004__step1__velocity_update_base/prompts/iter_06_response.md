**Idea: Temporal Autocorrelation with EMA-Driven Velocity Modulation**

Category F: Track centroid drift rate, population spread change, and velocity autocorrelation across generations to modulate inertia and cognitive/social coefficients. Uses exponential moving averages of centroid movement and spread delta to detect exploration vs. exploitation phases, with autocorrelation-based stagnation detection for adaptive inertia.

```python
def _velocity_update_base(self):
    """Velocity update with temporal dynamics tracking (Category F).
    
    Tracks across generations:
    - Centroid drift rate (EMA of centroid displacement)
    - Population spread change rate (EMA of std deviation delta)
    - Velocity autocorrelation (momentum persistence signal)
    
    Uses these temporal signals to modulate inertia, cognitive/social
    coefficients, and mutation intensity based on detected swarm dynamics.
    """
    # === TEMPORAL STATE TRACKING ===
    # Initialize EMA state on first call
    if not hasattr(self, '_ema_centroid_drift'):
        self._ema_centroid_drift = 0.0
        self._ema_spread_change = 0.0
        self._ema_improvement = 0.0
        self._prev_centroid = np.mean(self.population, axis=0)
        self._prev_spread = np.std(self.population)
        self._prev_velocity_mag = np.mean(np.linalg.norm(self.velocity, axis=1))
        self._velocity_history = []
    
    # Compute current centroid and track drift
    current_centroid = np.mean(self.population, axis=0)
    centroid_drift = np.linalg.norm(current_centroid - self._prev_centroid)
    self._prev_centroid = current_centroid.copy()
    
    # Update EMA of drift rate
    self._ema_centroid_drift = 0.2 * centroid_drift + 0.8 * self._ema_centroid_drift
    
    # Track population spread change rate
    current_spread = np.std(self.population)
    spread_delta = abs(current_spread - self._prev_spread)
    self._prev_spread = current_spread
    self._ema_spread_change = 0.2 * spread_delta + 0.8 * self._ema_spread_change
    
    # Velocity autocorrelation for stagnation detection
    current_velocity_mag = np.mean(np.linalg.norm(self.velocity, axis=1))
    if self._prev_velocity_mag > 1e-10:
        autocorr = current_velocity_mag / (self._prev_velocity_mag + 1e-10)
    else:
        autocorr = 1.0
    self._prev_velocity_mag = current_velocity_mag
    
    # Store history for autocorrelation
    self._velocity_history.append(autocorr)
    if len(self._velocity_history) > 10:
        self._velocity_history.pop(0)
    
    # === TEMPORAL-BASED PARAMETER MODULATION ===
    # Normalize temporal signals
    normalized_drift = self._ema_centroid_drift / (current_spread + 1e-10)
    normalized_spread_change = self._ema_spread_change / (current_spread + 1e-10)
    
    # Stagnation detection: low drift + low spread change over time
    stagnation_signal = 1.0 - min(1.0, np.mean(self._velocity_history[-5:]) if len(self._velocity_history) >= 5 else 1.0)
    
    # Adaptive inertia based on temporal signals
    if stagnation_signal > 0.7:
        # Stagnating: boost inertia for exploration
        temporal_inertia = min(0.95, self.inertia_weight + 0.1)
    elif normalized_drift < 0.02:
        # Low drift (converging): reduce inertia
        temporal_inertia = max(0.4, self.inertia_weight - 0.05)
    else:
        # Normal exploration: time-based decay
        temporal_inertia = max(0.4, 0.729 - 0.15 * (self.generation / 3000))
    
    # Adaptive cognitive/social based on temporal dynamics
    cognitive, social = self._adaptive_coefficients()
    
    if stagnation_signal > 0.5:
        # Stagnating: boost cognitive, reduce social (personal search)
        temporal_cog_mult = 1.3
        temporal_social_mult = 0.7
    elif normalized_drift > 0.1 and normalized_spread_change > 0.05:
        # High movement (exploring): balanced
        temporal_cog_mult = 1.0
        temporal_social_mult = 1.0
    elif normalized_drift < 0.02:
        # Converging: boost social for faster convergence
        temporal_cog_mult = 0.8
        temporal_social_mult = 1.2
    else:
        temporal_cog_mult = 1.0
        temporal_social_mult = 1.0
    
    cognitive = cognitive * temporal_cog_mult
    social = social * temporal_social_mult
    
    # === BASE VELOCITY COMPONENTS ===
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # === MUTATION COMPONENT (temporal-adaptive) ===
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    # Increase mutation when stagnating or spread is collapsing
    mutation_base = 0.1 * (1.0 - self.generation / 5000)
    mutation_boost = 0.05 * stagnation_signal + 0.03 * (1.0 - min(1.0, normalized_spread_change))
    mutation_threshold = max(0.01, mutation_base - mutation_boost)
    mutation_active = mutation_mask < mutation_threshold
    
    mutation_vectors = np.zeros((self.np, self.dim))
    for i in range(self.np):
        indices = np.random.choice(
            [j for j in range(self.np) if j != i], 3, replace=False
        )
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    # Boost mutation magnitude when stagnating
    mutation_scale = 0.3 * (1.0 + 0.5 * stagnation_signal)
    mutation_component = np.where(
        mutation_active,
        mutation_scale * (mutation_vectors - self.population),
        0.0
    )
    
    # === COMBINE WITH TEMPORAL INERTIA ===
    new_velocity = (
        temporal_inertia * self.velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```