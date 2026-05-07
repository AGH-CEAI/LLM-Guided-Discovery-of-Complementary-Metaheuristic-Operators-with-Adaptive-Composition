**Idea: Temporal Drift Detection with EMA Modulation**
Category F (Temporal / dynamical): Track centroid drift, velocity EMA, improvement EMA, and diversity EMA across generations to detect premature convergence and inject exploration when stagnation is detected. Targets worst tasks (17, 16, 6, 11) where large errors indicate the population is stuck.

```python
def _position_update_fitness_rank(self):
    """Temporal dynamics-based velocity modulation (Category F).

    Key insight: Track temporal signals ACROSS GENERATIONS to detect premature
    convergence. Uses centroid drift rate, velocity EMA decay, improvement EMA,
    and diversity EMA to detect when population is stagnating. When stagnation
    is detected, injects exploration via temporal perturbation and velocity boost.
    Targets worst tasks (17=5e4, 16=1.7e3, 6=7e2) where large errors indicate
    the algorithm is stuck in local optima.
    """
    # --- TEMPORAL SIGNALS TRACKED ACROSS GENERATIONS ---
    
    # 1. Centroid drift tracking
    current_centroid = np.mean(self.population, axis=0)
    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = current_centroid.copy()
        self._centroid_drift = np.zeros(self.dim)
    self._centroid_drift = current_centroid - self._ema_centroid
    # EMA update (smoothing across generations)
    self._ema_centroid = 0.1 * current_centroid + 0.9 * self._ema_centroid
    
    # 2. Velocity magnitude EMA (across generations)
    vel_mag = np.linalg.norm(self.velocity, axis=1)
    vel_mag_mean = np.mean(vel_mag)
    if not hasattr(self, '_ema_velocity_mag'):
        self._ema_velocity_mag = vel_mag_mean
    self._ema_velocity_mag = 0.95 * self._ema_velocity_mag + 0.05 * vel_mag_mean
    
    # 3. Global improvement rate EMA (across generations)
    if hasattr(self, '_prev_global_fitness'):
        improvement = max(0.0, self._prev_global_fitness - self.global_best_fitness)
    else:
        improvement = 0.0
    self._prev_global_fitness = self.global_best_fitness
    
    if not hasattr(self, '_ema_improvement'):
        self._ema_improvement = 0.0
    self._ema_improvement = 0.9 * self._ema_improvement + 0.1 * improvement
    
    # 4. Diversity EMA (across generations)
    current_diversity = self._compute_diversity()
    if not hasattr(self, '_ema_diversity'):
        self._ema_diversity = current_diversity
    self._ema_diversity = 0.95 * self._ema_diversity + 0.05 * current_diversity
    
    # 5. Stagnation counter EMA
    if not hasattr(self, '_ema_stagnation'):
        self._ema_stagnation = 0.0
    self._ema_stagnation = 0.9 * self._ema_stagnation + 0.1 * float(self.stagnation_counter)
    
    # --- STAGNATION DETECTION FROM TEMPORAL SIGNALS ---
    drift_mag = np.linalg.norm(self._centroid_drift) + 1e-10
    diversity_ratio = current_diversity / (self._ema_diversity + 1e-10)
    
    # Stagnation flags from temporal signals
    is_stagnant = (
        self._ema_improvement < 1e-8 or
        self._ema_velocity_mag < 0.01 * self.v_max or
        self._ema_diversity < 1e-5 or
        self._ema_stagnation > 30 or
        (drift_mag < 0.1 and self.generation > 10)
    )
    
    # --- TEMPORAL VELOCITY MODULATION ---
    if is_stagnant:
        # Inject exploration: boost velocity magnitude via temporal perturbation
        temporal_boost = 1.5 + 0.5 * np.clip(self._ema_stagnation / 100.0, 0.0, 1.0)
        vel_scale = np.full(self.np, temporal_boost)
        
        # Temporal perturbation toward unexplored regions (anticipate centroid movement)
        temporal_perturb = 0.3 * self._centroid_drift[np.newaxis, :] * temporal_boost
        temporal_perturb += 0.4 * np.random.uniform(-1, 1, (self.np, self.dim)) * current_diversity
    else:
        # Normal operation: scale velocity based on temporal improvement signals
        improvement_factor = np.clip(self._ema_improvement * 1000.0, 0.0, 1.0)
        
        # Diversity-aware scaling: low diversity → more exploration
        diversity_scale = np.clip(self._ema_diversity / max(current_diversity, 1e-10), 0.5, 1.5)
        
        vel_scale = np.full(self.np, 1.0)
        vel_scale *= (1.0 + 0.3 * (1.0 - improvement_factor))
        vel_scale *= diversity_scale
        
        temporal_perturb = 0.1 * self._centroid_drift[np.newaxis, :]
        temporal_perturb += 0.1 * np.random.uniform(-1, 1, (self.np, self.dim)) * current_diversity
    
    vel_scale = np.clip(vel_scale, 0.3, 2.5)
    
    # --- FITNESS RANK SIGNAL (Category D, blended) ---
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # --- DIRECTIONAL PERTURBATION ---
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        directional = 0.3 * (1.0 - fitness_ranks[:, np.newaxis]) * to_best_dir
    else:
        directional = 0.3 * (1.0 - fitness_ranks[:, np.newaxis]) * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # --- COMBINE ALL SIGNALS ---
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + temporal_perturb + directional
    self.population = self._clip_to_bounds(new_population)
```