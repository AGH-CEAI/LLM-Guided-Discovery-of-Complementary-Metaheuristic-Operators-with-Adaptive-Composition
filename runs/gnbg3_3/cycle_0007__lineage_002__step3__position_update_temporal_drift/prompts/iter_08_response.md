**Idea: Temporal-Drift + Diversity-Driven Exploration Switching**

Category H hybrid: Combines EMA-based temporal drift detection (from current approach) with diversity-triggered exploration bursts. The switch is data-driven: when improvement stagnates AND population collapses (low diversity), switch to diversity-driven exploration mode. This targets the worst tasks (17, 16, 6) where the optimizer gets stuck in poor regions with collapsed populations.

```python
def _position_update_temporal_drift(self):
    """Hybrid: temporal-drift detection + diversity-triggered exploration switching.
    
    Category H: Two distinct mechanisms with principled data-driven switching.
    Mechanism 1: EMA-based temporal dynamics (improvement rate, centroid autocorrelation)
    Mechanism 2: Diversity-driven exploration (population spread, random perturbations)
    Switching trigger: stagnation + low diversity (data-driven, not arbitrary)
    """
    # --- Mechanism 1: Temporal drift signals (from original approach) ---
    centroid = np.mean(self.population, axis=0)
    
    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = centroid.copy()
        self._ema_centroid_velocity = np.zeros(self.dim)
    
    alpha_pos = 0.1
    self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid
    
    current_centroid_vel = centroid - self._ema_centroid
    alpha_vel = 0.2
    self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity
    
    if not hasattr(self, '_centroid_vel_history'):
        self._centroid_vel_history = []
    self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
    if len(self._centroid_vel_history) > 20:
        self._centroid_vel_history.pop(0)
    
    if len(self._centroid_vel_history) >= 5:
        recent = np.array(self._centroid_vel_history[-5:])
        norm_a = np.linalg.norm(recent[:-1]) + 1e-10
        norm_b = np.linalg.norm(recent[1:]) + 1e-10
        autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
    else:
        autocorr = 0.0
    
    # Improvement tracking
    if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
        improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
    else:
        improvement = 0.0
    self._prev_global_best_fitness = self.global_best_fitness
    
    if not hasattr(self, '_ema_improvement'):
        self._ema_improvement = 0.0
    self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement
    
    convergence_signal = np.clip(autocorr, -1.0, 1.0)
    
    # --- Mechanism 2: Diversity signals ---
    centroid_for_div = np.mean(self.population, axis=0)
    dists_to_centroid = np.linalg.norm(self.population - centroid_for_div, axis=1)
    diversity = np.mean(dists_to_centroid)
    max_dist = np.max(dists_to_centroid) + 1e-10
    diversity_normalized = diversity / max_dist
    
    # --- Principled switching logic (data-driven) ---
    is_stagnant = self._ema_improvement < 1e-8
    is_low_diversity = diversity_normalized < 0.3
    use_exploration_mode = is_stagnant and is_low_diversity
    
    if use_exploration_mode:
        # EXPLORATION MODE: diversity-driven perturbations
        if not hasattr(self, '_exploration_counter'):
            self._exploration_counter = 0
        self._exploration_counter += 1
        
        # Decay exploration intensity over consecutive exploration gens
        explore_intensity = np.exp(-0.1 * self._exploration_counter)
        explore_intensity = np.clip(explore_intensity, 0.2, 1.0)
        
        # Large velocity scaling for escape
        vel_scale = 1.5 + 0.5 * (1.0 - convergence_signal)
        vel_scale = np.clip(vel_scale, 1.0, 2.5)
        
        # Diversity-directed random perturbations
        perturbation_scale = explore_intensity * 2.0 * (self.upper_bound - self.lower_bound) * 0.01
        random_perturb = np.random.uniform(-perturbation_scale, perturbation_scale, self.population.shape)
        
        # Direction toward unexplored regions (away from centroid)
        to_centroid = centroid_for_div - self.population
        to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        away_from_centroid = -to_centroid / to_centroid_norm
        directional_boost = explore_intensity * 0.5 * away_from_centroid * diversity_normalized
        
        new_population = self.population + vel_scale * self.velocity + random_perturb + directional_boost
    else:
        # EXPLOITATION MODE: temporal-drift-modulated scaling
        if hasattr(self, '_exploration_counter'):
            self._exploration_counter = 0
        
        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        temporal_scale = base_scale * (1.0 + 0.4 * convergence_signal)
        
        vel_magnitude = np.linalg.norm(self._ema_centroid_velocity) + 1e-10
        vel_factor = np.clip(1.0 / (1.0 + vel_magnitude * 0.005), 0.5, 1.5)
        temporal_scale *= vel_factor
        
        new_population = self.population + temporal_scale * self.velocity
    
    # --- Spectral conditioning (shared component) ---
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, _ = np.linalg.eigh(cov)
        cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
        spectral_factor = np.clip(cond / 100.0, 0.5, 2.0)
        
        if use_exploration_mode:
            temporal_scale = 1.0
        temporal_scale = temporal_scale * spectral_factor
        new_population = self.population + temporal_scale * (new_population - self.population)
    except np.linalg.LinAlgError:
        pass
    
    self.population = self._clip_to_bounds(new_population)
```