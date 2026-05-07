**Idea: Temporal Spectral Collapse Detection via EMA of Condition Number and Singular Value Ratio**

Category F: Temporal / dynamical — tracks condition number and singular value ratios across generations using exponential moving averages to detect population collapse vs healthy convergence. When EMA condition number rises (population collapsing), amplifies all velocity directions; when singular value ratios approach unity (degeneracy), injects isotropic exploration.

```python
def _position_update_svd_whitening(self):
    """SVD-whitening with temporal collapse detection via EMA of condition number.
    
    Category F: Tracks condition number and singular value ratios ACROSS
    GENERATIONS using exponential moving averages. Detects when population
    is collapsing (rising EMA condition) vs converging well (falling EMA).
    Uses rate-of-change of EMA to modulate exploration vs exploitation.
    """
    centered = self.population - np.mean(self.population, axis=0)

    try:
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

        singular_values = np.clip(singular_values, 1e-10, None)

        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        total_var = np.sum(singular_values ** 2) + 1e-10

        # --- TEMPORAL: EMA of condition number across generations ---
        if not hasattr(self, '_ema_cond'):
            self._ema_cond = cond
        alpha_cond = 0.1
        self._ema_cond = alpha_cond * cond + (1 - alpha_cond) * self._ema_cond

        # Detect COLLAPSE: EMA condition is rising → population becoming anisotropic
        cond_ema_diff = cond - self._ema_cond
        is_collapsing = cond_ema_diff > 0.0

        # --- TEMPORAL: EMA of singular value ratios for degeneracy detection ---
        if len(singular_values) >= 2:
            sv_ratio = singular_values[-1] / (singular_values[0] + 1e-10)
            if not hasattr(self, '_ema_sv_ratio'):
                self._ema_sv_ratio = sv_ratio
            alpha_sv = 0.05
            self._ema_sv_ratio = alpha_sv * sv_ratio + (1 - alpha_sv) * self._ema_sv_ratio
            # Degeneracy: ratio approaching 1 means collapsed distribution
            is_degenerate = self._ema_sv_ratio > 0.1
        else:
            is_degenerate = False

        # --- TEMPORAL: Rate of change of EMA (momentum of convergence) ---
        if not hasattr(self, '_prev_ema_cond'):
            self._prev_ema_cond = self._ema_cond
        cond_velocity = self._ema_cond - self._prev_ema_cond
        self._prev_ema_cond = self._ema_cond

        sv_norm = singular_values / (singular_values[0] + 1e-10)

        # Inverse-square-law: stronger effect on collapsed directions
        spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
        spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

        blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
        uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
        per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

        # --- TEMPORAL: Modulate based on collapse detection ---
        if is_collapsing:
            # Population collapsing: increase ALL directions for exploration
            collapse_boost = 1.0 + 0.5 * np.clip(cond_ema_diff / (self._ema_cond + 1e-10), 0.0, 2.0)
            per_component_scale *= collapse_boost
        elif is_degenerate:
            # Degenerate distribution: add isotropic exploration
            per_component_scale += 0.3

        # --- TEMPORAL: Momentum-based damping ---
        # If condition number is accelerating in either direction, dampen
        momentum_factor = 1.0 - 0.2 * np.sign(cond_velocity) * np.clip(abs(cond_velocity) / (self._ema_cond + 1e-10), 0.0, 1.0)
        per_component_scale *= np.clip(momentum_factor, 0.5, 1.5)

        vel_proj = self.velocity @ Vt.T
        vel_scaled = vel_proj * per_component_scale
        new_population = self.population + (vel_scaled @ Vt)

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```