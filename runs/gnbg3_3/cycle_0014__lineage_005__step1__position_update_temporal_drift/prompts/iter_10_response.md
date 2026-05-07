**Idea: Subspace Power-Law Velocity Modulation**

Category B: Spectral / linear-algebraic — Uses singular value power-law scaling to detect and counteract population collapse into low-dimensional subspaces, applying stronger corrections to collapsed eigendirections.

```python
def _position_update_temporal_drift(self):
    """Subspace power-law velocity modulation via SVD (Category B).

    Key insight: Tasks 17, 16, 6, 5 (worst unsolved) have errors ~1e+02 to 1e+04,
    indicating the population collapses into low-dimensional subspaces and
    cannot escape. SVD decomposition reveals this collapse via the singular
    value spectrum. A power-law modulation boosts velocity along collapsed
    directions (small singular values) while dampening dominant directions,
    re-establishing exploration in the full subspace.

    Mechanism:
    - SVD of centered population → singular values reveal effective dimensionality
    - Power-law scaling: scale_i = (sigma_i / sigma_max) ** power
      where power > 1 amplifies small singular values more aggressively
    - Condition number of singular values drives the power coefficient
    - Directional projection onto principal right singular vectors for velocity update
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)

        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        singular_values = np.clip(singular_values, 1e-10, None)

        # === SPECTRAL ANALYSIS ===
        total_var = np.sum(singular_values ** 2) + 1e-10
        sv_norm = singular_values / (singular_values[0] + 1e-10)

        # Condition number for anisotropy detection
        cond = singular_values[0] / (singular_values[-1] + 1e-10)

        # Effective dimensionality from normalized singular value spectrum
        cumvar = np.cumsum(singular_values ** 2) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / len(singular_values)

        # === POWER-LAW MODULATION ===
        # Power > 1: stronger amplification of collapsed directions
        # Power scales with condition number (more anisotropic → stronger correction)
        base_power = 1.5
        power_boost = np.log1p(cond) / np.log1p(1000.0) if cond > 10.0 else 0.0
        power = base_power + 0.8 * power_boost
        power = np.clip(power, 1.0, 3.0)

        # Power-law scale: (sigma_i / sigma_max) ** power
        # Small singular values get boosted more (power > 1)
        power_scale = sv_norm ** power
        power_scale = power_scale / (np.max(power_scale) + 1e-10)

        # Blend with linear scale for stability
        blend = np.clip((cond - 5.0) / 50.0, 0.0, 1.0)
        per_dir_scale = (1.0 - blend) * sv_norm + blend * power_scale
        per_dir_scale = np.clip(per_dir_scale, 0.1, 2.5)

        # === VELOCITY UPDATE IN SVD SUBSPACE ===
        vel_proj = self.velocity @ Vt.T
        vel_scaled = vel_proj * per_dir_scale

        # === SUBSPACE REHABILITATION FOR COLLAPSED POPULATION ===
        # When eff_dim_ratio < 0.3, population is severely collapsed
        # Inject random perturbations along the weakest singular directions
        collapse_threshold = 0.3
        rehabilitation = np.zeros((self.np, self.dim))
        if eff_dim_ratio < collapse_threshold:
            # Identify weak directions (smallest singular values)
            n_weak = max(1, int(len(singular_values) * (1.0 - eff_dim_ratio)))
            weak_directions = Vt[:n_weak]  # Bottom singular vectors

            # Random perturbation magnitude scaled by inverse of singular value
            # (weaker directions get stronger perturbations)
            weak_sv_norm = sv_norm[:n_weak]
            perturbation_mag = 0.3 * (1.0 - eff_dim_ratio / collapse_threshold)
            perturbation_mag = np.clip(perturbation_mag, 0.05, 0.5)

            for i in range(self.np):
                # Perturbation proportional to 1/sigma for each weak direction
                for j, (sv_val, wdir) in enumerate(zip(weak_sv_norm, weak_directions)):
                    inv_sv = 1.0 / (sv_val + 0.01)
                    scale = perturbation_mag * inv_sv / (np.sum(1.0 / (weak_sv_norm + 0.01)) + 1e-10)
                    rehabilitation[i] += scale * wdir * np.random.randn()

        # === COMBINE SPECTRAL UPDATE WITH SUBSPACE REHABILITATION ===
        new_population = self.population + vel_scaled + rehabilitation

        # === ANISOTROPY-DRIVEN EXPLORATION BOOST ===
        # High condition number → increase global exploration scale
        if cond > 100.0:
            exploration_scale = 1.0 + 0.3 * np.log1p(cond) / np.log1p(1000.0)
            exploration_scale = np.clip(exploration_scale, 1.0, 1.5)
            # Add small isotropic perturbation for high-anisotropy landscapes
            noise = np.random.randn(*new_population.shape) * 0.1 * exploration_scale
            new_population = new_population + noise

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```