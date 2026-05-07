**Idea: SVD-Rank Collapse Detection with Reciprocal Singular Value Scaling**

This variant replaces eigendecomposition with direct SVD analysis, using numerical rank estimation and reciprocal singular value scaling to aggressively counteract population collapse into low-dimensional subspaces — a distinct mechanism from the current eigenvalue-based approach.
```python
def _position_update_temporal_drift(self):
    """SVD-based rank collapse detection with reciprocal singular scaling (Category B).

    Uses SVD instead of eigendecomposition: numerical rank detection identifies when
    population collapses to a subspace. Reciprocal singular value scaling (1/sv)
    amplifies velocity in collapsed directions more aggressively than sqrt(sv).
    Tracks condition number and effective dimensionality to detect and escape
    rank-deficient states on the worst unsolved tasks (17, 16, 6, 11).
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        singular_values = np.clip(singular_values, 1e-10, None)

        cond = singular_values[0] / (singular_values[-1] + 1e-10)

        total_var = np.sum(singular_values ** 2) + 1e-10
        cumvar = np.cumsum(singular_values ** 2) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim

        sv_norm = singular_values / (singular_values[0] + 1e-10)

        spectral_entropy = -np.sum((sv_norm ** 2) * np.log(sv_norm ** 2 + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

        spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_spectral_ema_improvement'):
            self._spectral_ema_improvement = 0.0
        self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement

        if self._spectral_ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._spectral_ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7

        temporal_scale = base_scale * (1.0 + 0.4 * spectral_signal)

        if cond > 50.0:
            log_damp = np.log1p(cond) / np.log1p(500.0)
            log_damp = np.clip(log_damp, 0.0, 1.0)
            temporal_scale *= (1.0 + 0.4 * log_damp)

        reciprocal_sv = 1.0 / (sv_norm + 0.01)
        reciprocal_sv = reciprocal_sv / (np.max(reciprocal_sv) + 1e-10)
        per_comp_scale = np.clip(reciprocal_sv, 0.1, 3.0)

        vel_proj = self.velocity @ Vt.T
        vel_scaled = vel_proj * per_comp_scale
        new_population = self.population + temporal_scale * (vel_scaled @ Vt)

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```