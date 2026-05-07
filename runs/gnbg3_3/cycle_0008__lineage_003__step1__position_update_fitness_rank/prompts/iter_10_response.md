Looking at the current implementation, it uses graph-based (Category E) mechanisms, but I need to replace it with a Category B (Spectral / linear-algebraic) approach. The worst unsolved tasks (17, 16, 6, 11) have errors in the 1e+02 to 1e+04 range, suggesting severe anisotropy or population collapse into subspaces.

Category B analysis: The key spectral property to exploit is the **condition number of the population covariance**. High condition number = population elongated along a few axes = risk of premature convergence. My different angle from the previous Category B variant (which used SVD inverse-square-law): I'll use **conditional anisotropic damping based on condition number thresholding** — when anisotropy is detected, apply per-eigenvector velocity damping that preferentially boosts exploration in collapsed directions.

**Idea: Condition-Number-Gated Anisotropic Velocity Damping**

```python
def _position_update_fitness_rank(self):
    """Condition-number-gated anisotropic velocity damping (Category B).

    Key insight: Use condition number of population covariance as a gating
    signal. When cond > threshold (high anisotropy), apply per-eigenvector
    damping that boosts velocity in collapsed directions. When cond is
    moderate, use uniform scaling. This is orthogonal to SVD-whitening
    (which always applies inverse-square-law) and spectral-entropy (which
    uses distribution entropy as signal). Directly targets worst tasks
    with large condition numbers where population collapses to subspaces.
    """
    # Fitness rank signal (from Category D baseline)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    # Spectral analysis of population covariance
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_log = np.log1p(cond)

        # Effective dimensionality for adaptive scaling
        total_var = np.sum(eigenvalues) + 1e-10
        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim

        # Condition-number-gated modulation
        cond_threshold = 50.0
        if cond > cond_threshold:
            # High anisotropy: per-eigenvector damping
            sv_scale = np.sqrt(eigenvalues)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            # Inverse scaling: boost collapsed directions
            per_comp_scale = 1.0 / (sv_norm + 0.05)
            per_comp_scale = per_comp_scale / (np.max(per_comp_scale) + 1e-10)
            # Blend with uniform based on severity
            blend = np.clip((cond - cond_threshold) / 200.0, 0.0, 1.0)
            uniform_scale = np.clip(cond_log / np.log1p(1000.0), 0.5, 2.0)
            final_scale = uniform_scale * (1.0 - blend) + per_comp_scale * blend
        else:
            # Moderate anisotropy: uniform scaling
            final_scale = np.clip(cond_log / np.log1p(100.0), 0.7, 1.5)

        # Project velocity to eigenspace, apply per-component scaling
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * final_scale

        # Effective-dim-based exploration bonus
        explore_boost = 1.0 + 0.3 * (1.0 - eff_dim_ratio)

        new_population = self.population + explore_boost * (vel_scaled @ eigenvectors.T)

    except np.linalg.LinAlgError:
        # Fallback: uniform velocity update
        new_population = self.population + self.velocity

    # Fitness-based directional perturbation
    fitness_perturb = 0.4 * (1.0 - fitness_ranks[:, np.newaxis])
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        directional = fitness_perturb * to_best_dir
    else:
        directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

    random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
    new_population = new_population + directional + random_perturb

    self.population = self._clip_to_bounds(new_population)
```