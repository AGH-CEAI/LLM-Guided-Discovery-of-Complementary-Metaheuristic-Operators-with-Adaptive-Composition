**Idea: Eigenvalue-Anisotropic Velocity Scaling with Condition-Number Modulation**

Category B: Spectral / linear-algebraic approach using eigenvalue decomposition of population covariance to detect anisotropy and ill-conditioning. Projects velocity onto principal axes, scales inversely by eigenvalue magnitude, and modulates overall exploration intensity via condition number.

```python
def _position_update_fitness_rank(self):
    """Eigenvalue-anisotropic velocity scaling with condition-number modulation (Category B).

    Key insight: Use eigenvalue decomposition of population covariance (not SVD on centered
    data) to detect anisotropy and ill-conditioning. Scale velocity inversely along principal
    axes proportional to eigenvalue magnitude. Condition number modulates global exploration
    intensity. This is orthogonal to graph-based (E) and fitness-rank (D) approaches.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 1e6)

        # Effective dimensionality from eigenvalue spectrum
        total_var = np.sum(eigenvalues) + 1e-10
        sorted_desc = np.sort(eigenvalues)[::-1]
        cumvar = np.cumsum(sorted_desc) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1

        # Spectral signal from condition number: high cond = anisotropic landscape
        spectral_signal = np.clip(np.log1p(cond) / np.log1p(1e6), 0.0, 1.0)

        # Adaptive inertia from spectral signal
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = 0.4 + 0.55 * spectral_signal
        self.inertia_weight = np.clip(
            self.inertia_weight * 0.8 + (target_inertia * 0.6 + decay * 0.4) * 0.2,
            0.4, 0.95
        )

        # Fitness rank signal
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Spectral exploration factor
        # High condition number = clustered/ill-conditioned → more exploration
        spectral_explore = 1.0 + 1.0 * spectral_signal
        spectral_explore = np.clip(spectral_explore, 0.5, 2.5)

        # Velocity projection onto principal axes
        vel_proj = self.velocity @ eigenvectors

        # Inverse eigenvalue scaling: dampen high-variance directions, preserve low-variance
        sv_norm = eigenvalues / (eigenvalues[0] + 1e-10)
        # Use squared inverse for stronger effect on collapsed directions
        per_component_scale = 1.0 / (sv_norm ** 2 + 0.01)
        per_component_scale = per_component_scale / (np.max(per_component_scale) + 1e-10)

        # Blend uniform and per-component scaling based on condition number
        blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
        uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
        combined_scale = uniform_scale * (1.0 - blend_weight) + per_component_scale * blend_weight
        combined_scale = np.clip(combined_scale, 0.3, 2.5)

        vel_scaled = vel_proj * combined_scale * spectral_explore

        # Fitness-based directional perturbation
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = np.zeros((self.np, self.dim))

        # Random perturbation scaled by spectral signal
        random_perturb = spectral_explore * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + self.inertia_weight * vel_scaled + directional + random_perturb

    except (np.linalg.LinAlgError, FloatingPointError):
        spectral_explore = 1.0 + 0.5 * np.random.random()
        new_population = self.population + self.inertia_weight * self.velocity * spectral_explore

    self.population = self._clip_to_bounds(new_population)
```