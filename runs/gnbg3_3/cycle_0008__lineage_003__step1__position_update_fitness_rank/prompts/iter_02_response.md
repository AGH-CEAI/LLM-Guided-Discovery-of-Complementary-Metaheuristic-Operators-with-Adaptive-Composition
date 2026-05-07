**Idea: Rayleigh Quotient Velocity Modulation**
Category B: Uses eigenvalue decomposition of population covariance to compute velocity alignment via Rayleigh quotients, detecting anisotropy and modulating exploration in eigenspace.
```python
def _position_update_fitness_rank(self):
    """Rayleigh-quotient velocity modulation using covariance eigendecomposition (Category B).

    Key insight: Compute Rayleigh quotient R = v^T Cov v / v^T v to measure velocity
    alignment with population covariance principal axes. High Rayleigh quotient on
    collapsed eigendirections → population stuck in subspace → apply inverse scaling
    to velocity along dominant eigencomponents. Targets worst tasks (17, 16, 6) where
    large condition numbers indicate severe anisotropy and subspace collapse.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var

        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim

        rq_values = np.zeros(self.np)
        for i in range(self.np):
            v = self.velocity[i]
            v_cov_v = np.dot(v, cov @ v)
            v_v = np.dot(v, v) + 1e-10
            rq_values[i] = v_cov_v / v_v
        rq_mean = np.mean(rq_values)
        rq_std = np.std(rq_values) + 1e-10

        vel_proj = self.velocity @ eigenvectors
        vel_energy_per_comp = vel_proj ** 2
        total_vel_energy = np.sum(vel_energy_per_comp) + 1e-10
        vel_energy_fraction = vel_energy_per_comp / total_vel_energy

        alignment = vel_energy_fraction / (eigenvalues_norm + 1e-10)
        alignment = np.clip(alignment, 0.0, 10.0)

        spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5

        if cond > 100.0:
            log_damp = np.log1p(cond) / np.log1p(1000.0)
            log_damp = np.clip(log_damp, 0.0, 1.0)
            per_comp_scale = 1.0 + 0.5 * log_damp * (1.0 - alignment)
        else:
            per_comp_scale = 1.0 + 0.3 * spectral_signal * (1.0 - alignment)

        per_comp_scale = np.clip(per_comp_scale, 0.1, 2.5)
        vel_scaled = vel_proj * per_comp_scale

        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            directional = 0.4 * (1.0 - fitness_ranks[:, np.newaxis]) * to_best_dir
        else:
            directional = np.zeros((self.np, self.dim))

        random_perturb = 0.3 * np.random.uniform(-1, 1, (self.np, self.dim))

        new_population = self.population + (vel_scaled @ eigenvectors.T) + directional + random_perturb

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```