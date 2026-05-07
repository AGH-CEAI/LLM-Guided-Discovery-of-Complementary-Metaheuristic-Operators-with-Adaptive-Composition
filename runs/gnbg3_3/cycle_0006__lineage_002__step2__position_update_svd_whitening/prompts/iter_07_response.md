**Idea: Monte Carlo Random Subspace SVD Whitening**

One-line description: Uses multiple Monte Carlo random projections to compute SVD-based whitening across diverse subspaces, averaging the transformations for robust velocity modulation that avoids deterministic collapse on deceptive landscapes (Category G: Stochastic / sampling-based).

```python
def _position_update_svd_whitening(self):
    """Monte Carlo random subspace SVD for robust population whitening.
    
    Category G: Stochastic / sampling-based
    Uses multiple random orthogonal projections to compute SVD transformations
    in different subspaces, then averages the resulting velocity modulations.
    This stochastic approach provides robustness against population collapse
    and local optima traps on deceptive landscapes.
    """
    centered = self.population - np.mean(self.population, axis=0)
    n_mc_samples = 15  # Monte Carlo sampling count

    try:
        accumulated_velocity = np.zeros_like(self.velocity)

        for _ in range(n_mc_samples):
            # Generate random orthogonal projection matrix (Monte Carlo draw)
            R = np.random.randn(self.dim, self.dim)
            Q, _ = np.linalg.qr(R)

            # Project to random subspace
            proj_pop = centered @ Q
            proj_vel = self.velocity @ Q

            # SVD in projected subspace
            U, singular_values, Vt = np.linalg.svd(proj_pop, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)

            if singular_values[0] < 1e-10:
                continue

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            # Inverse-square-law modulation in projected space
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            # Scale velocity in projected space
            vel_scaled = proj_vel * per_component_scale

            # Transform back to original space and accumulate
            accumulated_velocity += vel_scaled @ Q.T

        # Average across Monte Carlo samples
        avg_velocity = accumulated_velocity / n_mc_samples
        new_population = self.population + avg_velocity

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```