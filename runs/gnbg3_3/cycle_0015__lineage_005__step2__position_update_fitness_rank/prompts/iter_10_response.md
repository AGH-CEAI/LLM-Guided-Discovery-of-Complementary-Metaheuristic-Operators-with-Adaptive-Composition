**Idea: Covariance-Eigenvalue Effective-Rank Velocity Modulation**

This variant takes a different spectral angle from the prior SVD-whitening approach (variant_02). Instead of inverse-square singular-value modulation, it computes the **effective rank** (entropy-based dimensionality measure) of the population covariance matrix and uses it to dynamically modulate velocity per eigendirection. High effective rank → population spans many modes → increase exploration; low effective rank → population collapsed → amplify in collapsed directions to escape. This directly targets ill-conditioned, high-dimensional landscapes like Tasks 17, 16, 6, 5.

```python
def _position_update_fitness_rank(self):
    """Covariance-eigenvalue effective-rank velocity modulation (Category B).

    Key insight: Use effective rank (entropy of normalized covariance eigenvalues)
    as a diversity signal, combined with per-eigendirection velocity scaling.
    High effective rank = diverse population → boost exploration.
    Low effective rank = collapsed population → amplify velocity in weak directions.
    Condition number modulates blend between uniform and anisotropy-targeted scaling.
    This is orthogonal to SVD-whitening (variant_02) which used inverse-square singular
    value modulation on position, not eigenvalue-ratio velocity scaling.
    """
    centered = self.population - np.mean(self.population, axis=0)

    try:
        # Covariance eigenvalue decomposition
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Effective rank: entropy-based dimensionality measure
        total_eig = np.sum(eigenvalues)
        eigenvalues_norm = eigenvalues / (total_eig + 1e-10)
        eigenvalues_norm = np.clip(eigenvalues_norm, 1e-15, None)
        entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-15))
        max_entropy = np.log(len(eigenvalues_norm))
        effective_rank = np.exp(entropy) / (len(eigenvalues_norm) + 1e-10)
        effective_rank = np.clip(effective_rank, 0.0, 1.0)

        # Condition number for anisotropy detection
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 10000.0)
        cond_signal = np.log1p(cond) / np.log1p(10000.0)

        # Per-eigendirection velocity scaling
        eig_ratio = eigenvalues / (eigenvalues[0] + 1e-10)
        eig_ratio = np.clip(eig_ratio, 0.0, 1.0)

        # Anisotropy-targeted scaling: penalize dominant, amplify weak
        # Low eig_ratio = weak direction → amplify
        # High eig_ratio = dominant direction → dampen
        spectral_scale = np.sqrt(eig_ratio + 0.01)
        spectral_scale = spectral_scale / (np.max(spectral_scale) + 1e-10)

        # Blend uniform and anisotropy-targeted based on condition number
        uniform_scale = 1.0
        blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
        per_component_scale = uniform_scale * (1.0 - blend) + spectral_scale * blend

        # Project velocity to eigenspace, scale, project back
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * per_component_scale
        velocity_correction = vel_scaled @ eigenvectors.T - self.velocity

        # Exploration modulation from effective rank
        # High effective_rank → population diverse → more exploration
        # Low effective_rank → population collapsed → more exploration to escape
        if effective_rank < 0.3:
            explore_boost = 1.5
        elif effective_rank > 0.7:
            explore_boost = 1.2
        else:
            explore_boost = 1.0

        # Add scaled velocity correction + random perturbation
        random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
        random_perturb *= explore_boost * (1.0 + 0.5 * cond_signal)

        new_population = self.population + self.velocity + 0.5 * velocity_correction + random_perturb

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```