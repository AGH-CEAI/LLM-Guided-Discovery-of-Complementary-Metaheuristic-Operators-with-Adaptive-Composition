**Idea: Spectral Entropy & Effective Dimensionality**
Adapt inertia using eigenvalue entropy and effective dimensionality instead of condition number. This captures how uniformly variance spreads across eigendirections and how many dimensions contribute meaningfully — orthogonal information to the current condition-number approach.
```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using spectral entropy and effective dimensionality."""
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Normalize eigenvalues to get probability distribution
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var

        # Spectral entropy: measures uniformity of variance distribution
        # High entropy = well-spread population = more exploitation (lower inertia)
        # Low entropy = concentrated population = more exploration (higher inertia)
        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(len(eigenvalues) + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

        # Effective dimensionality: fraction of dimensions needed for 95% variance
        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = np.clip(eff_dim / len(eigenvalues), 0.0, 1.0)

        # Spectral gap: ratio of largest to second-largest eigenvalue
        # Large gap = population aligned along one direction = higher inertia
        if len(eigenvalues) > 1 and eigenvalues[1] > 1e-10:
            spectral_gap = eigenvalues[0] / eigenvalues[1]
            spectral_gap = np.clip(np.log1p(spectral_gap) / np.log1p(1000.0), 0.0, 1.0)
        else:
            spectral_gap = 1.0

        # Combine signals: entropy drives base inertia, eff_dim and gap refine it
        # Low entropy + low eff_dim + high gap → high inertia for exploration
        # High entropy + high eff_dim + low gap → low inertia for exploitation
        base_inertia = 0.4 + 0.55 * entropy_ratio
        eff_dim_boost = 0.3 * (1.0 - eff_dim_ratio)
        gap_boost = 0.2 * spectral_gap
        target_inertia = base_inertia + eff_dim_boost + gap_boost

        # Blend with time-based baseline
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    except np.linalg.LinAlgError:
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```