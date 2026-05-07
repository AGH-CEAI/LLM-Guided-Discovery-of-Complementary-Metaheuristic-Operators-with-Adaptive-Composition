**Idea: Eigenvalue-Weighted Spectral Inertia**
Adapt inertia weight using the condition number (λ_max/λ_min) and effective dimensionality from eigendecomposition of the population covariance — low inertia when the population is highly anisotropic (high condition number) or confined to a low-dimensional subspace, encouraging escape from collapsed manifolds.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using spectral properties of population covariance.
    
    Category B (Spectral / linear-algebraic):
    - Uses eigenvalues of population covariance matrix
    - Condition number (λ_max / λ_min) measures anisotropy
    - Effective dimensionality from cumulative eigenvalue variance
    - Low inertia when population is highly anisotropic or low-dimensional
      (exploration needed to escape collapsed/subspace-trapped regions)
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-14, None)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending

        # Condition number: anisotropy indicator
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_normalized = np.clip(cond / 500.0, 0.0, 1.0)

        # Effective dimensionality: fraction of dimensions needed for 95% variance
        total_var = np.sum(eigenvalues) + 1e-10
        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = np.clip(eff_dim / self.dim, 0.0, 1.0)

        # Spectral signal: low when anisotropic OR low-dimensional
        # Both regimes need exploration → lower inertia
        spectral_explore_signal = cond_normalized * 0.6 + (1.0 - eff_dim_ratio) * 0.4

        # Base inertia: higher when population is well-spread (exploration done)
        base_inertia = 0.4 + 0.35 * eff_dim_ratio

        # Modulation: reduce inertia for anisotropic / low-dimensional cases
        target_inertia = base_inertia * (1.0 - 0.45 * spectral_explore_signal)

        # Generation decay: gentle schedule, not dominant
        decay = 0.729 - 0.08 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

        # EMA smoothing
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

    except np.linalg.LinAlgError:
        # Fallback: moderate fixed inertia
        fallback = 0.729 - 0.08 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + fallback * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```