**Idea: Condition-Number-Driven Spectral Inertia**

Adapt inertia weight using the condition number of the population covariance matrix. High condition number → population is collapsed along some axes (ill-conditioned landscape) → increase inertia to resist premature convergence along dominant directions. This is spectral (uses eigenvalues) and takes a different angle from the SVD-whitening position update in variant_02.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using condition number of population covariance."""
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 10000.0)

        # Spectral signal: map condition number to inertia adjustment
        # High cond = anisotropic/ill-conditioned → more exploration (higher inertia)
        # Low cond = well-conditioned → more exploitation (lower inertia)
        spectral_signal = np.log1p(cond) / np.log1p(10000.0)
        spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

        # Map spectral signal to inertia range [0.4, 0.95]
        target_inertia = 0.4 + 0.55 * spectral_signal

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