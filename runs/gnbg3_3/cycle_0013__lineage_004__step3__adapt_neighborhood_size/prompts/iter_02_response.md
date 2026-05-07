**Idea: Condition-Number-Driven Neighborhood Expansion**

Adapt neighborhood size using the spectral condition number of the population covariance. High condition number (particles collapsed along dominant eigenvectors) → expand neighborhood to accelerate inter-subpopulation information flow and prevent fragmentation. Low condition number (isotropic spread) → shrink neighborhood for faster ring convergence.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology neighborhood size based on spectral condition number.
    
    Category B (Spectral / linear-algebraic):
    - Computes eigenvalues of population covariance matrix
    - Derives condition number as anisotropy measure
    - High condition number (collapsed along eigenvectors) → expand neighborhood
      to accelerate cross-subpopulation communication and prevent fragmentation
    - Low condition number (isotropic) → contract neighborhood for faster convergence
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]

        cond = eigenvalues[0] / eigenvalues[-1]
        cond = np.clip(cond, 1.0, 10000.0)

        # Map condition number to spectral signal in [0, 1]
        spectral_signal = np.log1p(cond) / np.log1p(10000.0)
        spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

        # High anisotropy (high cond) → expand neighborhood for faster info spread
        # Low anisotropy (low cond) → contract neighborhood for faster convergence
        expansion_bias = 2.0 * spectral_signal - 1.0  # in [-1, 1]
        expansion_bias = np.clip(expansion_bias, -0.5, 0.5)

        min_size = max(1, self.np // 10)
        max_size = max(3, self.np // 3)

        if expansion_bias > 0:
            self.neighborhood_size = min(max_size, self.neighborhood_size + 1)
        elif expansion_bias < 0:
            self.neighborhood_size = max(min_size, self.neighborhood_size - 1)
    except np.linalg.LinAlgError:
        pass
```