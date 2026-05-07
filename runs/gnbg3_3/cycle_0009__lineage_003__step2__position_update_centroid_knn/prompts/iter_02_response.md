**Idea: Spectral Entropy + Condition-Number Anisotropic Damping**

One-line description: Use covariance eigenvalue distribution (spectral entropy, condition number) to detect convergence and modulate velocity anisotropically in the eigenspace, damping dominant directions when population is collapsing toward a subspace.

```python
def _position_update_centroid_knn(self):
    """Spectral entropy + condition-number anisotropic damping (Category B).
    
    Replaces geometric k-NN density with eigenvalue analysis. Key insight:
    spectral entropy detects when population collapses to a subspace (low entropy
    = concentrated along dominant eigenvectors = premature convergence). Condition
    number measures anisotropy. Damping along dominant eigendirections prevents
    collapse; exploration scaling handles ill-conditioned worst tasks (17, 16, 6).
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, np.argsort(np.linalg.eigvalsh(cov))[::-1]]

        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var

        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_log = np.log1p(cond)

        sv_norm = np.sqrt(eigenvalues) / (np.sqrt(eigenvalues[0]) + 1e-10)
        spectral_damping = sv_norm ** 0.5
        spectral_damping = spectral_damping / (np.max(spectral_damping) + 1e-10)

        entropy_scale = 0.5 + 0.5 * entropy_ratio
        cond_scale = np.clip(cond_log / 5.0, 0.0, 2.0)
        spectral_signal = entropy_scale * (1.0 + 0.3 * cond_scale)

        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * spectral_damping * spectral_signal
        new_population = self.population + vel_scaled @ eigenvectors.T

        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            correction = 0.3 * to_best_dir * (1.0 + 0.3 * cond_scale)
            new_population = new_population + correction

        exploration = 1.0 + 0.5 * np.log1p(cond)
        random_perturb = exploration * np.random.uniform(-0.3, 0.3, (self.np, self.dim))
        new_population = new_population + random_perturb

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```