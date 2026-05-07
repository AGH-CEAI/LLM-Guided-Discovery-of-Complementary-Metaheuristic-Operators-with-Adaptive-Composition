**Idea: Spectral-Condition + Temporal-Drift Hybrid**

Combine spectral condition number (geometric anisotropy detection) with centroid-velocity temporal drift (convergence/oscillation detection). Switch between high-inertia stabilization (oscillating) and low-inertia acceleration (converging) based on data-driven centroid velocity threshold.

```python
def _adapt_inertia_weight(self):
    """Hybrid: spectral condition (geometric) + centroid velocity (temporal)."""
    # --- Mechanism 1: Geometric (spectral condition number) ---
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)
        cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
        spectral_cond_normalized = np.clip(cond / 100.0, 0.0, 1.0)
    except np.linalg.LinAlgError:
        spectral_cond_normalized = 0.5
    
    # --- Mechanism 2: Temporal (centroid velocity drift) ---
    centroid = np.mean(self.population, axis=0)
    if not hasattr(self, '_ema_centroid_velocity'):
        self._ema_centroid_velocity = 0.0
    self._ema_centroid_velocity = (0.95 * self._ema_centroid_velocity +
                                    0.05 * np.linalg.norm(centroid))
    temporal_drift = self._ema_centroid_velocity
    
    # --- Principled switching based on convergence state ---
    if temporal_drift > 0.3:
        # Oscillating/unsteady → stabilize with higher inertia
        target_inertia = 0.4 + 0.3 * spectral_cond_normalized
    else:
        # Converging → accelerate with lower inertia
        target_inertia = 0.9 - 0.3 * spectral_cond_normalized
    
    # EMA smoothing + bounds
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```