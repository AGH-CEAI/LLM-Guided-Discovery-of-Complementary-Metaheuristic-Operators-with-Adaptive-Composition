**Idea: Spectral-Improvement Hybrid Inertia**

Combines spectral condition-number signal (mechanism 1) with fitness-improvement rate signal (mechanism 2) using data-driven stagnation detection to weight them. When stagnating, spectral dominates (exploration); when improving, improvement signal dominates (exploitation).

```python
def _adapt_inertia_weight(self):
    """Hybrid inertia adaptation: spectral condition + fitness improvement signals.
    
    Combines TWO distinct mechanisms:
      1. Spectral: condition number of population covariance (Category B)
      2. Fitness-improvement: temporal improvement rate (Category D/F)
    
    Principled switching: stagnation-based weighting.
      - Stagnating → trust spectral signal (exploration needed for hard tasks)
      - Improving → trust improvement signal (exploitation of good direction)
    """
    try:
        # === MECHANISM 1: SPECTRAL SIGNAL (condition number) ===
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 10000.0)

        # High cond = anisotropic/ill-conditioned → more exploration (higher inertia)
        spectral_signal = np.log1p(cond) / np.log1p(10000.0)
        spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

        # Map spectral signal to inertia range [0.4, 0.95]
        spectral_inertia = 0.4 + 0.55 * spectral_signal

        # === MECHANISM 2: FITNESS-IMPROVEMENT SIGNAL (temporal) ===
        if self.global_best is not None:
            if not hasattr(self, '_prev_best_fitness'):
                self._prev_best_fitness = self.global_best_fitness
            improvement = self._prev_best_fitness - self.global_best_fitness
            self._prev_best_fitness = self.global_best_fitness
        else:
            improvement = 0.0

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * max(0.0, improvement) + 0.7 * self._ema_improvement

        # Normalize improvement signal: high improvement → low inertia (exploit)
        # Use rank-based normalization relative to stagnation counter
        max_expected_improvement = 1.0 + self.stagnation_counter * 0.1
        improvement_signal = 1.0 - np.clip(self._ema_improvement / (max_expected_improvement + 1e-10), 0.0, 1.0)

        # Map improvement signal to inertia range [0.4, 0.95]
        improvement_inertia = 0.4 + 0.55 * improvement_signal

        # === PRINCIPLED SWITCHING: stagnation-based weighting ===
        # Data-driven: use stagnation counter to decide mechanism weights
        stagnation_threshold = 15
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            # Stagnating: trust spectral signal more (exploration needed)
            # Especially critical for hard tasks (17, 16, 6, 11) with large errors
            spectral_weight = 0.7
            improvement_weight = 0.3
        else:
            # Improving: trust improvement signal more (can exploit)
            spectral_weight = 0.3
            improvement_weight = 0.7

        target_inertia = (
            spectral_weight * spectral_inertia +
            improvement_weight * improvement_inertia
        )
        target_inertia = np.clip(target_inertia, 0.4, 0.95)

        # Blend with time-based baseline for smooth transition
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

        # EMA update for smooth inertia transitions
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)

    except np.linalg.LinAlgError:
        # Fallback to time-based decay
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```