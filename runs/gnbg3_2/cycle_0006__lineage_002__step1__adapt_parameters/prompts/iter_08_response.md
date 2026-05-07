**Idea: Temporal–Spectral Hybrid Adaptation**

Combine temporal EMA regime detection (success-rate dynamics) with spectral condition-number analysis (population covariance structure). The condition number of the population covariance provides a fundamentally different signal about search state—detecting when the population has collapsed into a subspace (high condition) versus when it's isotropic (low condition)—which is orthogonal to the success-rate signals. Weight the two mechanisms by a principled function of the condition number.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using temporal EMA dynamics combined with spectral condition-number analysis."""
    success_rate = np.mean(improved_mask)

    # Initialize state on first call
    if not hasattr(self, 'success_ewma_short'):
        self.success_ewma_short = success_rate
        self.success_ewma_long = success_rate
        self.F_history = [self.F]
        self.Cr_history = [Cr_used]

    # === MECHANISM 1: Temporal EMA dynamics ===
    alpha_short = 0.3
    alpha_long = 0.1
    self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
    self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

    # Temporal regime detection
    short_trending_up = self.success_ewma_short > self.success_ewma_long
    short_trending_down = self.success_ewma_short < self.success_ewma_long
    both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15

    if short_trending_up and self.success_ewma_short > 0.25:
        F_temp = 1.05
        Cr_temp = 1.08
    elif short_trending_down and self.success_ewma_short < 0.2:
        F_temp = 0.88
        Cr_temp = 0.92
    elif both_stagnant:
        F_temp = 0.75
        Cr_temp = 1.15
    else:
        delta = self.success_ewma_short - self.success_ewma_long
        F_temp = 1.0 + 0.08 * delta
        Cr_temp = 1.0 - 0.05 * delta

    # === MECHANISM 2: Spectral condition-number analysis ===
    # Compute population covariance condition number
    centered = self.population_buffer - np.mean(self.population_buffer, axis=0)
    cov = np.cov(centered.T)
    
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        # Condition number: ratio of largest to smallest eigenvalue
        if eigvals[-1] > 1e-12:
            cond = eigvals[0] / eigvals[-1]
        else:
            cond = 1.0
    except np.linalg.LinAlgError:
        cond = 1.0

    # Log-scale condition number for smoother weighting
    log_cond = np.log1p(np.clip(cond, 0.0, 1e6))

    # Spectral regime: high condition number → anisotropic population → need exploration
    if log_cond > 4.0:        # Strong anisotropy
        F_spectral = 1.15
        Cr_spectral = 0.85
    elif log_cond > 2.0:      # Moderate anisotropy
        F_spectral = 1.05
        Cr_spectral = 0.92
    elif log_cond < 0.5:      # Nearly isotropic
        F_spectral = 0.90
        Cr_spectral = 1.05
    else:
        F_spectral = 1.0
        Cr_spectral = 1.0

    # === Principled weighting via condition number ===
    # High condition → trust spectral more (population structure is informative)
    # Low condition → trust temporal more (success rates are more reliable)
    spectral_weight = np.clip(log_cond / 6.0, 0.0, 0.7)

    # Combine both mechanisms with data-driven weighting
    F_combined = F_temp * (1 - spectral_weight) + F_spectral * spectral_weight
    Cr_combined = Cr_temp * (1 - spectral_weight) + Cr_spectral * spectral_weight

    # Update parameter history
    self.F_history.append(F_used)
    self.Cr_history.append(Cr_used)
    if len(self.F_history) > 15:
        self.F_history.pop(0)
        self.Cr_history.pop(0)

    # Apply with bounded drift correction
    F_mean = np.mean(self.F_history)
    drift_factor = np.clip(self.F / (F_mean + 1e-10), 0.7, 1.4)

    self.F = np.clip(self.F * F_combined * drift_factor, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_combined, 0.1, 0.9)
```