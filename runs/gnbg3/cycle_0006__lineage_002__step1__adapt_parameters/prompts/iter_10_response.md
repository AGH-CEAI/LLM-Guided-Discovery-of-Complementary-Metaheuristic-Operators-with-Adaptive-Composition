**Idea: Covariance Condition Number and Effective Rank for Adaptive Scaling**

Adapt F and CR based on the condition number and effective rank of the population covariance matrix. High condition number (elongated distribution) → reduce F to intensify exploitation along principal axes; low condition number (isotropic) → increase F for exploration. Effective rank informs CR adaptation.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using spectral properties of population covariance.
    
    Uses condition number (ratio of max/min eigenvalues) to detect anisotropy,
    and effective rank (entropy of normalized eigenvalues) to measure effective
    dimensionality. These spectral signals guide parameter adaptation.
    """
    population = self._get_current_population()
    if population is None or len(population) < 3:
        return
    
    NP, dim = population.shape
    
    # Compute covariance matrix and its eigendecomposition
    centered = population - population.mean(axis=0)
    cov = centered.T @ centered / max(NP - 1, 1)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return
    
    # Filter positive eigenvalues (numerical stability)
    pos_eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(pos_eigenvalues) < 2:
        return
    
    # Condition number: ratio of max to min eigenvalue
    # High condition number -> elongated/ill-conditioned distribution
    cond_number = pos_eigenvalues.max() / pos_eigenvalues.min()
    log_cond = np.log1p(np.clip(cond_number, 0, 1e6))
    
    # Effective rank via entropy of normalized eigenvalues
    normalized_eig = pos_eigenvalues / pos_eigenvalues.sum()
    normalized_eig = normalized_eig[normalized_eig > 1e-12]
    entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-12))
    max_entropy = np.log(len(normalized_eig) + 1e-12)
    effective_rank = np.exp(entropy) / len(pos_eigenvalues) if max_entropy > 0 else 1.0
    
    # Spectral signal: combine condition number and effective rank
    # High cond_number -> reduce F (exploit along principal axes)
    # Low effective_rank -> increase F (explore collapsed dimensions)
    spectral_F_signal = np.clip(log_cond / 10.0 - (1 - effective_rank), -1.0, 1.0)
    spectral_CR_signal = np.clip((1 - log_cond / 20.0) * effective_rank, -1.0, 1.0)
    
    # Compute adaptation factor from improvement rate
    improvement_factor = np.clip(improvement_rate - 0.2, -0.5, 0.5) * 2.0
    
    # Adapt F: use spectral signal to modulate base F
    target_F = 0.6 - 0.2 * spectral_F_signal + 0.1 * improvement_factor
    self.F = np.clip(target_F, 0.3, 0.95)
    
    # Adapt CR: use spectral signal and condition number
    # High condition number -> higher CR (trust population structure)
    # Low effective rank -> higher CR (converged, need diversity)
    target_CR = 0.85 + 0.1 * spectral_CR_signal - 0.05 * improvement_factor
    self.CR = np.clip(target_CR, 0.5, 0.98)
    
    # Update history for temporal smoothing
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
    if len(self.F_history) > 50:
        self.F_history.pop(0)
        self.CR_history.pop(0)
    
    # Apply EMA smoothing to reduce parameter oscillation
    if len(self.F_history) >= 2:
        ema_alpha = 0.7
        self.F = ema_alpha * self.F + (1 - ema_alpha) * self.F_history[-2]
        self.CR = ema_alpha * self.CR + (1 - ema_alpha) * self.CR_history[-2]
        self.F = np.clip(self.F, 0.3, 0.95)
        self.CR = np.clip(self.CR, 0.5, 0.98)
```