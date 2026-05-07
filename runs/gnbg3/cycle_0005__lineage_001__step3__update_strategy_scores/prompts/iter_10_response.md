**Idea: Eigenvalue-Weighted Strategy Selection**
Use eigendecomposition of population covariance to compute condition number and effective rank, then weight strategy scores by spectral quality (well-conditioned populations favor exploitation, ill-conditioned favor exploration).

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Category B: Spectral analysis via eigendecomposition of population covariance."""
    # Initialize EMA tracking
    if not hasattr(self, '_spectral_success_ema'):
        self._spectral_success_ema = np.ones(len(self.strategy_names)) * 0.5
        self._ema_alpha = 0.3
    
    # Compute spectral metrics from population covariance
    if hasattr(self, '_current_population') and self._current_population is not None:
        pop = self._current_population
        if len(pop) >= self.dim + 1:
            # Center and compute covariance
            centered = pop - pop.mean(axis=0)
            cov = np.cov(centered.T)
            
            # Eigendecomposition: extract eigenvalues
            try:
                eigenvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
                eigenvals = eigenvals[eigenvals > 1e-12]
            except Exception:
                eigenvals = np.array([1.0])
            
            if len(eigenvals) >= 2:
                # Condition number: ratio of largest to smallest eigenvalue
                cond = eigenvals[0] / max(eigenvals[-1], 1e-12)
                cond = min(cond, 1e6)
                
                # Effective rank via eigenvalue entropy
                total = np.sum(eigenvals)
                probs = eigenvals / max(total, 1e-12)
                entropy = -np.sum(probs * np.log(probs + 1e-12))
                max_entropy = np.log(len(eigenvals))
                eff_rank = np.exp(entropy) / max(max_entropy, 1e-12)
            else:
                cond = 1e6
                eff_rank = 0.1
            
            # Spectral quality: 1.0 = well-conditioned (favor exploitation), 0.0 = ill-conditioned (favor exploration)
            spectral_quality = np.exp(-np.log(1 + cond) / 10.0) * np.clip(eff_rank, 0.1, 1.0)
        else:
            spectral_quality = 0.5
    else:
        spectral_quality = 0.5
    
    # Update strategy success rates via EMA
    for strat_idx in range(len(self.strategy_names)):
        mask = strategy_used == strat_idx
        if np.any(mask):
            success_rate = np.mean(improved[mask])
            self._spectral_success_ema[strat_idx] = (1 - self._ema_alpha) * self._spectral_success_ema[strat_idx] + self._ema_alpha * success_rate
    
    # Compute target scores: blend spectral quality with success rate
    # High spectral_quality -> reward strategies with high success (exploitation)
    # Low spectral_quality -> favor balanced exploration
    baseline = 0.3 + 0.4 * spectral_quality
    target_scores = baseline + 0.3 * self._spectral_success_ema
    
    # Smooth update: prevent drastic changes
    momentum = 0.4
    self.strategy_scores = (1 - momentum) * self.strategy_scores + momentum * target_scores
    
    # Ensure minimum score to avoid starvation
    self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)
```