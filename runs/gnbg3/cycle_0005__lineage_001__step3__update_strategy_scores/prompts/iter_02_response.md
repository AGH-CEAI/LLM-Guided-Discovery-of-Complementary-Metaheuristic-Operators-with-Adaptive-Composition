**Idea: Covariance Conditioning Monitor**
Use eigenvalue decomposition of the population covariance matrix to compute condition number and effective dimensionality, then credit strategies whose mutations improved these spectral properties.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Spectral credit: reward strategies that improve population conditioning."""
    if not hasattr(self, '_prev_cond') or self._prev_cond is None:
        self._prev_cond = None
        self._prev_eff_dim = None
        self._spec_scores = np.zeros(len(self.strategy_names))
        self._spec_momentum = 0.7
    
    pop = self._current_population
    if pop is None or len(pop) < 4:
        return
    
    # Compute current population covariance and its spectral properties
    cov = np.cov(pop, rowvar=False)
    if cov.size == 0:
        return
    
    # Ensure positive semi-definiteness
    cov = (cov + cov.T) / 2 + np.eye(self.dim) * 1e-8
    
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        eigvals = np.sort(eigvals)[::-1]
        
        # Condition number: ratio of max to min eigenvalue
        cond = eigvals[0] / eigvals[-1]
        log_cond = np.log1p(cond)
        
        # Effective dimensionality: count eigenvalues above threshold
        threshold = eigvals[0] * 1e-6
        eff_dim = np.sum(eigvals > threshold)
        
        # Compute spectral signals (positive = improvement)
        if self._prev_cond is not None:
            # Lower condition number = better conditioned = diversity signal
            cond_signal = self._prev_cond - cond
            # Higher effective dim = more diverse = exploration signal
            dim_signal = eff_dim - self._prev_eff_dim
        else:
            cond_signal = 0.0
            dim_signal = 0.0
        
        self._prev_cond = cond
        self._prev_eff_dim = eff_dim
        
        # Normalize signals to [-1, 1] range
        cond_signal = np.tanh(cond_signal / (log_cond + 1.0))
        dim_signal = np.tanh(dim_signal / max(eff_dim, 1.0))
        
        # Rank-based signal: proportion of improvements
        rank_signal = (np.mean(improved) - 0.2) * 2.0
        rank_signal = np.clip(rank_signal, -1.0, 1.0)
        
        # Combined spectral score for this generation
        combined = 0.5 * cond_signal + 0.3 * dim_signal + 0.2 * rank_signal
        combined = np.clip(combined, -1.0, 1.0)
        
        # Credit each strategy based on how many individuals improved
        for s in range(len(self.strategy_names)):
            mask = strategy_used == s
            n_used = np.sum(mask)
            if n_used > 0:
                success_rate = np.mean(improved[mask])
                # Weight by usage frequency to avoid over-crediting rare strategies
                usage_weight = np.clip(n_used / self.NP, 0.1, 1.0)
                delta = combined * usage_weight * (success_rate * 2.0 - 0.5)
                self._spec_scores[s] = self._spec_momentum * self._spec_scores[s] + (1 - self._spec_momentum) * delta
        
        # Blend with existing strategy_scores
        alpha = 0.3
        self.strategy_scores = (1 - alpha) * self.strategy_scores + alpha * self._spec_scores
        
    except np.linalg.LinAlgError:
        return
```