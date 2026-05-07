**Idea: Covariance Condition Number Delta with Momentum**

Use the condition number of population covariance as a spectral health metric. Credit strategies based on how they change the condition number—rewarding strategies that improve population conditioning (lower condition number = more diverse/spread population) and penalizing those that degrade it. Track per-strategy condition number deltas across successful vs unsuccessful trials, apply momentum-smoothed exponential updates. (Category B: Spectral/linear-algebraic, targeting task 10)

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment using covariance condition number changes (spectral analysis)."""
    if not hasattr(self, '_cond_history'):
        self._cond_history = []
    if not hasattr(self, '_strat_momentum'):
        self._strat_momentum = np.zeros(len(self.strategy_names))
    
    pop = self._current_population
    if pop is None or len(pop) < 4 or self.NP < 3:
        return
    
    NP, dim = pop.shape
    
    # Compute condition number of population covariance (spectral health metric)
    # Lower condition number = more isotropic spread = better diversity
    try:
        centered = pop - pop.mean(axis=0)
        # Use SVD on centered population matrix
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s = s[s > 1e-12 * s[0]]  # Filter near-zero singular values
        if len(s) < 2:
            cond = 1.0
        else:
            cond = s[0] / s[-1] if s[-1] > 1e-12 else 1e6
    except Exception:
        cond = 1e3
    
    cond = float(np.clip(cond, 1.0, 1e6))
    self._cond_history.append(cond)
    if len(self._cond_history) > 10:
        self._cond_history.pop(0)
    
    # Compute condition numbers for successful vs unsuccessful subsets
    imp = np.asarray(improved, dtype=bool)
    n_imp = np.sum(imp)
    n_nimp = NP - n_imp
    
    cond_imp = 1.0
    cond_nimp = 1.0
    
    if n_imp >= 2:
        try:
            pop_imp = pop[imp]
            centered_imp = pop_imp - pop_imp.mean(axis=0)
            _, s_imp, _ = np.linalg.svd(centered_imp, full_matrices=False)
            s_imp = s_imp[s_imp > 1e-12 * s_imp[0]]
            cond_imp = float(np.clip(s_imp[0] / s_imp[-1] if len(s_imp) >= 2 and s_imp[-1] > 1e-12 else 1e6, 1.0, 1e6))
        except Exception:
            cond_imp = 1e3
    
    if n_nimp >= 2:
        try:
            pop_nimp = pop[~imp]
            centered_nimp = pop_nimp - pop_nimp.mean(axis=0)
            _, s_nimp, _ = np.linalg.svd(centered_nimp, full_matrices=False)
            s_nimp = s_nimp[s_nimp > 1e-12 * s_nimp[0]]
            cond_nimp = float(np.clip(s_nimp[0] / s_nimp[-1] if len(s_nimp) >= 2 and s_nimp[-1] > 1e-12 else 1e6, 1.0, 1e6))
        except Exception:
            cond_nimp = 1e3
    
    # Log-scale condition number delta
    eps = 1e-6
    delta_cond = np.log(cond_nimp + eps) - np.log(cond_imp + eps)
    
    # Per-strategy: compute weighted condition delta
    # Weight = distance from centroid (diversity contribution)
    centroid = pop.mean(axis=0)
    distances = np.linalg.norm(pop - centroid, axis=1)
    weights = distances / (distances.sum() + eps)
    
    for s_idx in range(len(self.strategy_names)):
        mask = strategy_used == s_idx
        n_s = np.sum(mask)
        if n_s == 0:
            continue
        
        w_s = weights[mask].sum()
        
        # Condition number change for this strategy's successful trials
        strat_imp = imp[mask]
        n_s_imp = np.sum(strat_imp)
        
        if n_s_imp >= 2:
            try:
                pop_si = pop[mask][strat_imp]
                c_si = pop_si - pop_si.mean(axis=0)
                _, ss, _ = np.linalg.svd(c_si, full_matrices=False)
                ss = ss[ss > 1e-12 * ss[0]]
                cond_si = float(np.clip(ss[0] / ss[-1] if len(ss) >= 2 and ss[-1] > 1e-12 else 1e6, 1.0, 1e6))
            except Exception:
                cond_si = 1e3
        else:
            cond_si = cond_imp
        
        cond_delta_s = np.log(cond_nimp + eps) - np.log(cond_si + eps)
        
        # Normalize by strategy usage proportion
        delta_norm = cond_delta_s * (n_s / NP) * w_s
        delta_cond_s = delta_norm + 0.5 * delta_cond * (n_s_imp / max(n_s, 1))
        
        # Momentum-smoothed update
        momentum = 0.7
        self._strat_momentum[s_idx] = momentum * self._strat_momentum[s_idx] + (1 - momentum) * delta_cond_s
        self.strategy_scores[s_idx] += 0.15 * self._strat_momentum[s_idx]
    
    # Prevent score divergence
    self.strategy_scores = np.clip(self.strategy_scores, -50.0, 50.0)
```