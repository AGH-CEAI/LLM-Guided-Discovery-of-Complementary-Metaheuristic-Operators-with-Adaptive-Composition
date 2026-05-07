**Idea: SVD-based Effective Rank and Condition Number Credit Assignment**

Uses singular value decomposition of the population matrix to compute effective rank (entropy of normalized singular values) and condition number. Strategies that generate successful mutations with higher effective rank or help manage condition number trends receive positive credit. This directly assesses how strategies affect population subspace structure and effective dimensionality.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment using spectral analysis of population structure.
    
    Category B: Spectral / linear-algebraic
    Uses SVD-based metrics (effective rank, condition number) to assess
    how each strategy affects population subspace structure and diversity.
    """
    import numpy as np
    
    if not hasattr(self, '_current_population') or self._current_population is None:
        return
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < 4 or dim < 2:
        return
    
    pop_centered = pop - pop.mean(axis=0)
    try:
        U, s, Vt = np.linalg.svd(pop_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return
    
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    eff_rank = np.exp(entropy) / min(NP, dim)
    cond_num = s[0] / (s[-1] + 1e-12)
    
    if not hasattr(self, '_spectral_eff_rank_history'):
        self._spectral_eff_rank_history = []
        self._spectral_cond_history = []
        self._strategy_spectral_shifts = np.zeros(len(self.strategy_names))
        self._strategy_cond_shifts = np.zeros(len(self.strategy_names))
        self._strategy_use_counts = np.zeros(len(self.strategy_names))
    
    self._spectral_eff_rank_history.append(eff_rank)
    self._spectral_cond_history.append(cond_num)
    
    alpha = 0.3
    
    for strat_idx in range(len(self.strategy_names)):
        mask = strategy_used == strat_idx
        n_used = np.sum(mask)
        if n_used == 0:
            continue
        
        self._strategy_use_counts[strat_idx] += n_used
        
        improved_mask = mask & improved
        not_improved_mask = mask & ~improved
        
        n_imp = np.sum(improved_mask)
        n_not_imp = np.sum(not_improved_mask)
        
        eff_rank_imp = eff_rank
        eff_rank_not_imp = eff_rank
        
        if n_imp > 0:
            imp_pop = pop[improved_mask]
            imp_centered = imp_pop - imp_pop.mean(axis=0)
            try:
                _, s_imp, _ = np.linalg.svd(imp_centered, full_matrices=False)
                s_imp_norm = s_imp / s_imp.sum()
                ent_imp = -np.sum(s_imp_norm * np.log(s_imp_norm + 1e-12))
                eff_rank_imp = np.exp(ent_imp) / (min(*imp_pop.shape) + 1e-12)
            except:
                pass
        
        if n_not_imp > 0:
            not_imp_pop = pop[not_improved_mask]
            not_imp_centered = not_imp_pop - not_imp_pop.mean(axis=0)
            try:
                _, s_not_imp, _ = np.linalg.svd(not_imp_centered, full_matrices=False)
                s_not_imp_norm = s_not_imp / s_not_imp.sum()
                ent_not_imp = -np.sum(s_not_imp_norm * np.log(s_not_imp_norm + 1e-12))
                eff_rank_not_imp = np.exp(ent_not_imp) / (min(*not_imp_pop.shape) + 1e-12)
            except:
                pass
        
        rank_shift = eff_rank_imp - eff_rank_not_imp
        weight = n_imp / (n_imp + n_not_imp + 1e-12)
        self._strategy_spectral_shifts[strat_idx] = (1 - alpha) * self._strategy_spectral_shifts[strat_idx] + alpha * rank_shift * weight
        
        self.strategy_scores[strat_idx] += 0.1 * np.tanh(5.0 * self._strategy_spectral_shifts[strat_idx])
    
    if len(self._spectral_cond_history) >= 3:
        cond_trend = self._spectral_cond_history[-1] - self._spectral_cond_history[0]
        
        for strat_idx in range(len(self.strategy_names)):
            mask = strategy_used == strat_idx
            n_used = np.sum(mask)
            if n_used == 0:
                continue
            
            strat_pop = pop[mask]
            strat_centered = strat_pop - strat_pop.mean(axis=0)
            try:
                _, s_strat, _ = np.linalg.svd(strat_centered, full_matrices=False)
                cond_strat = s_strat[0] / (s_strat[-1] + 1e-12)
            except:
                continue
            
            cond_shift = cond_strat - cond_num
            self._strategy_cond_shifts[strat_idx] = (1 - alpha) * self._strategy_cond_shifts[strat_idx] + alpha * cond_shift
            
            if cond_trend > 0.1:
                credit = -0.05 * np.sign(self._strategy_cond_shifts[strat_idx]) * min(abs(self._strategy_cond_shifts[strat_idx]), 1.0)
                self.strategy_scores[strat_idx] += credit
            elif cond_trend < -0.1:
                credit = 0.05 * np.sign(self._strategy_cond_shifts[strat_idx]) * min(abs(self._strategy_cond_shifts[strat_idx]), 1.0)
                self.strategy_scores[strat_idx] += credit
    
    self.strategy_scores = np.clip(self.strategy_scores, -50.0, 50.0)
    scores_exp = np.exp(self.strategy_scores - self.strategy_scores.max())
    probs = scores_exp / (scores_exp.sum() + 1e-12)
```