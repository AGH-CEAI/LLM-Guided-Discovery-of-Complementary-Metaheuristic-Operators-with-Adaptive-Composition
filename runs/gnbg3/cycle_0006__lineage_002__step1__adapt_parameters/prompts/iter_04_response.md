**Idea: Rank-Order Stability Adaptation with Spearman Correlation**
Adapt F/CR using rank-order stability (Spearman correlation between consecutive generations), rank-change magnitude, and fitness percentile spread — all purely fitness-based signals.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using fitness rank-based signals only."""
    # Initialize temporal rank tracking
    if not hasattr(self, '_prev_fitness_ranks'):
        self._prev_fitness_ranks = self._compute_fitness_ranking(self._get_current_fitness())
        self._rank_change_ema = 0.0
        self._spread_ema = 0.0
        self._rank_corr_ema = 1.0
        self._F_adapt_momentum = 0.7
        self._CR_adapt_momentum = 0.7
        return

    current_ranks = self._compute_fitness_ranking(self._get_current_fitness())
    
    # Signal 1: Rank change magnitude (mean absolute rank displacement)
    rank_diff = np.abs(current_ranks - self._prev_fitness_ranks)
    rank_change = rank_diff.mean()
    
    # Signal 2: Spearman correlation between consecutive rank orderings
    # High correlation = stable ranking = need more exploration (higher F)
    if hasattr(self, '_prev_fitness_ranks') and len(self._prev_fitness_ranks) > 2:
        n = len(current_ranks)
        d = current_ranks - self._prev_fitness_ranks
        spearman_rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
        spearman_rho = np.clip(spearman_rho, -1.0, 1.0)
    else:
        spearman_rho = 0.5
    
    # Signal 3: Fitness percentile spread (diversity signal)
    sorted_fit = np.sort(self._get_current_fitness())
    p10_idx = max(0, int(0.1 * len(sorted_fit)))
    p90_idx = min(len(sorted_fit) - 1, int(0.9 * len(sorted_fit)))
    spread = sorted_fit[p90_idx] - sorted_fit[p10_idx] if p90_idx > p10_idx else 1.0
    
    # Update EMAs for smoothing (F: temporal/dynamical)
    alpha = 0.25
    self._rank_change_ema = (1 - alpha) * self._rank_change_ema + alpha * rank_change
    self._spread_ema = (1 - alpha) * self._spread_ema + alpha * spread
    self._rank_corr_ema = (1 - alpha) * self._rank_corr_ema + alpha * spearman_rho
    
    # Compute adaptive weights based on signal variance
    if not hasattr(self, '_signal_variance'):
        self._signal_variance = {'change': [], 'corr': [], 'spread': []}
    
    self._signal_variance['change'].append(self._rank_change_ema)
    self._signal_variance['corr'].append(self._rank_corr_ema)
    self._signal_variance['spread'].append(self._spread_ema)
    
    # Keep bounded history
    for key in self._signal_variance:
        if len(self._signal_variance[key]) > 15:
            self._signal_variance[key].pop(0)
    
    # Compute variance-normalized weights
    eps = 1e-8
    variances = {k: np.var(v) + eps for k, v in self._signal_variance.items()}
    inv_var_weights = {k: 1.0 / variances[k] for k in variances}
    total_inv_var = sum(inv_var_weights.values())
    norm_weights = {k: v / total_inv_var for k, v in inv_var_weights.items()}
    
    # Hybrid signal for F adaptation
    # High correlation + low change = stable = increase F for exploration
    # Low correlation + high change = dynamic = decrease F for exploitation
    corr_signal = self._rank_corr_ema  # 0 to 1
    change_signal = 1.0 - np.clip(self._rank_change_ema, 0, 1)  # inverted, 0 to 1
    
    f_signal = (norm_weights['corr'] * corr_signal + 
                norm_weights['change'] * change_signal)
    
    # CR adaptation based on spread
    # Wide spread = diverse population = higher CR for convergence
    # Narrow spread = converged population = lower CR for diversification
    spread_normalized = np.clip(self._spread_ema / (self._spread_ema + eps), 0, 1)
    cr_signal = spread_normalized
    
    # Apply momentum-based adaptation
    F_target = 0.5 + 0.7 * f_signal
    CR_target = 0.5 + 0.4 * cr_signal
    
    # Blend with momentum
    new_F = self._F_adapt_momentum * self.F + (1 - self._F_adapt_momentum) * F_target
    new_CR = self._CR_adapt_momentum * self.CR + (1 - self._CR_adapt_momentum) * CR_target
    
    # Apply improvement rate feedback (D: success-history based)
    # Low improvement = increase exploration
    if improvement_rate < 0.1:
        new_F = min(new_F * 1.2, 1.5)
        new_CR = max(new_CR * 0.8, 0.3)
    elif improvement_rate > 0.3:
        new_F = max(new_F * 0.9, 0.3)
        new_CR = min(new_CR * 1.1, 0.95)
    
    # Clip to valid ranges
    self.F = np.clip(new_F, 0.3, 1.5)
    self.CR = np.clip(new_CR, 0.1, 0.99)
    
    # Update tracking state
    self._prev_fitness_ranks = current_ranks
    
    # Store history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)

def _get_current_fitness(self):
    """Helper to get current fitness values for rank computation."""
    # This should return the fitness array of current population
    # The actual implementation depends on how fitness is stored
    if hasattr(self, '_current_fitness'):
        return self._current_fitness
    # Fallback: return array of ones if no fitness available
    return np.ones(self.NP)
```