**Idea: RankCorrelationDE**
Adapt F/Cr using Spearman rank correlation between control parameters and improvement rates, focusing on fitness-percentile-weighted success history to target worst-case tasks.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using rank-based fitness-landscape analysis.
    
    Category D: Uses Spearman rank correlations, fitness percentiles, and
    success-history to drive parameter adaptation. Focuses on improving the
    worst-performing variants by weighting recent success by fitness quartile.
    """
    pop_size = len(improved_mask)
    
    # Compute raw success rate for history
    success_rate = np.mean(improved_mask)
    
    # Maintain history of (success_rate, F, Cr) tuples
    self.F_success_history.append((success_rate, F_used))
    self.Cr_success_history.append((success_rate, Cr_used))
    
    if len(self.F_success_history) > self.adaptation_window:
        self.F_success_history.pop(0)
        self.Cr_success_history.pop(0)
    
    # Need sufficient history for meaningful rank correlation
    if len(self.F_success_history) < 5:
        return
    
    # Extract arrays from history
    F_arr = np.array([f for _, f in self.F_success_history])
    Cr_arr = np.array([c for _, c in self.Cr_success_history])
    imp_arr = np.array([s for s, _ in self.F_success_history])
    
    # Compute Spearman rank correlation between F and improvement rate
    # (uses ranks, not raw values - robust to outliers)
    F_ranks = np.argsort(np.argsort(-F_arr))  # descending: higher F = higher rank
    imp_ranks = np.argsort(np.argsort(-imp_arr))  # descending: higher success = higher rank
    
    F_spearman = np.corrcoef(F_ranks, imp_ranks)[0, 1]
    Cr_spearman = np.corrcoef(np.argsort(np.argsort(-Cr_arr)), imp_ranks)[0, 1]
    
    # Handle NaN from insufficient variance
    if np.isnan(F_spearman):
        F_spearman = 0.0
    if np.isnan(Cr_spearman):
        Cr_spearman = 0.0
    
    # Compute fitness-percentile-weighted recent success (focus on worst tasks)
    recent_success = [s for s, _ in self.F_success_history[-5:]]
    recent_fitness_quartiles = np.percentile(recent_success, [25, 50, 75])
    
    # Weight: give 2x importance to improvements in worst fitness quartile
    weights = np.ones(5)
    for i, s in enumerate(recent_success):
        if s <= recent_fitness_quartiles[0]:  # bottom 25%
            weights[i] = 2.0
        elif s <= recent_fitness_quartiles[1]:  # bottom 50%
            weights[i] = 1.5
    
    weighted_success = np.average(recent_success, weights=weights)
    
    # Adapt F based on Spearman correlation (sign indicates direction of relationship)
    if F_spearman > 0.15:
        # Higher F correlates with more improvements
        self.F = np.clip(self.F * 1.15, 0.1, 2.0)
    elif F_spearman < -0.15:
        # Lower F correlates with more improvements
        self.F = np.clip(self.F * 0.85, 0.1, 2.0)
    
    # Adapt Cr based on Spearman correlation
    if Cr_spearman > 0.15:
        self.Cr = np.clip(self.Cr * 1.15, 0.1, 0.9)
    elif Cr_spearman < -0.15:
        self.Cr = np.clip(self.Cr * 0.85, 0.1, 0.9)
    
    # Secondary adjustment based on weighted success rate
    # (addresses cases where correlation is weak but overall trend is clear)
    if weighted_success > 0.35:
        self.F = np.clip(self.F * 1.05, 0.1, 2.0)
        self.Cr = np.clip(self.Cr * 0.98, 0.1, 0.9)
    elif weighted_success < 0.15:
        self.F = np.clip(self.F * 0.95, 0.1, 2.0)
        self.Cr = np.clip(self.Cr * 1.02, 0.1, 0.9)
```