**Idea: KL Divergence Strategy Credit via Fitness Distribution Shift**

Use KL divergence to measure how each strategy shifts the population's fitness distribution toward lower (better) values. Strategies that produce more concentrated distributions on improvements gain higher credit. Combined with entropy of the improvement magnitude distribution to reward diverse exploration.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Information-theoretic credit assignment using KL divergence and entropy of fitness distributions."""
    n_strategies = len(self.strategy_scores)
    n = len(strategy_used)
    
    if n < 5:
        return
    
    # Get fitness values from the current population
    fitness = self._current_fitness
    
    # Compute reference distribution P(fitness) using histogram
    f_min, f_max = fitness.min(), fitness.max()
    f_range = max(f_max - f_min, 1e-10)
    n_bins = min(12, n // 3)
    bins = np.linspace(f_min - 0.05 * f_range, f_max + 0.05 * f_range, n_bins + 1)
    
    hist_ref, _ = np.histogram(fitness, bins=bins, density=True)
    hist_ref = np.maximum(hist_ref, 1e-10)
    hist_ref /= hist_ref.sum()
    
    # For each strategy, compute conditional distribution P(fitness | strategy=s)
    for s in range(n_strategies):
        mask_s = strategy_used == s
        n_s = np.sum(mask_s)
        
        if n_s < 2:
            continue
        
        fitness_s = fitness[mask_s]
        hist_s, _ = np.histogram(fitness_s, bins=bins, density=True)
        hist_s = np.maximum(hist_s, 1e-10)
        hist_s /= hist_s.sum()
        
        # KL divergence: measures how strategy shifts distribution
        kl_div = np.sum(hist_s * np.log(hist_s / hist_ref + 1e-10))
        
        # Compute improvement entropy: diversity of improvement magnitudes
        improved_s = improved[mask_s]
        if np.sum(improved_s) >= 2:
            fit_imp = fitness_s[improved_s]
            imp_min, imp_max = fit_imp.min(), fit_imp.max()
            imp_range = max(imp_max - imp_min, 1e-10)
            imp_bins = np.linspace(imp_min - 0.05 * imp_range, imp_max + 0.05 * imp_range, 8)
            imp_hist, _ = np.histogram(fit_imp, bins=imp_bins, density=True)
            imp_hist = np.maximum(imp_hist, 1e-10)
            imp_hist /= imp_hist.sum()
            entropy_imp = -np.sum(imp_hist * np.log(imp_hist))
        else:
            entropy_imp = 0.0
        
        # Success rate
        success_rate = np.mean(improved_s)
        
        # Normalize KL (can be negative for converging distributions)
        kl_norm = np.clip(kl_div / (np.log(n_bins) + 1e-10), -1, 1)
        
        # Information gain: KL captures distribution shift, entropy captures diversity
        info_gain = 0.5 * kl_norm + 0.3 * (entropy_imp / (np.log(8) + 1e-10)) + 0.2 * success_rate
        
        # Exponential moving average update
        self.strategy_scores[s] = 0.85 * self.strategy_scores[s] + 0.15 * info_gain
    
    # Bound scores to prevent numerical issues
    self.strategy_scores = np.clip(self.strategy_scores, -10, 10)
```