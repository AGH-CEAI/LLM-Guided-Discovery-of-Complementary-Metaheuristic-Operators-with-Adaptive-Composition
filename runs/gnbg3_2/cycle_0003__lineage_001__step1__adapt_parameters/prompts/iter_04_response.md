**Idea: Spearman Rank Correlation + Fitness Percentile Landscape Adaptation**

Adapt F and Cr based on Spearman rank correlation between consecutive generations, improvement magnitude percentiles, and fitness spread — all pure fitness-signals that detect convergence traps and deceptive landscapes without any distance or covariance computations.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using Spearman rank correlation and fitness percentile landscape analysis."""
    from scipy.stats import spearmanr
    
    # Get current fitness from stored state
    if not hasattr(self, '_cached_fitness'):
        return
    current_fitness = self._cached_fitness
    pop_size = len(current_fitness)
    
    # Initialize rank tracking on first call
    if not hasattr(self, 'fitness_rank_history'):
        self.fitness_rank_history = []
        self.improvement_percentile_history = []
        self.fitness_spread_history = []
    
    # Compute current fitness ranks (pure fitness signal)
    current_ranks = np.argsort(np.argsort(current_fitness)) / max(pop_size - 1, 1)
    
    # Compute Spearman rank correlation between consecutive generations
    if len(self.fitness_rank_history) >= 1:
        prev_ranks = self.fitness_rank_history[-1]
        rank_corr, _ = spearmanr(prev_ranks, current_ranks)
        if np.isnan(rank_corr):
            rank_corr = 0.5
    else:
        rank_corr = 0.5
    
    # Compute improvement magnitude percentiles (fitness signal: how good are the wins?)
    if np.sum(improved_mask) > 0 and hasattr(self, '_cached_trial_fitness'):
        trial_fitness = self._cached_trial_fitness
        fitness_diff = current_fitness - trial_fitness
        improvements = fitness_diff[improved_mask]
        if len(improvements) > 0:
            improv_percentile = np.percentile(improvements, 75)
        else:
            improv_percentile = 0.0
    else:
        improv_percentile = 0.0
    
    # Compute fitness percentile spread (landscape clustering signal)
    fit_25 = np.percentile(current_fitness, 25)
    fit_75 = np.percentile(current_fitness, 75)
    fit_mean = np.mean(current_fitness)
    fit_spread = (fit_75 - fit_25) / (np.abs(fit_mean) + 1e-10)
    
    # Regime detection via rank-based signals
    high_rank_stability = rank_corr > 0.75
    low_rank_stability = rank_corr < 0.4
    tight_clustering = fit_spread < 0.15
    small_improvements = improv_percentile < 0.25 * (fit_75 - fit_25 + 1e-10)
    
    # Fitness-landscape / rank-based adaptation logic
    if high_rank_stability and tight_clustering:
        # CONVERGENCE TRAP: ranks stable + population clustered = stuck in local optimum
        # Need aggressive exploration with high F
        F_adjustment = 1.35
        Cr_adjustment = 0.75
    elif low_rank_stability and improv_percentile > 0:
        # ACTIVE EXPLORATION: ranks shuffling + good improvements = healthy search
        # Can afford more exploitation with lower F
        F_adjustment = 0.85
        Cr_adjustment = 1.1
    elif small_improvements and tight_clustering:
        # DEEP TRAP: tiny wins in tight cluster = deceptive local optimum
        # Need maximum perturbation
        F_adjustment = 1.4
        Cr_adjustment = 0.7
    elif high_rank_stability and not tight_clustering:
        # RANK STABLE BUT SPREAD: converging to good region
        # Moderate exploitation
        F_adjustment = 0.9
        Cr_adjustment = 1.05
    else:
        # NEUTRAL: rank correlation moderate = normal search dynamics
        # Gentle correction based on rank correlation magnitude
        F_adjustment = 0.95 + 0.1 * (1.0 - rank_corr)
        Cr_adjustment = 0.95 + 0.15 * rank_corr
    
    # Apply adjustments with bounds
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    # Store current state for next generation's rank correlation
    self.fitness_rank_history.append(current_ranks.copy())
    if len(self.fitness_rank_history) > 8:
        self.fitness_rank_history.pop(0)
    
    self.improvement_percentile_history.append(improv_percentile)
    if len(self.improvement_percentile_history) > 8:
        self.improvement_percentile_history.pop(0)
    
    self.fitness_spread_history.append(fit_spread)
    if len(self.fitness_spread_history) > 8:
        self.fitness_spread_history.pop(0)
```