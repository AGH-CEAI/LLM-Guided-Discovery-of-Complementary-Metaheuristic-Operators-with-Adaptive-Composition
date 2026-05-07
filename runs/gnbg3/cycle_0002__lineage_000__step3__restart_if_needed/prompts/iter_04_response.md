Looking at the context, I need to implement `_restart_if_needed` using Category D (Fitness-landscape / rank-based) signals. The key is to analyze the fitness landscape through rank-based metrics like Spearman correlation and fitness percentiles, NOT raw distances or fitness values.

**Idea: Rank Correlation & Fitness Percentile Restart**
Uses Spearman rank correlation between generations to detect convergence, fitness percentile spread to measure landscape exploration, and success-rate-adjusted thresholds for adaptive restart decisions.

```python
def _restart_if_needed(self, population, fitness):
    """Full restart with diversity injection using rank-based fitness landscape analysis."""
    # Initialize tracking attributes for rank-based stagnation detection
    if not hasattr(self, '_prev_fitness_ranks'):
        self._prev_fitness_ranks = None
        self._rank_correlation_history = []
        self._ema_rank_correlation = 0.95
        self._ema_success_rate_restart = 0.0
        self._restart_count = 0
    
    # Compute current fitness ranks (D: fitness-landscape based)
    order = np.argsort(fitness)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(self.NP)
    ranks_normalized = ranks / max(self.NP - 1, 1)
    
    # Compute Spearman rank correlation with previous generation (D: rank-based)
    if self._prev_fitness_ranks is not None:
        # Spearman correlation = Pearson correlation on ranks
        mean_curr = np.mean(ranks_normalized)
        mean_prev = np.mean(self._prev_fitness_ranks)
        cov = np.mean((ranks_normalized - mean_curr) * (self._prev_fitness_ranks - mean_prev))
        std_curr = np.std(ranks_normalized)
        std_prev = np.std(self._prev_fitness_ranks)
        std_product = std_curr * std_prev
        
        if std_product > 1e-10:
            rank_correlation = np.clip(cov / std_product, -1.0, 1.0)
        else:
            rank_correlation = 0.0
        
        # Update EMA for temporal smoothing (D: success-history based)
        alpha = 0.2
        self._ema_rank_correlation = (1 - alpha) * self._ema_rank_correlation + alpha * rank_correlation
        
        # Track history for decision making
        self._rank_correlation_history.append(rank_correlation)
        if len(self._rank_correlation_history) > 10:
            self._rank_correlation_history.pop(0)
    else:
        rank_correlation = 0.0
        self._ema_rank_correlation = 0.0
    
    self._prev_fitness_ranks = ranks_normalized.copy()
    
    # Compute fitness percentile spread (D: fitness percentiles)
    p10 = np.percentile(fitness, 10)
    p50 = np.percentile(fitness, 50)
    p90 = np.percentile(fitness, 90)
    percentile_spread = (p90 - p10) / (abs(p50) + 1e-8)
    
    # Compute rank percentile spread (D: fitness percentiles)
    rank_p10 = np.percentile(ranks_normalized, 10)
    rank_p90 = np.percentile(ranks_normalized, 90)
    rank_spread = rank_p90 - rank_p10
    
    # Determine stagnation severity based on rank correlation and spread
    # High correlation (>0.85) + low spread = severe stagnation
    high_correlation = self._ema_rank_correlation > 0.85
    low_rank_spread = rank_spread < 0.4
    low_percentile_spread = percentile_spread < 0.1
    
    severe_stagnation = high_correlation and (low_rank_spread or low_percentile_spread)
    
    # Compute adaptive threshold based on success history (D: success-history based)
    # If recent restarts were successful, be more conservative
    self._ema_success_rate_restart = 0.9 * self._ema_success_rate_restart + 0.1 * (1.0 if severe_stagnation else 0.0)
    
    if self._ema_success_rate_restart > 0.7:
        # Recent restarts helped - require stronger signals
        threshold_multiplier = 1.5
        restart_prob = 0.5
    elif self._ema_success_rate_restart < 0.3:
        # Restarts not helping - be more aggressive to escape
        threshold_multiplier = 0.5
        restart_prob = 1.0
    else:
        threshold_multiplier = 1.0
        restart_prob = 0.8
    
    # Decide restart type based on stagnation severity
    base_threshold = 0.90 * threshold_multiplier
    
    if self._ema_rank_correlation > base_threshold or severe_stagnation:
        self._restart_count += 1
        
        # Partial restart: replace worst 60-80% based on severity
        if severe_stagnation:
            replace_fraction = 0.80
        else:
            replace_fraction = 0.60
        
        n_replace = max(int(self.NP * replace_fraction), 3)
        
        # Get worst individuals by rank (D: rank-based)
        worst_indices = np.argsort(ranks)[-n_replace:]
        
        # Generate new random individuals
        low, high = -100.0, 100.0
        new_individuals = np.random.uniform(low, high, (n_replace, self.dim))
        
        # Add adaptive noise based on rank spread
        noise_scale = max(5.0 * (1.0 - rank_spread), 1.0)
        new_individuals += np.random.randn(n_replace, self.dim) * noise_scale
        
        # Create new population
        new_population = population.copy()
        new_population[worst_indices] = np.clip(new_individuals, low, high)
        
        # Update fitness ranks for returned fitness
        new_fitness = fitness.copy()
        new_fitness[worst_indices] = np.inf
        
        # Get best solution
        best_idx = np.argmin(new_fitness)
        best_fitness = new_fitness[best_idx]
        best_solution = new_population[best_idx].copy()
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        return (new_population, best_fitness, best_solution)
    
    # No restart needed
    best_idx = np.argmin(fitness)
    return (None, fitness[best_idx], population[best_idx].copy())
```