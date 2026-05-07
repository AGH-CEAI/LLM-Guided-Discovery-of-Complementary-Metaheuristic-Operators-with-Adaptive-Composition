Looking at this problem, I need to implement a fitness-landscape/rank-based strategy scoring mechanism that focuses purely on fitness signals (no distances or geometry). The key insight is that strategies should be rewarded not just for success rate, but for the *rank context* in which they succeed — improving a low-ranked individual is more valuable than improving an already-good one.

**Idea: Kendall Tau Rank Correlation with Fitness Percentile Weighting**

This approach computes the rank correlation between strategy assignments and fitness outcomes, combined with percentile-weighted success rates. Strategies that consistently help struggling individuals climb the rankings get higher scores.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """
    Category D: Fitness-landscape / rank-based strategy scoring.
    Uses Kendall Tau rank correlation + fitness percentile-weighted success rates.
    No distances or geometry — pure fitness signal analysis.
    """
    NP = len(strategy_used)
    n_strategies = len(self.strategy_names)
    
    # Initialize per-strategy tracking if first call
    if not hasattr(self, '_strategy_rank_improvements'):
        self._strategy_rank_improvements = np.zeros(n_strategies)
        self._strategy_success_counts = np.zeros(n_strategies)
        self._strategy_attempt_counts = np.zeros(n_strategies)
        self._strategy_fitness_history = [[] for _ in range(n_strategies)]
        self._rank_corr_history = []
    
    # Compute fitness percentile for each individual (0 = worst, 1 = best)
    if not hasattr(self, '_prev_fitness_ranks'):
        self._prev_fitness_ranks = np.zeros(NP)
    
    # Get current fitness (need to access from optimizer state)
    if not hasattr(self, '_current_fitness'):
        return  # No fitness available yet
    
    current_fitness = self._current_fitness
    valid_mask = np.isfinite(current_fitness)
    
    if np.sum(valid_mask) < 5:
        return
    
    # Compute current fitness ranks (normalized 0-1)
    fitness_sorted = np.argsort(current_fitness[valid_mask])
    current_ranks = np.zeros(np.sum(valid_mask))
    current_ranks[fitness_sorted] = np.linspace(0, 1, len(fitness_sorted))
    
    # Compute rank improvement per individual
    rank_improvement = self._prev_fitness_ranks[valid_mask] - current_ranks
    
    # Update per-strategy metrics
    for s in range(n_strategies):
        s_mask = (strategy_used == s) & valid_mask
        if np.sum(s_mask) == 0:
            continue
        
        # Track attempts
        self._strategy_attempt_counts[s] += np.sum(s_mask)
        
        # Track successes
        successes = improved[s_mask]
        self._strategy_success_counts[s] += np.sum(successes)
        
        # Compute fitness percentile of individuals this strategy acted on
        avg_percentile = np.mean(current_ranks)  # percentile of individuals this strategy worked on
        
        # Weight success by how "difficult" the individuals were (lower percentile = harder)
        difficulty_weight = 1.0 - avg_percentile + 0.1  # Add 0.1 to avoid div by zero issues
        
        # Rank improvement for this strategy's successes
        if np.any(successes):
            avg_rank_improvement = np.mean(rank_improvement[s_mask][successes])
            self._strategy_rank_improvements[s] += difficulty_weight * max(avg_rank_improvement, 0)
        
        # Store fitness history for correlation analysis
        self._strategy_fitness_history[s].extend(current_fitness[s_mask].tolist())
        # Keep bounded
        if len(self._strategy_fitness_history[s]) > 1000:
            self._strategy_fitness_history[s] = self._strategy_fitness_history[s][-500:]
    
    # Compute Kendall Tau rank correlation between strategy indices and fitness
    # Higher fitness (lower value for minimization) should correlate with better strategies
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= 10:
        strat_vals = strategy_used[valid_indices].astype(float)
        fitness_vals = current_fitness[valid_indices]
        
        # Normalize fitness to 0-1 scale for correlation
        f_min, f_max = np.min(fitness_vals), np.max(fitness_vals)
        if f_max > f_min:
            fitness_norm = (fitness_vals - f_min) / (f_max - f_min)
        else:
            fitness_norm = np.zeros_like(fitness_vals)
        
        # Compute Kendall tau manually (avoid scipy dependency)
        n = len(strat_vals)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                strat_diff = (strat_vals[i] - strat_vals[j]) * (fitness_norm[i] - fitness_norm[j])
                if strat_diff > 0:
                    concordant += 1
                elif strat_diff < 0:
                    discordant += 1
        
        n_pairs = n * (n - 1) / 2
        if n_pairs > 0:
            tau = (concordant - discordant) / n_pairs
        else:
            tau = 0.0
        
        self._rank_corr_history.append(tau)
        if len(self._rank_corr_history) > 20:
            self._rank_corr_history.pop(0)
    
    # Compute success rates per strategy
    success_rates = np.zeros(n_strategies)
    for s in range(n_strategies):
        if self._strategy_attempt_counts[s] > 0:
            success_rates[s] = self._strategy_success_counts[s] / self._strategy_attempt_counts[s]
    
    # Compute rank improvement scores (normalized)
    rank_scores = self._strategy_rank_improvements.copy()
    max_rank = np.max(rank_scores) if np.max(rank_scores) > 0 else 1.0
    rank_scores = rank_scores / max_rank
    
    # Combine: 60% success rate, 30% rank improvement, 10% correlation adjustment
    correlation_factor = 1.0 + 0.1 * np.mean(self._rank_corr_history[-5:]) if len(self._rank_corr_history) > 0 else 1.0
    correlation_factor = np.clip(correlation_factor, 0.8, 1.2)
    
    combined_scores = 0.6 * success_rates + 0.3 * rank_scores
    combined_scores *= correlation_factor
    
    # Apply exponential moving average for temporal smoothing
    ema_decay = 0.7
    self.strategy_scores = ema_decay * self.strategy_scores + (1 - ema_decay) * combined_scores
    
    # Ensure minimum scores for exploration
    self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)
    
    # Update previous fitness ranks for next iteration
    self._prev_fitness_ranks[valid_mask] = current_ranks
```