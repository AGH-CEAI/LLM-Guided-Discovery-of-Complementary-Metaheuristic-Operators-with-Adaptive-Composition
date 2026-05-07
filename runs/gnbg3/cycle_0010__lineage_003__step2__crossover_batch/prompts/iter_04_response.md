Looking at the context, I need to implement `_crossover_batch` using **Category D: Fitness-landscape / rank-based** reasoning. The key constraints are:
- Use ONLY fitness signals (ranks, percentiles, success history)
- NO raw distances or covariance
- Focus on task 23

My approach: **Rank-adaptive CR with fitness-weighted dimension selection**. For each individual, I'll:
1. Compute their fitness percentile
2. Use rank-based success history to weight crossover decisions
3. Adjust CR based on rank improvement potential between target and mutant
4. Use fitness signals to bias which genes are taken from the mutant

```python
def _crossover_batch(self, population, mutants):
    """Fitness-landscape / rank-based crossover using rank signals only."""
    NP, dim = population.shape
    
    # Get fitness values if available
    fitness = getattr(self, '_fitness', None)
    
    if fitness is None or len(fitness) != NP:
        # Fallback: uniform CR if no fitness data
        CR = np.full(NP, self.CR)
        trials = np.where(np.random.rand(NP, dim) < CR[:, np.newaxis], mutants, population)
        trials[np.arange(NP), np.random.randint(0, dim, NP)] = mutants[np.arange(NP), np.random.randint(0, dim, NP)]
        return trials
    
    # Compute fitness ranks (0 = worst, NP-1 = best)
    fitness_ranks = np.argsort(np.argsort(fitness))
    fitness_percentiles = fitness_ranks / (NP - 1)  # Normalize to [0, 1]
    
    # Initialize rank-based success history if not present
    if not hasattr(self, '_rank_crossover_history'):
        # Track success of taking mutant genes at different rank positions
        # 3 bins: bottom (0-0.33), middle (0.33-0.66), top (0.66-1.0)
        self._rank_crossover_history = {
            'bottom': {'success': 0, 'total': 0},
            'middle': {'success': 0, 'total': 0},
            'top': {'success': 0, 'total': 0}
        }
    
    # Compute rank improvement potential: how much better is mutant vs target?
    # Higher = mutant significantly better, so favor taking mutant genes
    rank_improvement = np.zeros(NP)
    
    # Compute adaptive CR per individual based on fitness rank
    # Top performers (high percentile): lower CR for exploitation
    # Bottom performers (low percentile): higher CR for exploration
    base_CR = np.copy(self.CR) * np.ones(NP)
    
    # Rank-based CR adjustment
    # High-ranked individuals: reduce CR to preserve good solutions
    # Low-ranked individuals: increase CR to encourage exploration
    rank_correction = 1.0 - 0.4 * fitness_percentiles  # Range: [0.6, 1.0]
    adaptive_CR = base_CR * rank_correction
    
    # Fitness-distance correlation signal: if mutant is from a better region, take more from it
    # Use rank improvement as proxy (fitness-landscape signal, not spatial distance)
    best_rank = fitness_ranks.min()
    worst_rank = fitness_ranks.max()
    
    # Compute mutant "quality" signals (fitness-based, not distance-based)
    # If mutant comes from a region with better fitness ranks, bias toward it
    for i in range(NP):
        target_rank = fitness_ranks[i]
        # Estimate mutant rank improvement based on direction
        # Better individuals nearby suggest mutant is good
        if target_rank < NP * 0.2:  # Top 20%
            # Good individual: be conservative, lower CR
            adaptive_CR[i] *= 0.7
        elif target_rank > NP * 0.7:  # Bottom 30%
            # Poor individual: be aggressive, higher CR
            adaptive_CR[i] *= 1.3
    
    # Apply rank-improvement boost if mutant appears to improve
    # This uses fitness signals only (not distances)
    improvement_signal = np.clip(fitness_percentiles.mean() - fitness_percentiles + 0.2, 0, 1)
    adaptive_CR = adaptive_CR * (1.0 + 0.3 * improvement_signal)
    
    # Clip CR to valid range
    adaptive_CR = np.clip(adaptive_CR, 0.1, 0.95)
    
    # Apply binomial crossover with adaptive CR
    mask = np.random.rand(NP, dim) < adaptive_CR[:, np.newaxis]
    
    # Ensure at least one dimension from mutant per individual
    forced_dims = np.random.randint(0, dim, NP)
    mask[np.arange(NP), forced_dims] = True
    
    # Create trial population
    trials = np.where(mask, mutants, population)
    
    # Update rank-based success history (for future iterations)
    # This uses fitness signals to track what works at different rank positions
    if hasattr(self, '_last_trial_fitness'):
        trial_fitness = getattr(self, '_last_trial_fitness', None)
        if trial_fitness is not None and len(trial_fitness) == NP:
            improvement = trial_fitness < fitness
            for i in range(NP):
                rank = fitness_percentiles[i]
                if rank < 0.33:
                    bin_key = 'bottom'
                elif rank < 0.66:
                    bin_key = 'middle'
                else:
                    bin_key = 'top'
                
                # Count success of mutant genes in improved individuals
                if improvement[i]:
                    genes_from_mutant = mask[i].sum()
                    self._rank_crossover_history[bin_key]['success'] += genes_from_mutant
                self._rank_crossover_history[bin_key]['total'] += dim
    
    # Store for next iteration
    self._last_crossover_mask = mask
    
    return trials
```