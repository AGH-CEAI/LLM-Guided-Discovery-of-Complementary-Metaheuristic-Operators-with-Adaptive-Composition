**Idea: Bootstrap UCB Neighborhood Sampler**

Monte Carlo sampling of neighborhood size candidates with bootstrap confidence estimation and UCB-based stochastic selection. Uses empirical performance estimates from subsampled fitness evaluations to probabilistically choose neighborhood size, fundamentally stochastic unlike the deterministic threshold-based original.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology via stochastic sampling of neighborhood candidates.
    
    Category G: Uses Monte Carlo sampling and bootstrap estimation to
    probabilistically select neighborhood size based on empirical performance.
    """
    # Generate candidate neighborhood sizes (stochastic candidate generation)
    candidates = list(range(1, max(2, min(self.np // 2, self.neighborhood_size + 3)) + 1))
    if len(candidates) < 2:
        return  # Not enough candidates to sample meaningfully
    
    # Bootstrap sampling: estimate expected improvement for each candidate
    n_bootstrap = 20
    bootstrap_scores = {c: [] for c in candidates}
    
    for _ in range(n_bootstrap):
        # Subsample population for Monte Carlo estimate
        subsample_size = max(5, self.np // 2)
        subsample_idx = np.random.choice(self.np, size=subsample_size, replace=False)
        
        for candidate in candidates:
            # Estimate local best quality for this neighborhood size
            local_best_qualities = []
            for idx in subsample_idx:
                # Ring neighborhood around this particle
                left = (idx - candidate) % self.np
                right = (idx + candidate + 1) % self.np
                if left < right:
                    nb_indices = np.arange(left, right)
                else:
                    nb_indices = np.concatenate([np.arange(left, self.np), np.arange(0, right)])
                
                if len(nb_indices) > 0:
                    nb_fitness = self.personal_best_fitness[nb_indices]
                    local_best_qualities.append(np.min(nb_fitness))
            
            if local_best_qualities:
                # Bootstrap estimate: quality of local bests accessible
                bootstrap_scores[candidate].append(np.mean(local_best_qualities))
    
    # Compute expected performance and uncertainty for each candidate
    candidate_stats = {}
    for candidate in candidates:
        scores = bootstrap_scores[candidate]
        if scores:
            mean_perf = np.mean(scores)
            std_perf = np.std(scores) + 1e-10
            # UCB-style: higher is better, penalize high variance
            candidate_stats[candidate] = mean_perf - 0.5 * std_perf
        else:
            candidate_stats[candidate] = 0.0
    
    # Diversity signal as additional stochastic probe
    diversity = self._compute_diversity()
    diversity_norm = np.clip(diversity / (self.diversity_threshold_high + 1e-10), 0.0, 1.0)
    
    # Probabilistic selection via softmax sampling
    temperatures = [0.5 + 0.5 * diversity_norm for _ in candidates]
    
    # Convert stats to probabilities (lower fitness = better)
    min_stat = min(candidate_stats.values())
    max_stat = max(candidate_stats.values())
    stat_range = max_stat - min_stat + 1e-10
    
    weights = []
    for i, candidate in enumerate(candidates):
        normalized = (candidate_stats[candidate] - min_stat) / stat_range
        # Temperature controls stochasticity: high diversity → more exploration
        weight = np.exp(normalized / temperatures[i])
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-10)
    
    # Sample neighborhood size stochastically
    chosen_idx = np.random.choice(len(candidates), p=weights)
    chosen_size = candidates[chosen_idx]
    
    # Update with momentum for stability
    if not hasattr(self, '_neighborhood_momentum'):
        self._neighborhood_momentum = chosen_size
    self._neighborhood_momentum = 0.7 * self._neighborhood_momentum + 0.3 * chosen_size
    
    # Final decision with bounded update
    target = int(np.round(self._neighborhood_momentum))
    self.neighborhood_size = max(1, min(self.np // 2, target))
```