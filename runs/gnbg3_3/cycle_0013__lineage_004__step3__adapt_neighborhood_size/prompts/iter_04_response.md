**Idea: Success-History Rank Entropy Adaptation**
Adapt neighborhood size using fitness rank entropy (smoothness of fitness landscape) combined with success-history tracking of which neighborhood sizes yielded improvements. High rank entropy = rugged landscape → larger neighborhoods for exploration; low rank entropy = converged → smaller neighborhoods for exploitation.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology neighborhood size using fitness-landscape signals.
    
    Category D: Uses ONLY fitness signals (ranks, entropy, success-history).
    - Fitness rank entropy: detects multimodal vs unimodal landscape
    - Spearman-like rank correlation: detects landscape smoothness across gens
    - Success-history: tracks which neighborhood sizes yielded improvements
    """
    # Fitness rank entropy: measure of fitness distribution uniformity
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    # Clip to avoid log(0)
    fr_clipped = np.clip(fitness_ranks, 1e-10, 1.0 - 1e-10)
    rank_entropy = -np.mean(fr_clipped * np.log(fr_clipped) - (1 - fr_clipped) * np.log(1 - fr_clipped + 1e-10))
    max_entropy = -np.log(0.5)
    entropy_ratio = np.clip(rank_entropy / (max_entropy + 1e-10), 0.0, 1.0)
    
    # Spearman-like rank correlation across generations (landscape smoothness)
    if hasattr(self, '_prev_fitness_ranks') and self._prev_fitness_ranks is not None:
        if len(self._prev_fitness_ranks) == len(fitness_ranks):
            if np.std(fitness_ranks) > 1e-10 and np.std(self._prev_fitness_ranks) > 1e-10:
                rank_corr = np.corrcoef(fitness_ranks, self._prev_fitness_ranks)[0, 1]
                rank_corr = np.clip(rank_corr, -1.0, 1.0)
            else:
                rank_corr = 0.0
        else:
            rank_corr = 0.0
    else:
        rank_corr = 0.0
    self._prev_fitness_ranks = fitness_ranks.copy()
    
    # Success-history for neighborhood sizes
    if not hasattr(self, '_neighborhood_history'):
        self._neighborhood_history = []
    if not hasattr(self, '_neighborhood_success'):
        self._neighborhood_success = {}
    
    # Track improvement this generation
    if self.global_best_fitness < getattr(self, '_prev_best_for_history', np.inf):
        improvement = True
    else:
        improvement = False
    self._prev_best_for_history = self.global_best_fitness
    
    current_ns = self.neighborhood_size
    if current_ns not in self._neighborhood_success:
        self._neighborhood_success[current_ns] = []
    self._neighborhood_success[current_ns].append(1.0 if improvement else 0.0)
    if len(self._neighborhood_success[current_ns]) > 10:
        self._neighborhood_success[current_ns].pop(0)
    
    # Compute success rate for current and neighboring sizes
    def success_rate(ns):
        if ns in self._neighborhood_success and len(self._neighborhood_success[ns]) >= 3:
            return np.mean(self._neighborhood_success[ns])
        return 0.5
    
    current_success = success_rate(current_ns)
    smaller_success = success_rate(max(1, current_ns - 1))
    larger_success = success_rate(min(self.np // 2, current_ns + 1))
    
    # Decision: combine entropy, rank correlation, and success-history
    # High entropy (rugged) OR low rank_corr (unstable) OR low success → increase neighborhood
    # Low entropy (converged) AND high rank_corr (stable) AND high success → decrease neighborhood
    
    # Exploration signal: rugged landscape needs larger neighborhoods
    exploration_signal = (1.0 - entropy_ratio) * 0.4 + (1.0 - rank_corr) * 0.3 + (1.0 - current_success) * 0.3
    exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
    
    # If neighboring size has higher success rate, consider switching
    if larger_success > current_success + 0.1:
        exploration_signal = min(1.0, exploration_signal + 0.3)
    if smaller_success > current_success + 0.1 and exploration_signal < 0.5:
        exploration_signal = max(0.0, exploration_signal - 0.2)
    
    # Adapt neighborhood size based on exploration signal
    if exploration_signal > 0.6:
        # High exploration need → increase neighborhood
        self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
    elif exploration_signal < 0.3:
        # Low exploration need → decrease neighborhood
        self.neighborhood_size = max(1, self.neighborhood_size - 1)
    # else: keep current neighborhood size
```