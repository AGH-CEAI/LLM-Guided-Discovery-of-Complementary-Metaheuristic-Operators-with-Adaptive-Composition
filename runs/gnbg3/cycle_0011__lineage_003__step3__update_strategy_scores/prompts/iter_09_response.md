**Idea: Centroid Proximity Score**

Update strategy scores based on geometric proximity of each strategy's users to the centroid of all improved individuals. Strategies whose users cluster near where improvements occurred receive higher scores, combining spatial positioning with improvement rate in a dimension-normalized geometric signal.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Update strategy scores using centroid proximity to improved individuals (Category A: Geometry/Spatial).
    
    Strategies whose users cluster near the centroid of improved solutions score higher.
    This rewards spatial positioning that correlates with improvement, not raw fitness values.
    """
    NP = len(strategy_used)
    if NP == 0 or not hasattr(self, '_current_population'):
        return
    
    pop = self._current_population
    if pop is None or len(pop) != NP:
        return
    
    improved_mask = np.array(improved, dtype=bool)
    improved_idx = np.where(improved_mask)[0]
    
    # No improved individuals - decay all scores
    if len(improved_idx) == 0:
        self.strategy_scores *= 0.95
        return
    
    # Compute centroid of improved individuals (geometric center of success)
    improved_centroid = pop[improved_idx].mean(axis=0)
    
    # For each strategy, compute centroid distance to improved centroid
    for s in range(len(self.strategy_scores)):
        s_indices = np.where(strategy_used == s)[0]
        if len(s_indices) == 0:
            continue
        
        # Centroid of this strategy's users
        s_centroid = pop[s_indices].mean(axis=0)
        
        # Euclidean distance to improved centroid (dimension-normalized)
        centroid_dist = np.linalg.norm(s_centroid - improved_centroid)
        centroid_dist_normalized = centroid_dist / np.sqrt(self.dim)
        
        # Improvement rate for this strategy
        s_improved_count = np.sum(improved[s_indices])
        improvement_rate = s_improved_count / len(s_indices)
        
        # Geometric score: exponential decay with distance
        # Closer to improved centroid = higher score
        geo_score = np.exp(-centroid_dist_normalized * 2.0)
        
        # Combined score: improvement rate weighted by geometric proximity
        raw_score = improvement_rate * geo_score
        
        # Momentum-based update
        self.strategy_scores[s] = (1 - 0.1) * self.strategy_scores[s] + 0.1 * raw_score
    
    # Softmax normalization to convert to probabilities
    scores = self.strategy_scores - self.strategy_scores.max()
    exp_scores = np.exp(scores)
    self.strategy_scores = np.clip(exp_scores / exp_scores.sum(), 1e-10, None)
```