**Idea: Centroid-Relative Directional Alignment**

This approach evaluates strategy credit purely by geometric relationships: for each strategy, compute the mutation vectors, measure their alignment with the population movement toward the centroid of improved solutions, and their contribution to population spread. Strategies whose mutations consistently point toward promising spatial regions get higher scores.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment based on geometric alignment of mutation directions."""
    if len(improved) == 0 or self.NP < 2:
        return
    
    NP = self.NP
    
    # Get current population from optimizer state
    pop = getattr(self, '_current_population', None)
    if pop is None or len(pop) < 5:
        return
    
    # Compute population geometric properties
    centroid = pop.mean(axis=0)
    pop_spread = np.std(pop, axis=0).mean()
    
    # Compute mutation direction toward centroid
    to_centroid = centroid - pop
    
    # For each strategy, compute directional alignment and spread contribution
    strategy_counts = np.bincount(strategy_used, minlength=len(self.strategy_scores))
    
    for s_idx in range(len(self.strategy_scores)):
        mask = strategy_used == s_idx
        n_s = mask.sum()
        
        if n_s < 1:
            continue
        
        # Get individuals using this strategy
        s_improved = improved[mask]
        s_to_centroid = to_centroid[mask]
        
        if s_improved.sum() == 0:
            # No improvements: small penalty
            self.strategy_scores[s_idx] *= 0.95
            continue
        
        # Compute centroid of improved solutions using this strategy
        s_pop_improved = pop[mask][s_improved]
        improved_centroid = s_pop_improved.mean(axis=0)
        
        # Direction from population to improved centroid
        dir_to_improved = improved_centroid - centroid
        
        # Directional alignment: cosine similarity between individual directions and global direction
        norms_individual = np.linalg.norm(s_to_centroid, axis=1)
        valid_mask = norms_individual > 1e-12
        
        if valid_mask.sum() > 0:
            norm_global = np.linalg.norm(dir_to_improved)
            if norm_global > 1e-12:
                cos_sims = np.dot(s_to_centroid[valid_mask], dir_to_improved) / (
                    norms_individual[valid_mask] * norm_global
                )
                alignment_score = np.clip(cos_sims.mean(), -1, 1)
            else:
                alignment_score = 0.0
        else:
            alignment_score = 0.0
        
        # Spatial spread contribution: how much does this strategy expand population coverage
        s_pop_all = pop[mask]
        s_spread = np.std(s_pop_all, axis=0).mean()
        spread_ratio = s_spread / max(pop_spread, 1e-10)
        
        # Combined geometric score
        geo_score = 0.6 * (alignment_score + 1) / 2 + 0.4 * np.clip(spread_ratio, 0.5, 2.0)
        
        # Improvement rate for this strategy
        improvement_rate = s_improved.mean()
        
        # Weighted update
        weight = improvement_rate * 0.3 + 0.1
        self.strategy_scores[s_idx] += weight * (geo_score - 0.5)
    
    # Dampen scores to prevent runaway
    self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
```