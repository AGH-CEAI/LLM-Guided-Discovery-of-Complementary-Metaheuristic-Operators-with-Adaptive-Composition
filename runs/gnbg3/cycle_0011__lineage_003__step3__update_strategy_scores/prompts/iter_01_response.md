**Idea: Geometric Strategy Attribution via Pairwise Distance Analysis**

This approach credits strategies based on the geometric distribution of successful vs. unsuccessful mutations in the population, measuring spatial spread, centroid displacement, and local neighborhood density changes. Purely geometric—no fitness values or information theory.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Geometry-based strategy scoring using spatial distribution of success."""
    if not hasattr(self, '_prev_population'):
        self._prev_population = None
        self._strategy_geo_scores = np.ones(len(self.strategy_names))
    
    # Get population from the optimizer state
    if not hasattr(self, '_current_population') or self._current_population is None:
        return
    
    pop = self._current_population
    NP, dim = pop.shape
    
    # Compute pairwise Euclidean distances (vectorized)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    distances = np.sqrt(sq_dists)
    
    # Global population metrics for normalization
    global_spread = np.mean(distances[np.isfinite(distances)])
    centroid = pop.mean(axis=0)
    
    # Per-strategy geometric analysis
    new_scores = np.zeros(len(self.strategy_names))
    
    for s in range(len(self.strategy_names)):
        # Mask for individuals where this strategy was used
        strategy_mask = (strategy_used == s)
        n_used = np.sum(strategy_mask)
        
        if n_used < 1:
            new_scores[s] = 0.0
            continue
        
        # Separate successful and unsuccessful individuals for this strategy
        success_mask = strategy_mask & improved
        fail_mask = strategy_mask & ~improved
        
        n_success = np.sum(success_mask)
        n_fail = np.sum(fail_mask)
        
        if n_success == 0:
            # Strategy had no successes - check geometric diversity it created
            success_rate = 0.0
            # Measure how spread out the failed attempts were (exploration value)
            if n_used > 1:
                failed_points = pop[fail_mask]
                spread = np.std(failed_points, axis=0).mean()
                exploration_bonus = spread / (global_spread + 1e-10)
            else:
                exploration_bonus = 0.5
            geometric_score = exploration_bonus
        else:
            success_rate = n_success / n_used
            
            # Geometric spread of successful mutations
            success_points = pop[success_mask]
            success_centroid = success_points.mean(axis=0)
            success_spread = np.std(success_points, axis=0).mean()
            
            # Normalized spread: higher is better (found diverse good regions)
            normalized_spread = success_spread / (global_spread + 1e-10)
            
            # Centroid distance from overall population centroid
            # Successful region away from average = found new territory
            centroid_offset = np.linalg.norm(success_centroid - centroid)
            centroid_bonus = centroid_offset / (global_spread + 1e-10)
            
            # Local neighborhood quality: avg distance to k nearest neighbors
            k = min(3, n_success)
            if k >= 1 and n_success > 1:
                s_indices = np.where(success_mask)[0]
                local_densities = []
                for idx in s_indices:
                    nn_dists = np.sort(distances[idx, s_indices])
                    if len(nn_dists) > 1:
                        local_densities.append(np.mean(nn_dists[1:k+1]))
                neighborhood_coherence = 1.0 / (np.mean(local_densities) + 1e-10)
            else:
                neighborhood_coherence = 1.0
            
            # Compare success region vs failure region geometrically
            if n_fail > 0:
                fail_points = pop[fail_mask]
                fail_centroid = fail_points.mean(axis=0)
                fail_spread = np.std(fail_points, axis=0).mean()
                
                # Distance between success and failure centroids
                region_separation = np.linalg.norm(success_centroid - fail_centroid)
                
                # Ratio of spreads: success more spread = better exploration
                spread_ratio = (success_spread + 1e-10) / (fail_spread + 1e-10)
                
                # Combined geometric quality
                geometric_score = (normalized_spread * 0.3 + 
                                  centroid_bonus * 0.2 + 
                                  np.clip(region_separation / (global_spread + 1e-10), 0, 1) * 0.3 +
                                  np.clip(spread_ratio, 0, 2) * 0.1 +
                                  np.clip(neighborhood_coherence * 0.01, 0, 1) * 0.1)
            else:
                # All attempts succeeded - great strategy for this region
                geometric_score = (normalized_spread * 0.4 + 
                                  centroid_bonus * 0.3 +
                                  np.clip(neighborhood_coherence * 0.01, 0, 1) * 0.3)
        
        # Combine success rate with geometric quality
        # Geometric quality is weighted by how confident we are (more samples)
        sample_confidence = np.sqrt(n_used / NP)
        new_scores[s] = success_rate * (0.5 + 0.5 * geometric_score * sample_confidence)
    
    # Apply momentum-based update
    momentum = 0.7
    self._strategy_geo_scores = (momentum * self._strategy_geo_scores + 
                                  (1 - momentum) * new_scores)
    
    # Ensure positive scores and normalize
    self._strategy_geo_scores = np.clip(self._strategy_geo_scores, 0.01, None)
    self._strategy_geo_scores /= self._strategy_geo_scores.sum()
    
    # Update main strategy scores
    self.strategy_scores = self._strategy_geo_scores.copy()
    
    # Store population for next iteration comparison
    self._prev_population = pop.copy()
```