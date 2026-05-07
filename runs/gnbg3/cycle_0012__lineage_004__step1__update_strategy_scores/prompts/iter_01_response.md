**Idea: Geometric Quality Score via k-NN Graph Structure**

Category A: Geometry / spatial approach that scores strategies based on how their successful mutations affect the k-NN graph structure of the population (connectivity, clustering, and spatial coherence) rather than using fitness or information-theoretic measures.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """
    Geometry-based strategy credit assignment using k-NN graph structure analysis.
    
    Scores strategies based on how mutations affect the population's spatial
    organization: connectivity in k-NN graph, local clustering, and spread.
    """
    if not hasattr(self, '_geo_pop_history'):
        self._geo_pop_history = []
    
    # Store current population geometric state for next iteration comparison
    if hasattr(self, '_current_population') and self._current_population is not None:
        pop = self._current_population
        if len(pop) >= 5:
            # Compute current geometric features
            centroid = pop.mean(axis=0)
            distances_to_centroid = np.linalg.norm(pop - centroid, axis=1)
            geo_features = {
                'spread': distances_to_centroid.mean(),
                'spread_var': distances_to_centroid.var(),
                'max_dist': distances_to_centroid.max(),
            }
            self._geo_pop_history.append(geo_features)
            if len(self._geo_pop_history) > 5:
                self._geo_pop_history.pop(0)
    
    if not hasattr(self, '_strategy_geo_scores'):
        self._strategy_geo_scores = np.zeros(len(self.strategy_names))
        self._strategy_geo_counts = np.zeros(len(self.strategy_names))
    
    # Need population to compute geometric features
    if not hasattr(self, '_current_population') or self._current_population is None:
        return
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < 5:
        return
    
    # Compute k-NN graph for current population (k based on NP)
    k = max(2, min(5, NP // 10))
    
    # Pairwise squared distances
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # k-nearest neighbors indices
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    knn_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], knn_indices]
    
    # Compute geometric features per individual
    centroid = pop.mean(axis=0)
    dist_to_centroid = np.linalg.norm(pop - centroid, axis=1)
    
    # Local clustering coefficient
    clustering = np.zeros(NP)
    for i in range(NP):
        neighbors = set(knn_indices[i])
        if len(neighbors) >= 2:
            edges = 0
            possible = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj:
                        possible += 1
                        if nj in set(knn_indices[ni]):
                            edges += 1
            clustering[i] = edges / max(possible, 1)
    
    # Compute per-strategy geometric quality
    for strat_idx in range(len(self.strategy_names)):
        # Get individuals where this strategy was used
        mask = strategy_used == strat_idx
        if not np.any(mask):
            continue
        
        n_this = mask.sum()
        
        # Geometric features for individuals where this strategy succeeded
        improved_mask = mask & improved
        if np.any(improved_mask):
            # Quality 1: Average local clustering of successful mutations
            # Higher clustering = mutations stay within well-connected regions
            quality_clustering = clustering[improved_mask].mean()
            
            # Quality 2: Average distance to centroid
            # Moderate distances suggest balanced exploration
            quality_spread = dist_to_centroid[improved_mask].mean()
            
            # Quality 3: Local edge length consistency
            # Lower variance in k-NN distances = more stable geometric position
            local_edges = knn_sq_dists[improved_mask]
            quality_stability = 1.0 / (1.0 + local_edges.var())
            
            # Quality 4: Proximity to population boundary (inverse)
            # Mutations too close to boundary may reduce diversity
            pop_radius = dist_to_centroid.max()
            boundary_proximity = dist_to_centroid[improved_mask] / max(pop_radius, 1e-6)
            quality_boundary = 1.0 - boundary_proximity.mean()
            
            # Combined geometric quality score
            geo_quality = (
                0.35 * quality_clustering +
                0.25 * (quality_spread / max(dist_to_centroid.mean(), 1e-6)) +
                0.25 * quality_stability +
                0.15 * max(quality_boundary, 0.0)
            )
            
            # Update running geometric score with exponential moving average
            alpha = 0.3
            self._strategy_geo_scores[strat_idx] = (
                (1 - alpha) * self._strategy_geo_scores[strat_idx] +
                alpha * geo_quality
            )
            self._strategy_geo_counts[strat_idx] += n_this
    
    # Convert geometric scores to strategy selection weights
    # Use softmax with temperature based on geometric variance
    if self._strategy_geo_counts.sum() > 0:
        # Normalize counts to get experience weight
        experience = self._strategy_geo_counts / max(self._strategy_geo_counts.sum(), 1)
        
        # Blend geometric quality with experience
        geo_normalized = self._strategy_geo_scores / max(self._strategy_geo_scores.max(), 1e-6)
        
        # Combined score: geometric quality * sqrt(experience) for exploration bonus
        combined = geo_normalized * np.sqrt(experience + 0.1)
        
        # Softmax conversion
        temperature = 0.5
        exp_scores = np.exp((combined - combined.max()) / temperature)
        softmax_weights = exp_scores / exp_scores.sum()
        
        # Blend with existing strategy_scores (preserve some history)
        blend_factor = 0.7
        self.strategy_scores = (
            blend_factor * self.strategy_scores +
            (1 - blend_factor) * (softmax_weights * 10.0 + 1.0)
        )
        
        # Ensure minimum scores
        self.strategy_scores = np.maximum(self.strategy_scores, 0.1)
```