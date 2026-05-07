Looking at the task context and the assigned category (E - Topology / graph-based), I need to create a fundamentally different approach to updating strategy scores that uses graph-based reasoning about the population structure.

The key insight: Instead of just counting strategy successes, I'll build a k-NN graph over the population and weight credit assignment by graph centrality. Strategies that produce improvements in central (well-connected) individuals are more valuable because those individuals have more influence on the population's trajectory.

**Idea: Centrality-Weighted Graph Credit Assignment**
Use k-NN graph to compute each individual's centrality score, then weight strategy credit assignment by how central the improving individual is. This captures which strategies work well in different population topologies.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Topology-aware strategy credit using k-NN graph centrality weights."""
    NP = len(strategy_used)
    
    if NP < 5 or not hasattr(self, '_current_population') or self._current_population is None:
        # Fallback: simple success-rate update
        for s in range(len(self.strategy_scores)):
            mask = strategy_used == s
            if np.any(mask):
                success_rate = np.mean(improved[mask])
                self.strategy_scores[s] = 0.9 * self.strategy_scores[s] + 0.1 * success_rate
        return
    
    pop = self._current_population
    
    # Build k-NN graph
    k = max(2, min(5, NP // 8))
    
    # Compute pairwise squared distances
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute graph centrality: count how often each individual appears in others' neighborhoods
    centrality = np.zeros(NP)
    for i in range(NP):
        centrality[nearest[i]] += 1
    
    # Normalize centrality to [0, 1]
    if centrality.max() > 0:
        centrality /= centrality.max()
    else:
        centrality = np.ones(NP) / NP
    
    # Also compute local clustering coefficient for each individual
    clustering = np.zeros(NP)
    for i in range(NP):
        neighbors = set(nearest[i])
        if len(neighbors) >= 2:
            edges = 0
            possible = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj:
                        possible += 1
                        if nj in set(nearest[ni]):
                            edges += 1
            clustering[i] = edges / max(possible, 1)
    
    # Combine centrality and clustering into topology influence score
    # High centrality + low clustering = valuable bridge individuals
    topology_weight = 0.7 * centrality + 0.3 * (1.0 - clustering)
    topology_weight /= (topology_weight.max() + 1e-12)
    
    # Track smoothed topology history for adaptive weighting
    if not hasattr(self, '_topology_weight_history'):
        self._topology_weight_history = []
    self._topology_weight_history.append(topology_weight.copy())
    if len(self._topology_weight_history) > 5:
        self._topology_weight_history.pop(0)
    
    # Compute population fragmentation index from graph
    avg_centrality = np.mean(centrality)
    fragmentation = 1.0 - avg_centrality  # High when population is fragmented
    
    # Adaptive learning rate based on population state
    if fragmentation > 0.7:
        # Fragmented population: explore more, use higher learning rate
        learn_rate = 0.15
    elif fragmentation < 0.3:
        # Well-connected population: exploit, use lower learning rate
        learn_rate = 0.05
    else:
        learn_rate = 0.10
    
    # Update strategy scores using topology-weighted credit
    for s in range(len(self.strategy_scores)):
        mask = strategy_used == s
        if np.any(mask):
            # Weight by topology influence of improving individuals
            topology_weights = topology_weight[mask & improved]
            if len(topology_weights) > 0:
                weighted_success = np.mean(topology_weights)
            else:
                weighted_success = 0.0
            
            # Also compute raw success rate for fallback
            raw_success = np.mean(improved[mask])
            
            # Blend weighted and raw success
            combined = 0.7 * weighted_success + 0.3 * raw_success
            
            # Exponential moving average update
            self.strategy_scores[s] = 0.85 * self.strategy_scores[s] + learn_rate * combined
    
    # Apply softmax-like normalization to keep scores in reasonable range
    self.strategy_scores = np.clip(self.strategy_scores, -5.0, 5.0)
```