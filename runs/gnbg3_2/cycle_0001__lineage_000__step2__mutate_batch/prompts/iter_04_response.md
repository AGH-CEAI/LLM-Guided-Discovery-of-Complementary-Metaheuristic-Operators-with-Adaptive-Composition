**Idea: Rank-Based Adaptive Mutation with Fitness Percentile Scaling**

Using fitness percentile ranks to modulate mutation strength and select mutation sources based on rank-correlated success history, targeting difficult multi-modal landscapes where rank-order information reveals exploitable structure.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Rank-based mutation using fitness percentiles and rank-correlated success history.
        Category D: Only fitness signals, ranks, and percentiles — no raw distances.
        """
        np_pop, dim = len(population), self.dim
        
        # Compute fitness percentile rank (0=best, 1=worst) — pure fitness signal
        fitness_ranks = np.argsort(np.argsort(fitness)) / max(np_pop - 1, 1)
        
        # Rank-based adaptive scaling: top performers get tighter F, bottom get wider
        F_exploit = np.clip(self.F * 0.7, 0.3, 0.9)
        F_explore = np.clip(self.F * 1.3, 0.6, 1.5)
        F_percentile = F_exploit + fitness_ranks * (F_explore - F_exploit)
        
        # Rank-weighted centroid (fitness-only weighting, no distances)
        rank_weights = 1.0 / (fitness_ranks * np_pop + 1.0)
        rank_weights = rank_weights / np.sum(rank_weights)
        rank_centroid = np.sum(population * rank_weights[:, np.newaxis], axis=0)
        
        # Rank-correlated directional bias from success history
        if len(self.successful_directions) > 0:
            direction_stack = np.array(self.successful_directions)
            recency_weights = np.exp(-0.3 * np.arange(len(direction_stack) - 1, -1, -1))
            recency_weights = recency_weights / np.sum(recency_weights)
            rank_direction = np.sum(direction_stack * recency_weights[:, np.newaxis], axis=0)
        else:
            rank_direction = np.zeros(dim)
        
        # Select mutation sources by rank: top performers as r1, diverse ranks as r2
        top_idx = np.argmin(fitness)
        sorted_indices = np.argsort(fitness)
        r1_indices = sorted_indices[np.clip(
            (fitness_ranks * (np_pop - 1)).astype(int) - 1, 0, np_pop - 1
        )]
        r2_indices = sorted_indices[np.clip(
            (fitness_ranks * (np_pop - 1)).astype(int) + 1, 0, np_pop - 1
        )]
        
        # Rank-based mutation vectors
        r1_vectors = population[r1_indices]
        r2_vectors = population[r2_indices]
        
        # Differential with rank-adaptive F
        base_mutation = r1_vectors + F_percentile[:, np.newaxis] * (r2_vectors - population)
        
        # Blend components: rank determines exploitation vs exploration balance
        blend_to_best = 0.15 * (1.0 - fitness_ranks)  # Top performers lean toward best
        blend_to_centroid = 0.2 * fitness_ranks        # Bottom performers lean toward centroid
        
        trials = (
            (1 - blend_to_best - blend_to_centroid)[:, np.newaxis] * base_mutation
            + blend_to_best[:, np.newaxis] * population[top_idx]
            + blend_to_centroid[:, np.newaxis] * rank_centroid
        )
        
        # Add rank-directional bias
        trials = trials + 0.15 * rank_direction
        
        return trials, F_percentile.mean()
```