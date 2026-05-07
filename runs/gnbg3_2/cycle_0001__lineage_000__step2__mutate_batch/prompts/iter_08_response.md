**Idea: Adaptive k-NN Diversity + Fitness Improvement Hybrid Mutation**

Category H: Hybrid / multi-strategy combining (A) k-NN centroid-based spatial exploration with (D) fitness improvement rate detection, using principled data-driven switching based on population diversity state.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Hybrid mutation combining:
        - k-NN spatial mechanism: centroid from k nearest better fitness members (Category A)
        - Fitness improvement mechanism: success-rate based perturbation (Category D)
        Switching is data-driven based on population diversity state.
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Initialize hybrid state tracking
        if not hasattr(self, 'hybrid_success_ewma'):
            self.hybrid_success_ewma = 0.5
            self.hybrid_diversity_ewma = 1.0
        
        # --- Mechanism 1: k-NN Spatial (Category A) ---
        # For each individual, find k nearest neighbors with better fitness
        k = min(5, np_pop - 1)
        sorted_indices = np.argsort(fitness)  # 0 = best
        knn_centroids = np.zeros((np_pop, dim))
        
        for i in range(np_pop):
            # Find k individuals with better fitness (earlier in sorted list)
            better_mask = np.isin(sorted_indices, np.arange(0, k + 1))
            better_indices = sorted_indices[:min(k + 1, np_pop)]
            better_indices = better_indices[better_indices != i]
            
            if len(better_indices) > 0:
                knn_neighbors = population[better_indices]
                knn_centroids[i] = np.mean(knn_neighbors, axis=0)
            else:
                knn_centroids[i] = np.mean(population, axis=0)
        
        # --- Mechanism 2: Fitness Improvement Rate (Category D) ---
        # Track improvement rate for switching decision
        current_best = np.min(fitness)
        if hasattr(self, 'prev_best_fitness'):
            improvement_occurred = current_best < self.prev_best_fitness
            improvement_signal = 1.0 if improvement_occurred else 0.0
            self.hybrid_success_ewma = 0.7 * self.hybrid_success_ewma + 0.3 * improvement_signal
        self.prev_best_fitness = current_best
        
        # --- Compute Population Diversity for Switching ---
        centroid = np.mean(population, axis=0)
        avg_dist = np.mean(np.linalg.norm(population - centroid, axis=1))
        self.hybrid_diversity_ewma = 0.9 * self.hybrid_diversity_ewma + 0.1 * avg_dist
        
        # Diversity-based switching threshold (principled: relative to search space)
        search_space_scale = self.upper - self.lower
        diversity_ratio = avg_dist / (search_space_scale + 1e-10)
        diversity_threshold = 0.05  # 5% of search space as low-diversity threshold
        
        # --- Principled Weighting/Switching Logic ---
        # If diversity is low OR improvement rate is low: favor spatial mechanism
        # If diversity is adequate AND improvement is occurring: balance both
        low_diversity = diversity_ratio < diversity_threshold
        low_improvement = self.hybrid_success_ewma < 0.3
        
        if low_diversity:
            # Exploration mode: prioritize spatial k-NN mechanism
            spatial_weight = 0.8
        elif low_improvement:
            # Stagnation mode: increase spatial weight for escape
            spatial_weight = 0.65
        else:
            # Normal mode: balanced hybrid
            spatial_weight = 0.5
        
        # --- Generate Mutation Vectors ---
        # Get ring-based parents
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F based on diversity
        diversity_factor = np.clip(avg_dist / 50.0, 0.5, 2.0)
        current_F = self.F * diversity_factor * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Mechanism 1: k-NN spatial mutation
        spatial_mutation = knn_centroids + current_F * (rand1_vectors - rand2_vectors)
        
        # Mechanism 2: Fitness-weighted centroid perturbation
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        fitness_centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        
        # Direction toward fitness centroid
        to_centroid = fitness_centroid - population
        fitness_mutation = population + current_F * to_centroid + 0.5 * (rand1_vectors - rand2_vectors)
        
        # --- Hybrid Combination ---
        trials = spatial_weight * spatial_mutation + (1 - spatial_weight) * fitness_mutation
        
        return trials, current_F
```