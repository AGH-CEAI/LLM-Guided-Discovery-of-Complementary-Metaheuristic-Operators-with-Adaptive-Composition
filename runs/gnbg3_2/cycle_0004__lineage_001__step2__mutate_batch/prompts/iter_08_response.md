Looking at the worst unsolved tasks (17, 16, 11, 6, 5) with errors 10+ decades above target, these are likely highly multimodal or deceptive landscapes where the current single-strategy hybrid (ring + directional bias) fails to provide sufficient diversity or exploitation precision.

**Idea: Exploration-Exploitation Regime Switch with Data-Driven Weighting**

Combine DE/rand/1 (pure exploration) with a best+centroid strategy (pure exploitation), switching based on:
1. **Diversity ratio** (population spread vs initial spread) — when diversity is LOW, we need more exploration
2. **Success rate EMA** — when success is HIGH, we're in a good basin and should exploit more

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Hybrid mutation: switch between exploration (DE/rand/1) and 
        exploitation (best+centroid) based on diversity and success rate.
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        # Initialize regime tracking
        if not hasattr(self, 'exploration_ema'):
            self.exploration_ema = 0.8  # Start exploratory
            self.recent_success_rate = 0.5
        
        # Compute diversity ratio (current spread / initial spread)
        current_spread = np.std(population, axis=0)
        initial_spread = (self.upper - self.lower) / 3.0  # ~3-sigma coverage
        diversity_ratio = np.mean(current_spread / (initial_spread + 1e-10))
        diversity_ratio = np.clip(diversity_ratio, 0.01, 1.0)
        
        # Update success rate EMA
        if hasattr(self, 'last_improved_count'):
            current_success = self.last_improved_count / np_pop
            self.recent_success_rate = 0.3 * current_success + 0.7 * self.recent_success_rate
        
        # Regime detection: exploration vs exploitation weighting
        # High diversity + low success -> favor exploitation (intensify)
        # Low diversity + high success -> favor exploration (diversify)
        # High diversity + high success -> favor exploitation (ride the wave)
        # Low diversity + low success -> favor exploration (escape)
        
        diversity_score = diversity_ratio  # 0 = clustered, 1 = spread
        success_score = self.recent_success_rate  # 0 = failing, 1 = succeeding
        
        # Exploration weight: high when diversity is low OR when failing badly
        exploration_weight = 0.5 + 0.4 * (1.0 - diversity_score) + 0.2 * (0.5 - success_score)
        exploration_weight = np.clip(exploration_weight, 0.1, 0.9)
        
        # Smooth the exploration weight (regime shouldn't flip wildly)
        self.exploration_ema = 0.7 * self.exploration_ema + 0.3 * exploration_weight
        final_exploration_weight = self.exploration_ema
        
        # Strategy 1: DE/rand/1 (pure exploration)
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        F_explore = np.clip(self.F * (1.0 + 0.15 * np.random.randn()), 0.3, 2.0)
        explore_mutation = rand1_vectors + F_explore * (rand2_vectors - population)
        
        # Strategy 2: best + centroid direction (exploitation)
        best_idx = np.argmin(fitness)
        best_vector = population[best_idx]
        
        centroid = self._compute_population_centroid(population, fitness)
        centroid_direction = centroid - best_vector  # Direction from best toward centroid
        
        # Scale by distance to prevent overshooting
        centroid_dist = np.linalg.norm(centroid_direction) + 1e-10
        F_exploit = np.clip(0.5 * self.F, 0.1, 1.0)
        exploit_mutation = best_vector + F_exploit * centroid_direction
        
        # Broadcast exploitation vector to all individuals
        exploit_mutation = np.tile(exploit_mutation, (np_pop, 1))
        
        # Blend based on regime
        trials = final_exploration_weight * explore_mutation + (1.0 - final_exploration_weight) * exploit_mutation
        
        # Add small perturbation to prevent stagnation
        perturbation = 0.01 * (np.random.rand(np_pop, self.dim) - 0.5) * (self.upper - self.lower)
        trials = trials + perturbation
        
        return trials, F_explore
```