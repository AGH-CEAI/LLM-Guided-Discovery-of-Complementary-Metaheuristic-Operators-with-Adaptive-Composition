**Idea: Temporal Momentum Mutation with EMA Stagnation Detection**

Category F: Temporal / dynamical. This approach tracks the population's centroid drift and improvement velocity using exponential moving averages (EMAs) across generations to detect stagnation, then triggers aggressive exploration mode with amplified scaling and directional perturbation when the population stops moving.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using temporal momentum dynamics:
        - Track centroid drift and improvement velocity via EMA
        - Detect stagnation from population movement patterns
        - Adapt scaling and exploration intensity based on temporal signals
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        # --- Temporal signal computation via EMA ---
        current_centroid = np.mean(population, axis=0)
        
        if not hasattr(self, 'ema_centroid'):
            self.ema_centroid = current_centroid.copy()
            self.ema_movement = 0.0
            self.ema_improvement = 0.0
            self.prev_centroid = current_centroid.copy()
        
        # Compute raw movement and improvement signals
        raw_movement = np.linalg.norm(current_centroid - self.prev_centroid)
        current_best = np.min(fitness)
        if not hasattr(self, 'prev_best_fitness'):
            self.prev_best_fitness = current_best
        raw_improvement = max(0.0, self.prev_best_fitness - current_best)
        
        # Update EMAs with different smoothing
        alpha_movement = 0.2
        alpha_improvement = 0.15
        self.ema_movement = alpha_movement * raw_movement + (1 - alpha_movement) * self.ema_movement
        self.ema_improvement = alpha_improvement * raw_improvement + (1 - alpha_improvement) * self.ema_improvement
        
        # Store for next iteration
        self.prev_centroid = current_centroid.copy()
        self.prev_best_fitness = current_best
        
        # --- Stagnation detection from temporal signals ---
        movement_threshold = 1e-6 * self.dim
        improvement_threshold = 1e-10
        
        is_stagnant = (self.ema_movement < movement_threshold and 
                       self.ema_improvement < improvement_threshold)
        
        # --- Adaptive scaling based on temporal state ---
        if is_stagnant:
            # Exploration mode: amplify scaling significantly
            scale_factor = 2.0 + 0.5 * np.random.random()
            perturbation = np.random.randn(np_pop, self.dim) * self.dim * 0.5
        else:
            # Normal mode: scale based on movement momentum
            movement_ratio = np.clip(self.ema_movement / (movement_threshold + 1e-10), 0.1, 2.0)
            scale_factor = 1.0 / (movement_ratio + 0.1)
            perturbation = np.zeros((np_pop, self.dim))
        
        current_F = self.F * scale_factor
        current_F = np.clip(current_F, 0.1, 3.0)
        
        # --- Ring-based rand/1 mutation with temporal bias ---
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # --- Directional bias from centroid drift (temporal) ---
        if not hasattr(self, 'ema_centroid_drift'):
            self.ema_centroid_drift = np.zeros(self.dim)
        
        raw_drift = current_centroid - self.ema_centroid
        self.ema_centroid_drift = 0.3 * raw_drift + 0.7 * self.ema_centroid_drift
        self.ema_centroid = current_centroid.copy()
        
        # Bias direction: opposite to drift when stagnant, along drift otherwise
        if is_stagnant:
            drift_bias = -0.5 * self.ema_centroid_drift
        else:
            drift_bias = 0.2 * self.ema_centroid_drift
        
        drift_bias = np.tile(drift_bias, (np_pop, 1))
        
        # --- Combine components ---
        blend_weight = 0.15 if is_stagnant else 0.25
        trials = (1 - blend_weight) * (base_mutation + drift_bias) + blend_weight * perturbation
        
        return trials, current_F
```