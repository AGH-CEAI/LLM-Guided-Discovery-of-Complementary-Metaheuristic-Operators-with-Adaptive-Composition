**Idea: Centroid Drift Momentum**

Use exponential moving average (EMA) of population centroid position across generations to detect convergence drift. When centroid movement slows (indicating potential stagnation in a narrow basin), inject momentum from historical successful mutation vectors to break out. This temporal/geometric hybrid gives a fundamentally different signal than fitness-based adaptation.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using temporal centroid-drift dynamics.
        Track centroid position EMA across generations; when drift slows,
        apply momentum from historical successful mutations to escape stagnation.
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        # --- Temporal: Track centroid drift via EMA ---
        current_centroid = np.mean(population, axis=0)
        
        if not hasattr(self, 'centroid_ema'):
            self.centroid_ema = current_centroid.copy()
            self.centroid_ema_old = current_centroid.copy()
            self.centroid_drift_ema = 1.0
            self.momentum_vec = np.zeros(self.dim)
            self.drift_history = [1.0]
        
        # Update centroid EMA with momentum
        alpha_centroid = 0.2
        self.centroid_ema_old = self.centroid_ema.copy()
        self.centroid_ema = alpha_centroid * current_centroid + (1 - alpha_centroid) * self.centroid_ema
        
        # Compute drift magnitude (rate of centroid position change)
        centroid_delta = np.linalg.norm(self.centroid_ema - self.centroid_ema_old)
        
        # EMA of drift for smoothing
        alpha_drift = 0.1
        self.centroid_drift_ema = alpha_drift * (centroid_delta + 1e-10) + (1 - alpha_drift) * self.centroid_drift_ema
        
        # Track drift history for stagnation detection
        self.drift_history.append(centroid_delta)
        if len(self.drift_history) > 20:
            self.drift_history.pop(0)
        
        # --- Temporal: Stagnation detection via drift window ---
        drift_mean = np.mean(self.drift_history) if self.drift_history else 1e-6
        drift_stagnant = centroid_delta < 0.1 * drift_mean and len(self.drift_history) >= 10
        
        # --- Temporal: Momentum from successful mutation history ---
        if hasattr(self, 'successful_mutations_ema') and self.successful_mutations_ema is not None:
            momentum_magnitude = np.linalg.norm(self.momentum_vec)
            if momentum_magnitude > 1e-10:
                self.momentum_vec = self.momentum_vec / momentum_magnitude
        
        # --- Ring-based rand/1 base ---
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # --- Temporal: Blend based on drift regime ---
        if drift_stagnant:
            # Stagnation: boost momentum-driven exploration
            blend_weight = 0.5
            momentum_scale = 2.0 * self.centroid_drift_ema
            trials = (1 - blend_weight) * base_mutation + blend_weight * (self.momentum_vec * momentum_scale)
        else:
            # Normal: gentle momentum contribution
            blend_weight = 0.25
            momentum_scale = 0.5 * self.centroid_drift_ema
            trials = (1 - blend_weight) * base_mutation + blend_weight * (self.centroid_ema + self.momentum_vec * momentum_scale)
        
        return trials, current_F
```