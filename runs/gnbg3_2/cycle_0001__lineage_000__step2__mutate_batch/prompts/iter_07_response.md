**Idea: Monte Carlo Bootstrap Direction Estimation**

This variant uses Monte Carlo sampling with bootstrap aggregation to stochastically estimate promising mutation directions for each individual. By sampling multiple random perturbations around each population member and evaluating their relative fitness, we build a statistically robust directional estimate that can escape local optima and navigate deceptive landscapes—particularly critical for the worst unsolved tasks (17, 16, 11, 19, 6) with errors exceeding 1e+02.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using Monte Carlo bootstrap direction estimation.
        For each individual, sample random perturbations, evaluate fitness,
        and estimate the improvement direction via weighted bootstrap aggregation.
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Monte Carlo parameters
        n_bootstrap = 8  # Number of bootstrap samples per individual
        perturbation_scale = 2.0 * self.F  # Scale for random perturbations
        
        # Ring-based parent indices for rand/1 component
        r1 = ring_prev
        r2 = ring_next
        
        # Standard DE rand/1 vectors
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Bootstrap direction estimation for each individual
        for i in range(np_pop):
            # Sample n_bootstrap random perturbation directions
            # Using random projections in different subspaces for diversity
            samples = np.zeros((n_bootstrap, dim))
            sample_fitness = np.zeros(n_bootstrap)
            
            for b in range(n_bootstrap):
                # Random sparse projection: mutate ~30% of dimensions
                mask = np.random.rand(dim) < 0.3
                perturbation = np.zeros(dim)
                perturbation[mask] = np.random.randn(np.sum(mask)) * perturbation_scale
                
                # Sample point
                samples[b] = population[i] + perturbation
                samples[b] = np.clip(samples[b], self.lower, self.upper)
            
            # Evaluate samples (vectorized for efficiency)
            sample_fitness = self._evaluate_batch(samples, lambda x: np.array([fitness[i]] * len(x)))
            
            # Compute improvement directions and weights
            delta_fitness = sample_fitness - fitness[i]
            
            # Only consider improving directions
            improving_mask = delta_fitness < 0
            
            if np.any(improving_mask):
                # Compute improvement directions
                improving_samples = samples[improving_mask]
                improving_deltas = improving_samples - population[i]
                improving_weights = -delta_fitness[improving_mask]
                
                # Normalize weights
                improving_weights = improving_weights / (np.sum(improving_weights) + 1e-10)
                
                # Weighted average direction
                bootstrap_direction = np.sum(improving_deltas * improving_weights[:, np.newaxis], axis=0)
                
                # Blend with base mutation
                blend_weight = 0.4
                trials[i] = (1 - blend_weight) * base_mutation[i] + blend_weight * (population[i] + bootstrap_direction)
            else:
                trials[i] = base_mutation[i]
        
        # Apply diversity-triggered perturbation for struggling populations
        if len(self.F_history) > 5:
            recent_success = np.mean(self.F_history[-5:])
            if recent_success < 0.15:
                # Add extra stochastic exploration when success rate is low
                extra_noise = np.random.randn(np_pop, dim) * 0.5
                trials = trials + extra_noise
        
        trials = np.clip(trials, self.lower, self.upper)
        return trials, current_F
```