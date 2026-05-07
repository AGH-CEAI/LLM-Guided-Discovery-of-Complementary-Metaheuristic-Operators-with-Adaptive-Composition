**Idea: Success-Rate × Diversity-Driven Hybrid F/CR Adaptation**

Combine (1) fitness-improvement-based parameter tuning with (2) population-spread-based tuning, switching weight based on measured diversity confidence. When population clusters tightly, geometry signal dominates to force exploration; when spread is healthy, success-rate signal dominates for exploitation.

```python
def _adapt_parameters(self, improvement_rate):
    """Hybrid F/CR adaptation: Success-rate + Diversity-driven with principled switching."""
    # Initialize temporal tracking
    if not hasattr(self, '_F_ema'):
        self._F_ema = self.F
        self._CR_ema = self.CR
        self._improvement_ema = 0.0
        self._diversity_ema = 0.0
        self._diversity_history = []
        self._iteration = 0
    
    self._iteration += 1
    
    # --- MECHANISM 1: Success-Rate Driven (Fitness-Landscape based) ---
    # Higher improvement rate -> encourage exploitation (lower F, more focused CR)
    alpha_success = 0.15
    self._improvement_ema = (1 - alpha_success) * self._improvement_ema + alpha_success * improvement_rate
    
    # Map improvement rate to target parameters via sigmoid for smooth transitions
    sig = 1.0 / (1.0 + np.exp(-10 * (self._improvement_ema - 0.3)))
    target_F_success = 0.9 - 0.5 * sig  # Range [0.4, 0.9]
    target_CR_success = 0.5 + 0.45 * sig  # Range [0.5, 0.95]
    
    # --- MECHANISM 2: Diversity-Driven (Geometry/Spatial based) ---
    # Compute population spread: average distance from centroid
    centroid = self.population.mean(axis=0)
    distances = np.linalg.norm(self.population - centroid, axis=1)
    current_diversity = distances.mean()
    
    # Update diversity EMA and bounded history
    alpha_div = 0.1
    self._diversity_ema = (1 - alpha_div) * self._diversity_ema + alpha_div * current_diversity
    self._diversity_history.append(current_diversity)
    if len(self._diversity_history) > 30:
        self._diversity_history.pop(0)
    
    # Diversity ratio: how does current spread compare to recent history?
    hist_mean = np.mean(self._diversity_history)
    hist_std = np.std(self._diversity_history) + 1e-8
    diversity_z = (current_diversity - hist_mean) / hist_std
    
    # Low diversity (negative z) -> encourage exploration (lower F, wider CR)
    target_F_diversity = np.clip(0.9 + 0.4 * diversity_z, 0.3, 0.9)
    target_CR_diversity = np.clip(0.5 - 0.25 * diversity_z, 0.5, 0.95)
    
    # --- PRINCIPLED SWITCHING: Weight by diversity signal confidence ---
    # Confidence is high when diversity is anomalous (far from historical mean)
    diversity_confidence = np.clip(abs(diversity_z) / 2.0, 0.0, 1.0)
    
    # When diversity is concerning (low or shrinking), trust geometry signal more
    # When diversity is normal, trust success-rate signal more
    if diversity_z < -0.5:
        # Diversity is significantly below average -> exploration mode
        w_diversity = 0.7
    elif diversity_z > 0.5:
        # Diversity is high -> exploitation mode
        w_diversity = 0.3
    else:
        # Transition zone -> blend based on confidence
        w_diversity = 0.5 - 0.2 * diversity_z
    
    w_success = 1.0 - w_diversity
    
    # --- HYBRID COMBINATION ---
    new_F = w_success * target_F_success + w_diversity * target_F_diversity
    new_CR = w_success * target_CR_success + w_diversity * target_CR_diversity
    
    # Apply momentum for temporal stability
    momentum = 0.7
    new_F = momentum * self._F_ema + (1 - momentum) * new_F
    new_CR = momentum * self._CR_ema + (1 - momentum) * new_CR
    
    # Clip to valid ranges
    self.F = np.clip(new_F, 0.3, 1.0)
    self.CR = np.clip(new_CR, 0.3, 0.98)
    
    # Store for next iteration
    self._F_ema = self.F
    self._CR_ema = self.CR
```