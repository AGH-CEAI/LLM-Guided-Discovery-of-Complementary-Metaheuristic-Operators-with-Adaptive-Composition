**Idea: Geometric-Temporal Hybrid Adaptation**

Category H: Hybrid / multi-strategy combining population geometric diversity (pairwise distances) with temporal success rate EMA. The weighting uses a sigmoid-based schedule driven by normalized diversity, creating principled switching between diversity-driven (exploration) and success-driven (exploitation) parameter control.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Hybrid adaptation: combine geometric diversity with temporal success signals."""
    success_rate = np.mean(improved_mask)
    np_pop, dim = self.np, self.dim

    # Initialize state
    if not hasattr(self, 'success_ewma'):
        self.success_ewma = success_rate
        self.F_history = [self.F]
        self.Cr_history = [self.Cr]
        self.diversity_history = []

    # Temporal mechanism: EMA of success rate
    alpha_temp = 0.2
    self.success_ewma = alpha_temp * success_rate + (1 - alpha_temp) * self.success_ewma

    # Geometric mechanism: population spread via average centroid distance
    centroid = np.mean(self.population, axis=0)
    centroid_distances = np.linalg.norm(self.population - centroid, axis=1)
    current_diversity = np.mean(centroid_distances)
    
    # Track diversity history for normalization
    self.diversity_history.append(current_diversity)
    if len(self.diversity_history) > 20:
        self.diversity_history.pop(0)
    
    # Normalize diversity to [0, 1] using running min-max
    d_min = min(self.diversity_history) + 1e-10
    d_max = max(self.diversity_history) + 1e-10
    norm_diversity = np.clip((current_diversity - d_min) / (d_max - d_min), 0.0, 1.0)

    # Principled weighting: sigmoid-based schedule on normalized diversity
    # High diversity -> weight toward success signals (exploitation)
    # Low diversity -> weight toward diversity signals (exploration)
    diversity_weight = 1.0 / (1.0 + np.exp(-10.0 * (norm_diversity - 0.4)))
    success_weight = 1.0 - diversity_weight

    # Compute geometric signal: larger spread -> increase F, decrease Cr
    geo_F_signal = 0.7 + 0.6 * norm_diversity
    geo_Cr_signal = 0.95 - 0.5 * norm_diversity

    # Compute temporal signal: higher success -> decrease F, increase Cr  
    temp_F_signal = 1.1 - 0.4 * self.success_ewma
    temp_Cr_signal = 0.9 + 0.2 * self.success_ewma

    # Hybrid combination via principled weighting
    F_signal = diversity_weight * geo_F_signal + success_weight * temp_F_signal
    Cr_signal = diversity_weight * geo_Cr_signal + success_weight * temp_Cr_signal

    # Apply adjustments
    self.F = np.clip(self.F * F_signal, 0.1, 1.9)
    self.Cr = np.clip(self.Cr * Cr_signal, 0.05, 0.95)

    # Track history for drift correction
    self.F_history.append(F_used)
    self.Cr_history.append(Cr_used)
    if len(self.F_history) > 15:
        self.F_history.pop(0)
        self.Cr_history.pop(0)
```