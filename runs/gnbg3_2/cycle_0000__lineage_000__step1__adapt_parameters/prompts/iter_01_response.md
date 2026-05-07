**Idea: Geometric Spread-Diversity Adaptation**

Adapt F and Cr based on population spatial distribution: use centroid distances and axis-aligned spread to detect clustering (→ increase F) or balanced exploration (→ fine-tune Cr).

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr based on geometric/spatial properties of the population."""
    # Compute population centroid
    centroid = np.mean(self.population, axis=0)
    
    # Average distance from centroid as diversity measure
    distances = np.linalg.norm(self.population - centroid, axis=1)
    avg_distance = np.mean(distances)
    
    # Axis-aligned spread ratio (balance across dimensions)
    axis_std = np.std(self.population, axis=0)
    spread_ratio = np.min(axis_std) / (np.max(axis_std) + 1e-10)
    
    # Normalize spread to [0, 1] relative to search space
    search_space_size = self.upper - self.lower
    normalized_spread = avg_distance / (search_space_size + 1e-10)
    
    # Compute recent success rate
    success_rate = np.mean(improved_mask)
    
    self.F_success_history.append((success_rate, F_used))
    self.Cr_success_history.append((success_rate, Cr_used))
    
    if len(self.F_success_history) > self.adaptation_window:
        self.F_success_history.pop(0)
        self.Cr_success_history.pop(0)
    
    # Weighted average of recent success (exponential decay)
    if len(self.F_success_history) >= 3:
        weights = np.exp(-0.5 * np.arange(len(self.F_success_history))[::-1])
        weights = weights / np.sum(weights)
        recent_success = [s for s, _ in self.F_success_history]
        weighted_success = np.sum([w * s for w, s in zip(weights, recent_success)])
    else:
        weighted_success = success_rate
    
    # Adapt F: geometric spread drives exploration/exploitation
    # Low spread (clustered) + high success → reduce F (intensify)
    # High spread + low success → increase F (explore more)
    spread_factor = 1.0 - 2.0 * (normalized_spread - 0.5)
    success_factor = 1.0 - 2.0 * (weighted_success - 0.5)
    
    F_adjustment = 1.0 + 0.15 * (spread_factor + success_factor)
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    
    # Adapt Cr: axis spread ratio and success drive recombination
    # Balanced dimensions + high success → reduce Cr (preserve good solutions)
    # Imbalanced dimensions + low success → increase Cr (recombine more)
    Cr_adjustment = 1.0 + 0.1 * ((1.0 - spread_ratio) + success_factor)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```