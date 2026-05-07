**Idea: Axis-Aligned Spread Adaptation**
Adapt F and Cr based on population geometric spread (axis-aligned bounding box volume and per-dimension range), detecting convergence via spread ratio relative to historical EMA.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using population geometric spread (axis-aligned bounding box)."""
    # Compute axis-aligned bounding box
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    pop_range = pop_max - pop_min
    
    # Current geometric spread: sum of per-dimension ranges
    current_spread = np.sum(pop_range)
    
    # Initialize spread history with EMA on first call
    if not hasattr(self, 'spread_ewma'):
        self.spread_ewma = current_spread
        self.spread_history = [current_spread]
    
    # Update spread EMA
    alpha_spread = 0.2
    self.spread_ewma = alpha_spread * current_spread + (1 - alpha_spread) * self.spread_ewma
    
    # Track spread history
    self.spread_history.append(current_spread)
    if len(self.spread_history) > 20:
        self.spread_history.pop(0)
    
    # Compute spread ratio: current / historical EMA
    spread_ratio = current_spread / (self.spread_ewma + 1e-10)
    spread_ratio = np.clip(spread_ratio, 0.1, 10.0)
    
    # Per-dimension spread analysis
    dim_spreads = pop_range / (self.upper - self.lower + 1e-10)
    avg_dim_spread = np.mean(dim_spreads)
    
    # Detect convergence via spread ratio
    # Low ratio (< 0.8) → population clustering → increase exploration
    # High ratio (> 1.2) → population dispersing → increase exploitation
    converged = spread_ratio < 0.8
    dispersing = spread_ratio > 1.2
    
    if converged:
        # Population clustering: boost exploration with higher F, lower Cr
        F_adjustment = 1.15
        Cr_adjustment = 0.88
    elif dispersing:
        # Population dispersing: boost exploitation with lower F
        F_adjustment = 0.90
        Cr_adjustment = 1.05
    else:
        # Neutral regime: proportional adjustment based on spread_ratio distance from 1.0
        deviation = spread_ratio - 1.0
        F_adjustment = 1.0 - 0.10 * deviation
        Cr_adjustment = 1.0 + 0.05 * deviation
    
    # Apply adjustments
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```