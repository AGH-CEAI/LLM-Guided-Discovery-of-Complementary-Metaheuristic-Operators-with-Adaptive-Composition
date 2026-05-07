**Idea: Fitness-Rank Compression & FDC Inertia Modulation**

Adapt inertia using fitness-rank spread (detect local-optimum trapping via rank compression) combined with fitness-distance correlation (FDC) to detect deceptive landscapes where low FDC signals premature convergence. Category D: Fitness-landscape / rank-based.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using fitness-rank landscape analysis.
    
    Uses:
    1. Fitness-rank spread: detects when population clusters at similar fitness (local optimum trap)
    2. Fitness-distance correlation (FDC): detects deceptive landscapes where population
       clusters around local optima despite being far from global best
    3. Rank quartile analysis: identifies population compression state
    
    High inertia when ranks are diverse (exploring basin), low inertia when
    ranks compress (stuck in local optimum) or FDC is low (deceptive landscape).
    """
    # Compute fitness percentile ranks (0=worst, 1=best)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, len(self.current_fitness) - 1)
    
    # Fitness-rank spread: low spread = all individuals at similar fitness = trapped
    rank_spread = np.std(fitness_ranks) + 1e-10
    
    # Rank interquartile range: compressed IQR = population stuck
    q75, q25 = np.percentile(fitness_ranks, [75, 25])
    rank_iqr = q75 - q25 + 1e-10
    
    # Fitness-distance correlation (FDC): measures landscape structure
    # High FDC = fitness correlates with distance to global best (good exploration)
    # Low/negative FDC = deceptive landscape, population clustered away from global optimum
    if self.global_best is not None:
        distances = np.linalg.norm(self.population - self.global_best, axis=1)
        distance_ranks = np.argsort(np.argsort(distances)) / max(1, len(distances) - 1)
        
        # Pearson correlation on ranks (approximates Spearman)
        if np.std(distance_ranks) > 1e-10 and np.std(fitness_ranks) > 1e-10:
            fdc = np.corrcoef(fitness_ranks, distance_ranks)[0, 1]
        else:
            fdc = 0.0
    else:
        fdc = 0.0
    
    # Clamp FDC to reasonable range
    fdc = np.clip(fdc, -0.5, 1.0)
    
    # Determine landscape state from rank analysis
    # State 1: Rank compression (stuck in local optimum) -> aggressive exploration
    if rank_spread < 0.12 or rank_iqr < 0.15:
        # Population clustered at similar fitness: inject exploration
        self.inertia_weight = max(0.3, self.inertia_weight * 0.75)
    # State 2: Deceptive landscape (low FDC) -> moderate exploration boost
    elif fdc < 0.2:
        # Low fitness-distance correlation: population scattered in fitness space
        self.inertia_weight = max(0.4, self.inertia_weight * 0.85)
    # State 3: Rank diversity + good FDC -> increase exploitation
    elif rank_spread > 0.30 and rank_iqr > 0.35 and fdc > 0.5:
        # Population well-spread in fitness with good landscape structure
        self.inertia_weight = min(0.90, self.inertia_weight * 1.08)
    # State 4: Normal convergence
    else:
        # Gradual convergence schedule
        self.inertia_weight = 0.729 - 0.08 * (self.generation / 1000)
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    # Apply FDC modulation as multiplicative factor
    # Low FDC -> reduce inertia further (extra exploration pressure)
    fdc_modulation = 0.6 + 0.4 * np.clip(fdc + 0.5, 0.0, 1.0)
    self.inertia_weight *= fdc_modulation
    
    # Final bounds
    self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
```