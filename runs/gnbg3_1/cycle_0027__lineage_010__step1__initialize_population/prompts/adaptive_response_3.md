```python
import numpy as np


class AdaptiveInitEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best initialization
    AND covariance adaptation strategy during optimization.
    
    Features:
    - 9 initialization operators (original + 8 variants) with Thompson Sampling
    - 5 covariance adaptation operators with Thompson Sampling
    - Sliding window credit assignment for both
    - Automatic restart on stagnation with restart archive
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
