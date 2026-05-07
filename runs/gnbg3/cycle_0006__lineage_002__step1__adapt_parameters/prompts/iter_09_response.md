**Idea: Geometric Diameter & k-NN Density Adaptation**

Category A: Geometry / spatial. Use population diameter (max pairwise distance), average k-NN distances, and centroid drift to adapt F and CR. Small diameter → increase F for exploration; dense clusters → increase F to escape; centroid drift → increase CR to propagate good directions.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR based on geometric layout of population."""
    pop = self._get_population()
    if pop is None or len(pop) < 4:
        return
    
    NP, dim = pop.shape
    
    # Geometric signal 1: population diameter (max pairwise distance)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, 0.0)
    diameter = np.sqrt(sq_dists.max())
    
    # Geometric signal 2: k-NN density (average distance to k nearest neighbors)
    k = max(2, min(5, NP // 4))
    sorted_dists = np.sort(sq_dists, axis=1)
    knn_dist = np.mean(np.sqrt(sorted_dists[:, 1:k+1]))  # exclude self (index 0 after fill_diagonal)
    
    # Geometric signal 3: centroid drift (temporal centroid displacement)
    centroid = pop.mean(axis=0)
    if not hasattr(self, '_prev_centroid'):
        self._prev_centroid = centroid.copy()
        self._centroid_drift = 0.0
    else:
        self._centroid_drift = np.linalg.norm(centroid - self._prev_centroid)
        self._prev_centroid = centroid.copy()
    
    # Initialize history for normalization
    if not hasattr(self, '_geo_diameter_history'):
        self._geo_diameter_history = []
        self._geo_knn_history = []
        self._geo_drift_history = []
    
    # Track history (bounded)
    self._geo_diameter_history.append(diameter)
    self._geo_knn_history.append(knn_dist)
    self._geo_drift_history.append(self._centroid_drift)
    if len(self._geo_diameter_history) > 20:
        self._geo_diameter_history.pop(0)
        self._geo_knn_history.pop(0)
        self._geo_drift_history.pop(0)
    
    # Compute normalized geometric signals using history
    n_hist = len(self._geo_diameter_history)
    if n_hist >= 3:
        # Diameter signal: relative to running mean
        d_mean = np.mean(self._geo_diameter_history)
        d_std = np.std(self._geo_diameter_history) + 1e-8
        d_signal = (diameter - d_mean) / d_std
        
        # k-NN density signal: relative to running mean
        knn_mean = np.mean(self._geo_knn_history)
        knn_std = np.std(self._geo_knn_history) + 1e-8
        knn_signal = (knn_dist - knn_mean) / knn_std
        
        # Centroid drift signal: relative to running mean
        drift_mean = np.mean(self._geo_drift_history)
        drift_std = np.std(self._geo_drift_history) + 1e-8
        drift_signal = (self._centroid_drift - drift_mean) / (drift_std + 1e-8)
    else:
        d_signal = 0.0
        knn_signal = 0.0
        drift_signal = 0.0
    
    # Map geometric signals to parameter adjustments
    # Small diameter (clustered) → increase F for exploration
    # Dense k-NN (local clustering) → increase F to escape
    # Large centroid drift → increase CR to propagate good directions
    f_adjustment = 0.1 * (-d_signal) + 0.1 * (-knn_signal) + 0.05 * drift_signal
    cr_adjustment = 0.1 * drift_signal + 0.05 * knn_signal
    
    # Apply adjustments with momentum
    if not hasattr(self, '_geo_f_momentum'):
        self._geo_f_momentum = 0.0
        self._geo_cr_momentum = 0.0
    
    momentum = 0.7
    self._geo_f_momentum = momentum * self._geo_f_momentum + (1 - momentum) * f_adjustment
    self._geo_cr_momentum = momentum * self._geo_cr_momentum + (1 - momentum) * cr_adjustment
    
    # Update parameters
    self.F = np.clip(self.F + 0.15 * self._geo_f_momentum, 0.3, 1.5)
    self.CR = np.clip(self.CR + 0.1 * self._geo_cr_momentum, 0.3, 0.98)
    
    # Store for history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```