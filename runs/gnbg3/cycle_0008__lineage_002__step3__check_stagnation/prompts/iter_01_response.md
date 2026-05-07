**Idea: Centroid Drift & Pairwise Regularity Stagnation**

Detect stagnation using two geometric signals: (1) centroid movement across generations (temporal) and (2) pairwise distance regularity (spatial uniformity). A collapsed or overly regular population triggers restart.

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation via geometric analysis: centroid drift + pairwise regularity."""
    if not hasattr(self, '_current_population') or self._current_population is None:
        return False
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < 4:
        return False
    
    # --- A: Geometry / spatial analysis ---
    
    # 1. Compute current centroid and pairwise distance statistics
    centroid = pop.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    
    # Pairwise distances (upper triangle)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    i, j = np.triu_indices(NP, k=1)
    dists = np.sqrt(sq_dists[i, j])
    
    median_d = np.median(dists)
    std_d = np.std(dists)
    
    # Regularity ratio: std/median (regular simplex → 0, random → higher)
    regularity = std_d / (median_d + 1e-10)
    
    # 2. Initialize or update centroid history (F: temporal tracking)
    if not hasattr(self, '_centroid_history'):
        self._centroid_history = []
        self._dist_regularity_history = []
    
    self._centroid_history.append(centroid_norm)
    self._dist_regularity_history.append(regularity)
    
    # Keep bounded history
    max_history = 15
    if len(self._centroid_history) > max_history:
        self._centroid_history.pop(0)
        self._dist_regularity_history.pop(0)
    
    # 3. Stagnation detection using geometric signals
    
    # Signal A: Centroid drift (temporal)
    if len(self._centroid_history) >= 5:
        recent = np.array(self._centroid_history[-5:])
        centroid_range = recent.max() - recent.min()
        # Population stagnant if centroid barely moved relative to search space
        search_scale = 200.0 * np.sqrt(dim)  # approximate search space scale
        centroid_stagnant = centroid_range < search_scale * 1e-3
    else:
        centroid_stagnant = False
    
    # Signal B: Pairwise regularity (spatial) — population too uniform
    regularity_stagnant = regularity < 0.05
    
    # Signal C: Pairwise spread collapsed (spatial) — population too tight
    spread_collapsed = median_d < 1e-4 * np.sqrt(dim)
    
    # Combined geometric stagnation
    geometric_stagnant = centroid_stagnant or regularity_stagnant or spread_collapsed
    
    if geometric_stagnant:
        self.stagnation_counter += 1
    else:
        self.stagnation_counter = 0
    
    # Clear history on restart trigger (prevents stale signals)
    if self.stagnation_counter >= 10:
        if hasattr(self, '_centroid_history'):
            self._centroid_history.clear()
        if hasattr(self, '_dist_regularity_history'):
            self._dist_regularity_history.clear()
        return True
    
    return False
```