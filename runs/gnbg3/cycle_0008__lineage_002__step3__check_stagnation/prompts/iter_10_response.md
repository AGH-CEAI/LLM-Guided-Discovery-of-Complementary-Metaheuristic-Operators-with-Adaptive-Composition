Looking at the task, I need to implement a spectral/linear-algebraic stagnation detection for `_check_stagnation`. The previous Category B variant (4.0923e+02) performed poorly on task 17, so I need a fundamentally different approach.

My strategy: Use **effective rank** and **eigenvalue distribution entropy** to detect when the population collapses to a low-dimensional subspace, combined with condition number analysis. This directly measures anisotropy and effective dimensionality.

**Idea: Effective Rank + Eigenvalue Entropy Stagnation Detection**

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation via spectral analysis of population covariance.
    
    Uses effective rank and eigenvalue entropy to detect when population
    collapses to low-dimensional subspace (spectral stagnation).
    """
    if not hasattr(self, '_prev_best_fitness'):
        self._prev_best_fitness = np.inf
        self._fitness_history = []
        self._spectral_history = []
    
    # Track fitness improvement
    self._fitness_history.append(best_fitness)
    if len(self._fitness_history) > 15:
        self._fitness_history.pop(0)
    
    # Compute fitness stagnation
    fitness_improved = best_fitness < self._prev_best_fitness * 0.9999
    self._prev_best_fitness = best_fitness
    
    if len(self._fitness_history) >= 5:
        recent_range = max(self._fitness_history[-5:]) - min(self._fitness_history[-5:])
        fitness_stagnant = recent_range < 1e-8 * max(1.0, abs(best_fitness))
    else:
        fitness_stagnant = False
    
    # Get population from optimizer state
    if not hasattr(self, '_current_population') or self._current_population is None:
        return fitness_stagnant
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < dim + 1:
        return fitness_stagnant
    
    # Center the population
    centroid = pop.mean(axis=0)
    centered = pop - centroid
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Handle near-singular covariance
    reg = 1e-10 * np.eye(dim)
    cov_reg = cov + reg
    
    # Eigendecomposition
    try:
        eigenvals = np.linalg.eigvalsh(cov_reg)
        eigenvals = np.maximum(eigenvals, 1e-12)
        eigenvals = eigenvals[::-1]  # Sort descending
    except np.linalg.LinAlgError:
        return fitness_stagnant
    
    # Compute normalized eigenvalues (probability distribution)
    eigenvals_sum = eigenvals.sum()
    if eigenvals_sum <= 0:
        return fitness_stagnant
    p = eigenvals / eigenvals_sum
    
    # Effective rank via eigenvalue entropy
    p_safe = np.maximum(p, 1e-12)
    entropy = -np.sum(p_safe * np.log(p_safe))
    max_entropy = np.log(dim)
    effective_rank = np.exp(entropy) if max_entropy > 0 else 0
    effective_rank_normalized = effective_rank / dim
    
    # Eigenvalue concentration ratio (top-k to total)
    k = max(1, dim // 4)
    top_k_ratio = np.sum(eigenvals[:k]) / eigenvals_sum
    
    # Condition number (spectral condition number)
    cond_number = eigenvals[-1] / (eigenvals[0] + 1e-12) if eigenvals[0] > 0 else np.inf
    cond_log = np.log10(cond_number + 1.0)
    
    # Store spectral metrics
    spectral_metrics = {
        'effective_rank_norm': effective_rank_normalized,
        'top_k_ratio': top_k_ratio,
        'cond_log': cond_log,
        'eigenvals': eigenvals.copy()
    }
    self._spectral_history.append(spectral_metrics)
    if len(self._spectral_history) > 10:
        self._spectral_history.pop(0)
    
    # Stagnation detection via spectral signals
    # Case 1: Effective rank collapsed (population in low-dim subspace)
    rank_stagnant = effective_rank_normalized < 0.25
    
    # Case 2: Eigenvalue concentration (one dominant direction)
    concentration_stagnant = top_k_ratio > 0.85
    
    # Case 3: Condition number indicates ill-conditioning
    cond_stagnant = cond_log > 5.0
    
    # Case 4: Effective rank decreasing trend
    rank_trend_stagnant = False
    if len(self._spectral_history) >= 3:
        rank_trends = [m['effective_rank_norm'] for m in self._spectral_history[-3:]]
        if rank_trends[-1] < rank_trends[0] * 0.7:
            rank_trend_stagnant = True
    
    # Combine signals
    spectral_stagnant = (rank_stagnant and concentration_stagnant) or \
                        (rank_stagnant and cond_stagnant) or \
                        rank_trend_stagnant
    
    # Update stagnation counter
    if spectral_stagnant or fitness_stagnant:
        self.stagnation_counter += 1
    else:
        self.stagnation_counter = max(0, self.stagnation_counter - 1)
    
    # Trigger restart if stagnant
    return self.stagnation_counter >= 3
```