**Idea: Eigenvalue Collapse Detector**
Uses eigendecomposition of the population covariance matrix to detect when the population collapses into a low-dimensional subspace, measuring effective rank, condition number, and spectral concentration.

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation via spectral analysis of population covariance.
    
    Uses eigenvalue decomposition to measure effective dimensionality,
    condition number, and spectral concentration of the population.
    Stagnation is detected when the population has collapsed to a 
    low-rank subspace or when the covariance structure indicates
    loss of exploration capability (Category B: Spectral/linear-algebraic).
    """
    # Need population from optimizer state
    if not hasattr(self, '_current_population') or self._current_population is None:
        return False
    
    pop = self._current_population
    NP, dim = pop.shape
    
    # Need sufficient samples for reliable covariance estimate
    if NP < dim or NP < 5:
        return False
    
    # Initialize spectral tracking
    if not hasattr(self, '_spectral_history'):
        self._spectral_history = []
        self._prev_eigenvalues = None
        self._prev_effective_rank = float(dim)
    
    # Center the population
    centroid = pop.mean(axis=0)
    centered_pop = pop - centroid
    
    # Compute covariance matrix with regularization
    cov = np.cov(centered_pop, rowvar=False)
    
    # Add small regularization for numerical stability
    eps = 1e-10
    cov_reg = cov + eps * np.eye(dim)
    
    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
    except np.linalg.LinAlgError:
        return False
    
    # Sort eigenvalues descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Remove near-zero eigenvalues (numerical artifacts)
    eigenvalues = eigenvalues[eigenvalues > eps]
    if len(eigenvalues) < 2:
        return True  # Severely collapsed
    
    # Normalize eigenvalues to probability distribution
    eig_sum = eigenvalues.sum()
    if eig_sum < eps:
        return True
    eig_normalized = eigenvalues / eig_sum
    
    # --- Spectral Metrics ---
    
    # 1. Effective rank via eigenvalue entropy
    # H = -sum(p * log(p)), normalized by max entropy log(dim)
    p_nonzero = eig_normalized[eig_normalized > eps]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero + eps))
    max_entropy = np.log(len(eigenvalues))
    effective_rank = (np.exp(entropy) if entropy > 0 else 1.0)
    
    # 2. Condition number (ratio of max to min eigenvalue)
    condition_number = eigenvalues[0] / max(eigenvalues[-1], eps)
    
    # 3. Dominance ratio: what fraction of variance is in top eigenvalue
    dominance_ratio = eigenvalues[0] / eig_sum
    
    # 4. Cumulative variance in top k dimensions
    k = max(1, int(0.1 * dim))  # Top 10% of dimensions
    top_k_eig = eigenvalues[:k]
    cumulative_variance_topk = top_k_eig.sum() / eig_sum
    
    # 5. Spectral spread (variance of normalized eigenvalues)
    spectral_spread = np.var(eig_normalized)
    
    # --- Stagnation Detection ---
    
    # Track changes
    rank_change = self._prev_effective_rank - effective_rank
    self._prev_effective_rank = effective_rank
    
    # Compute stagnation indicators
    rank_deficit = 1.0 - (effective_rank / dim)  # How much rank is lost
    condition_issue = np.log1p(condition_number) / 20.0  # Log-scaled condition
    dominance_issue = dominance_ratio  # Single eigenvalue dominance
    
    # Combined stagnation score
    stagnation_score = (
        0.4 * rank_deficit +
        0.3 * np.clip(condition_issue, 0, 1) +
        0.3 * dominance_issue
    )
    
    # Detect collapse: low effective rank OR high condition number
    is_stagnated = False
    
    # Criterion 1: Effective rank below threshold
    if effective_rank < 0.15 * dim:
        is_stagnated = True
    
    # Criterion 2: Severe condition number (ill-conditioned)
    if condition_number > 1e6:
        is_stagnated = True
    
    # Criterion 3: Single eigenvalue dominates
    if dominance_ratio > 0.95:
        is_stagnated = True
    
    # Criterion 4: Combined score threshold
    if stagnation_score > 0.75:
        is_stagnated = True
    
    # Criterion 5: Rapid rank collapse
    if rank_change > 0.5 * dim and effective_rank < 0.25 * dim:
        is_stagnated = True
    
    # Update history
    self._spectral_history.append({
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'dominance_ratio': dominance_ratio,
        'stagnation_score': stagnation_score
    })
    
    if len(self._spectral_history) > 20:
        self._spectral_history.pop(0)
    
    # Track previous eigenvalues
    self._prev_eigenvalues = eigenvalues.copy()
    
    return is_stagnated
```