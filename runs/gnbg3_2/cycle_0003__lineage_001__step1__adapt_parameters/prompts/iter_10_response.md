Looking at the data, I need to understand why the worst tasks (17, 16, 11, 6, 5) are stuck at errors of 10² to 10⁵ — these suggest the population has likely collapsed into a degenerate subspace or is stuck in an ill-conditioned region of the fitness landscape.

The previous Category B variant (variant_02) crashed with -inf, so I need a numerically robust spectral approach. My angle: use the **condition number and effective rank of the population covariance matrix** to diagnose anisotropy and subspace collapse, then adapt F and Cr accordingly. High condition number → population is elongated/ill-conditioned → need more exploration; low-rank population → diversity collapse → need radical perturbation.

**Idea: Spectral Condition Number Adaptation**
Use eigenvalues of population covariance to compute condition number (κ = λ_max/λ_min) and effective dimensionality rank. Map spectral diagnostics to F and Cr adjustments.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using spectral properties of the population covariance matrix.
    
    Category B: Spectral / linear-algebraic
    Uses eigenvalue analysis to detect anisotropy and effective dimensionality collapse.
    """
    success_rate = np.mean(improved_mask)
    
    # Initialize spectral state
    if not hasattr(self, 'cond_history'):
        self.cond_history = []
        self.rank_history = []
        self.stagnation_counter = 0
    
    # Compute population covariance matrix (np x dim)
    pop_centered = self.population - np.mean(self.population, axis=0)
    # Covariance over population -> dim x dim matrix
    cov_matrix = np.cov(pop_centered.T)
    
    # Compute eigenvalues via eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Clip small negative eigenvalues from numerical errors
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        
        # Condition number: ratio of largest to smallest positive eigenvalue
        pos_mask = eigenvalues > 1e-12 * np.max(eigenvalues)
        if np.sum(pos_mask) >= 2:
            cond_number = eigenvalues[pos_mask].max() / (eigenvalues[pos_mask].min() + 1e-12)
        else:
            cond_number = 1.0
        
        # Effective rank: count of significant eigenvalues
        effective_rank = np.sum(pos_mask)
        rank_ratio = effective_rank / max(self.dim, 1)
        
    except np.linalg.LinAlgError:
        # Fallback if eigendecomposition fails
        cond_number = 1.0
        rank_ratio = 1.0
        eigenvalues = np.array([1.0])
    
    # Track spectral history for stagnation detection
    self.cond_history.append(cond_number)
    self.rank_history.append(rank_ratio)
    if len(self.cond_history) > 20:
        self.cond_history.pop(0)
        self.rank_history.pop(0)
    
    # Detect spectral stagnation: persistently high condition number
    avg_cond = np.mean(self.cond_history)
    spectral_stagnation = len(self.cond_history) >= 10 and avg_cond > 50
    
    # Compute eigenvalue spread for additional insight
    if len(eigenvalues) > 1 and eigenvalues.max() > 1e-12:
        eig_spread = eigenvalues.max() / (np.median(eigenvalues[eigenvalues > 1e-12]) + 1e-12)
    else:
        eig_spread = 1.0
    
    # Map condition number to F adjustment (logarithmic scale)
    # High condition number -> anisotropic landscape -> increase F for exploration
    if cond_number > 500:
        F_adjustment = 1.35
    elif cond_number > 100:
        F_adjustment = 1.2
    elif cond_number > 30:
        F_adjustment = 1.08
    elif cond_number > 10:
        F_adjustment = 1.0
    elif cond_number > 3:
        F_adjustment = 0.95
    else:
        F_adjustment = 0.85
    
    # Map effective rank ratio to Cr adjustment
    # Low rank (collapsed population) -> need more exploration via lower Cr
    # High rank (diverse population) -> can afford higher recombination
    if rank_ratio < 0.2:
        Cr_adjustment = 0.75
    elif rank_ratio < 0.4:
        Cr_adjustment = 0.88
    elif rank_ratio < 0.7:
        Cr_adjustment = 0.95
    else:
        Cr_adjustment = 1.08
    
    # Apply eigenvalue spread bonus
    if eig_spread > 10:
        F_adjustment *= 1.1
        Cr_adjustment *= 0.9
    
    # Spectral stagnation override: force exploration mode
    if spectral_stagnation:
        F_adjustment = 0.7
        Cr_adjustment = 0.8
        self.stagnation_counter += 1
    else:
        # Update stagnation based on success rate
        if success_rate < 0.1:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
    
    # Apply adjustments with bounds
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```