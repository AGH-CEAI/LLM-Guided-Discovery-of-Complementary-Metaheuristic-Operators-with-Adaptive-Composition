**Idea: Eigenvalue-Driven Parameter Adaptation**
Use the condition number and effective dimensionality of the population covariance matrix (via eigendecomposition) to modulate F and CR. High condition number (anisotropy) → reduce F to avoid overshooting; high effective rank → reduce CR to preserve diverse exploration directions.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using spectral properties of the population covariance.
    
    Category B: Spectral / linear-algebraic
    Uses eigendecomposition of the population covariance to measure:
    - Condition number: anisotropy of the fitness landscape
    - Effective rank: number of independent exploration directions
    """
    # Compute population covariance and its eigendecomposition
    centroid = self.population.mean(axis=0)
    centered = self.population - centroid
    
    # Covariance with numerical stability
    cov = centered.T @ centered / self.NP
    
    # Add small regularization for numerical stability
    cov += np.eye(self.dim) * 1e-8
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        
        # Condition number: ratio of max to min eigenvalue (anisotropy measure)
        cond_num = eigenvalues.max() / eigenvalues.min()
        cond_factor = 1.0 / np.sqrt(np.clip(cond_num, 1.0, 1e6))
        
        # Effective rank: exponential of entropy of normalized eigenvalues
        eigenvalues_norm = eigenvalues / eigenvalues.sum()
        entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-12))
        effective_rank = np.exp(entropy)
        rank_factor = np.clip(effective_rank / self.dim, 0.1, 1.0)
        
        # Spectral-based F adaptation
        # High anisotropy (high cond_num) → reduce F to avoid overshooting
        target_F = 0.5 + 0.4 * cond_factor
        target_F = np.clip(target_F, 0.2, 1.2)
        
        # Spectral-based CR adaptation  
        # High effective rank → reduce CR to explore different dimensions
        target_CR = 0.95 - 0.4 * (1.0 - rank_factor)
        target_CR = np.clip(target_CR, 0.3, 0.98)
        
    except np.linalg.LinAlgError:
        # Fallback if eigendecomposition fails
        target_F = 0.6
        target_CR = 0.85
    
    # Smooth adaptation with momentum, biased by improvement rate
    momentum = 0.7
    improvement_weight = np.clip(improvement_rate, 0.0, 1.0)
    
    # If improving well, trust the spectral signal more
    # If stagnating, be more conservative
    alpha = momentum * (1.0 - 0.3 * improvement_weight)
    
    self.F = alpha * self.F + (1 - alpha) * target_F
    self.CR = alpha * self.CR + (1 - alpha) * target_CR
    
    # Clamp to valid ranges
    self.F = np.clip(self.F, 0.1, 1.5)
    self.CR = np.clip(self.CR, 0.1, 0.99)
    
    # Record history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```