**Idea: Gaussian Entropy Collapse Restart**
Use differential entropy of a fitted Gaussian to detect when the population distribution has degenerated, triggering restart when entropy falls below a dimension-scaled threshold.

```python
def _restart_if_needed(self, population, fitness):
    """
    Restart detection using information-theoretic measures on population distribution.
    
    Category C: Information-theoretic / distributional
    Fits a multivariate Gaussian to the population and monitors:
    - Differential entropy: detects distribution collapse
    - Condition number: detects elongated/needle-like distributions
    - Eigenvalue entropy: detects loss of multidimensional spread
    """
    NP, dim = population.shape
    
    # Fit multivariate Gaussian to population
    mean = np.mean(population, axis=0)
    cov = np.cov(population, rowvar=False)
    
    # Regularize for numerical stability and positive definiteness
    eps = 1e-10
    cov_reg = cov + eps * np.eye(dim)
    
    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov_reg)
    eigenvalues = np.maximum(eigenvalues, eps)
    
    # Condition number: high values indicate elongated/collapsed distribution
    cond_number = eigenvalues.max() / eigenvalues.min()
    
    # Differential entropy of Gaussian: H = 0.5 * log(det(2*pi*e*cov))
    # Low entropy = population collapsed into tight region
    log_det = np.sum(np.log(eigenvalues))
    differential_entropy = 0.5 * (dim * np.log(2 * np.pi * np.e) + log_det)
    
    # Normalized eigenvalue entropy: measures how spread the variance is
    # across dimensions. Low value = variance concentrated in few directions
    total_var = np.sum(eigenvalues) + eps
    normalized_eigenvalues = eigenvalues / total_var
    normalized_entropy = -np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues + eps))
    max_entropy = np.log(dim)
    eigenvalue_entropy_ratio = normalized_entropy / (max_entropy + eps)
    
    # Thresholds (dimension-adaptive)
    COND_THRESHOLD = 1e4 * dim
    ENTROPY_MIN = -5.0 * dim
    EIGENVALUE_ENTROPY_MIN = 0.15
    
    should_restart = (
        cond_number > COND_THRESHOLD or
        differential_entropy < ENTROPY_MIN or
        eigenvalue_entropy_ratio < EIGENVALUE_ENTROPY_MIN
    )
    
    if should_restart:
        new_pop = self._initialize_population()
        new_fitness = np.array([np.nan] * self.NP)
        self.stagnation_counter = 0
        self.archive = []
        return new_pop, new_fitness
    
    return population, fitness
```