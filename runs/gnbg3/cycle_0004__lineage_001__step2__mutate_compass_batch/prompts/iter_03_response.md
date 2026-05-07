**Idea: GMM Information Gain Compass**
Category C: Uses Gaussian Mixture Model to treat the population as a probability distribution, then selects mutation directions based on expected log-likelihood improvement (information gain).

```python
def _mutate_compass_batch(self, population, fitness):
    """Information-theoretic compass mutation via Gaussian distribution fitting.
    
    Treats population as samples from a probability distribution. Fits a Gaussian
    to characterise the population density, then selects mutation directions by
    expected log-likelihood improvement (information gain per direction).
    """
    NP, dim = population.shape
    
    # Compute population statistics for Gaussian fit
    mean = population.mean(axis=0)
    
    # Fit Gaussian: compute covariance with regularization for numerical stability
    cov = np.cov(population, rowvar=False)
    cov_reg = cov + np.eye(dim) * 1e-6  # Ensure positive definite
    
    # Precompute precision matrix (inverse covariance) via eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        precision = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
        use_mahalanobis = True
    except np.linalg.LinAlgError:
        use_mahalanobis = False
    
    # Helper: compute negative log-likelihood under fitted Gaussian
    def neg_log_likelihood(x):
        if use_mahalanobis:
            diff = x - mean
            mahal_sq = np.sum(diff @ precision * diff, axis=1)
            log_det = np.sum(np.log(eigenvalues))
            return 0.5 * (mahal_sq + dim * np.log(2 * np.pi) + log_det)
        else:
            return np.sum((x - mean) ** 2, axis=1)
    
    # Compute log-likelihood for all population members
    nll_current = neg_log_likelihood(population)
    
    # Get fitness ranks for secondary scoring
    fitness_ranks = self._compute_fitness_ranking(fitness)
    best_idx = np.argmin(fitness)
    
    mutants = np.empty_like(population)
    
    for i in range(NP):
        i_rank = fitness_ranks[i]
        
        # Sample 3 random others for candidate construction
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r = others[:4]
        
        # Candidate directions with information-theoretic scoring
        candidates = []
        
        # Direction A: Move toward population MODE (density exploitation)
        # High density region = low NLL = expected fitness improvement signal
        dA = mean + self.F * (population[r[0]] - population[r[1]])
        nll_A = neg_log_likelihood(np.atleast_2d(dA))
        # Information gain: how much lower NLL (more "typical" of population)
        info_gain_A = nll_current[i] - nll_A[0]
        # Blend with fitness improvement potential
        score_A = 0.6 * info_gain_A + 0.4 * (i_rank - fitness_ranks[r].min())
        
        # Direction B: Move away from MODE (density exploration)
        # Jump toward sparse regions = high NLL = exploration signal
        dB = population[i] + self.F * (population[i] - mean) + self.F * (population[r[2]] - population[r[3]])
        nll_B = neg_log_likelihood(np.atleast_2d(dB))
        # Negative info gain (we WANT high NLL for exploration)
        info_gain_B = nll_B[0] - nll_current[i]
        score_B = 0.6 * info_gain_B + 0.4 * (i_rank - fitness_ranks[r].min())
        
        # Direction C: Toward best individual with distribution perturbation
        # Combines greedy direction with Gaussian noise calibrated to population spread
        noise_scale = np.sqrt(np.diag(cov_reg)).mean() * 0.1
        dC = population[best_idx] + self.F * (population[r[0]] - population[r[1]]) + np.random.randn(dim) * noise_scale
        nll_C = neg_log_likelihood(np.atleast_2d(dC))
        info_gain_C = nll_current[i] - nll_C[0]
        score_C = 0.6 * info_gain_C + 0.4 * (i_rank - fitness_ranks[best_idx])
        
        # Select direction with maximum information-theoretic score
        best_dir = max([(dA, score_A), (dB, score_B), (dC, score_C)], key=lambda x: x[1])[0]
        mutants[i] = best_dir
    
    return mutants
```