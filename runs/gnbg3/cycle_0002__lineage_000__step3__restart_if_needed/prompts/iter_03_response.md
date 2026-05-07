**Idea: Entropy-Collapse Restart**

Uses differential entropy of the population's multivariate Gaussian fit to detect when the distribution has collapsed. If current entropy falls below a threshold relative to initial entropy, performs a diversified restart by sampling from a mixture of the best solution, archive, and fresh uniform exploration.

```python
def _restart_if_needed(self, population, fitness):
    """
    Information-theoretic restart via entropy collapse detection.
    
    Fits a multivariate Gaussian to the population and monitors its
    differential entropy. When entropy drops below a fraction of the
    initial entropy, the population has collapsed and a diversified
    restart is triggered. New individuals are sampled from a mixture
    of: (1) the best solution with perturbations, (2) archive samples,
    (3) uniform random initialization.
    """
    # Compute current population covariance (information-theoretic core)
    current_mean = np.mean(population, axis=0)
    current_cov = np.cov(population, rowvar=False)
    
    # Regularize for numerical stability
    eps = 1e-10
    if current_cov.ndim == 0:
        current_cov = np.array([[current_cov]])
    current_cov = current_cov + eps * np.eye(self.dim)
    
    # Differential entropy of d-dimensional Gaussian: H = 0.5*ln((2*pi*e)^d * |Sigma|)
    try:
        current_det = np.linalg.det(current_cov)
        current_det = max(current_det, eps)
        current_entropy = 0.5 * np.log(current_det) + 0.5 * self.dim * (1 + np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        current_entropy = -np.inf
    
    # Initialize reference entropy on first call
    if not hasattr(self, '_ref_entropy'):
        self._ref_entropy = current_entropy
        self._ref_mean = current_mean.copy()
        return (population, fitness, None)
    
    # Entropy collapse detection: if current entropy is too low relative to reference
    entropy_ratio = current_entropy / max(self._ref_entropy, eps)
    
    # Also check KL divergence from initial distribution for robustness
    # KL(P||Q) = 0.5*(tr(Sigma_Q^-1 Sigma_P) + delta^T Sigma_Q^-1 delta - d + ln(|Sigma_Q|/|Sigma_P|))
    try:
        ref_cov_inv = np.linalg.inv(self._ref_cov if hasattr(self, '_ref_cov') else current_cov + eps * np.eye(self.dim))
        if not hasattr(self, '_ref_cov'):
            self._ref_cov = current_cov.copy()
        delta = self._ref_mean - current_mean
        kl_div = 0.5 * (
            np.trace(ref_cov_inv @ current_cov) +
            delta.T @ ref_cov_inv @ delta -
            self.dim +
            np.log(max(np.linalg.det(self._ref_cov), eps) / current_det)
        )
        kl_div = max(kl_div, 0.0)
    except np.linalg.LinAlgError:
        kl_div = 0.0
    
    # Trigger restart if entropy collapsed OR KL divergence is low (distribution hasn't moved much)
    should_restart = (entropy_ratio < 0.6) or (kl_div < 0.05 and entropy_ratio < 0.8)
    
    if not should_restart:
        return (population, fitness, None)
    
    # Diversified restart: mixture of best, archive, and random samples
    new_population = np.empty((self.NP, self.dim))
    bounds = (-100.0, 100.0)
    
    # Best individual preserved with small perturbation
    best_idx = np.argmin(fitness)
    best = population[best_idx].copy()
    new_population[0] = best + np.random.randn(self.dim) * 0.5
    new_population[0] = np.clip(new_population[0], bounds[0], bounds[1])
    
    # Archive injection: sample from good historical solutions
    archive_samples = []
    if len(self.archive) > 0:
        archive_concat = np.vstack(self.archive)
        if len(archive_concat) > 0:
            n_archive = min(len(archive_concat), self.NP // 3)
            indices = np.random.choice(len(archive_concat), n_archive, replace=False)
            archive_samples = archive_concat[indices]
    
    # Fill remaining spots with mixture strategy
    for i in range(1, self.NP):
        r = np.random.rand()
        if r < 0.3 and len(archive_samples) > 0:
            # Sample from archive with perturbation
            donor = archive_samples[np.random.randint(len(archive_samples))]
            new_population[i] = donor + np.random.randn(self.dim) * 15.0
        elif r < 0.6:
            # Explore around best with larger radius
            new_population[i] = best + np.random.randn(self.dim) * 20.0
        else:
            # Fresh uniform initialization
            new_population[i] = np.random.uniform(bounds[0], bounds[1], self.dim)
        new_population[i] = np.clip(new_population[i], bounds[0], bounds[1])
    
    # Re-evaluate fitness
    new_fitness = func(new_population)
    if len(new_fitness) < self.NP:
        new_fitness = np.full(self.NP, np.nan)
    
    # Reset diversity tracking
    self.stagnation_counter = 0
    self.archive = []  # Clear archive after restart
    
    return (new_population, new_fitness, best_idx)
```