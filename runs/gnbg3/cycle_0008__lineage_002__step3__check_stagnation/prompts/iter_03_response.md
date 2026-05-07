**Idea: Entropy-KL Divergence Stagnation Detector**
Detect stagnation by treating the population as a probability distribution and monitoring entropy decay and KL divergence from the previous generation — category C information-theoretic reasoning.

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation via information-theoretic analysis of population distribution.
    
    Treats the population as samples from a multivariate Gaussian distribution and
    monitors:
    1. Entropy: lower entropy = population collapsed = stagnation
    2. KL divergence: measures distributional change between generations
    3. Eigenvalue spread: degenerate covariance = exploration limited
    """
    if not hasattr(self, '_prev_population'):
        self._prev_population = None
        self._entropy_history = []
        self._kl_history = []
        self._stagnation_counter = 0
        return False
    
    pop = self._current_population
    if pop is None or len(pop) < 4:
        return False
    
    NP, dim = pop.shape
    
    # Compute population Gaussian parameters
    mean = pop.mean(axis=0)
    
    # Covariance with regularization for numerical stability
    cov = np.cov(pop, rowvar=False)
    cov += np.eye(dim) * 1e-8
    
    # Compute entropy: H = 0.5 * log((2πe)^d * |Σ|)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        entropy = 0.0
    else:
        entropy = 0.5 * (dim * np.log(2 * np.pi * np.e) + logdet)
    
    self._entropy_history.append(entropy)
    if len(self._entropy_history) > 15:
        self._entropy_history.pop(0)
    
    # Compute KL divergence from previous population (symmetrized)
    kl_div = 0.0
    if self._prev_population is not None and len(self._prev_population) >= 4:
        prev_mean = self._prev_population.mean(axis=0)
        prev_cov = np.cov(self._prev_population, rowvar=False)
        prev_cov += np.eye(dim) * 1e-8
        
        # Symmetrized KL: (KL(p||q) + KL(q||p)) / 2
        # For Gaussians, KL(p||q) = 0.5 * (tr(Σq⁻¹Σp) + (μq-μp)ᵀΣq⁻¹(μq-μp) - d + ln(|Σq|/|Σp|))
        try:
            prev_cov_inv = np.linalg.inv(prev_cov)
            cov_inv = np.linalg.inv(cov)
            
            # KL from prev to current
            mean_diff = mean - prev_mean
            kl_forward = 0.5 * (
                np.trace(prev_cov_inv @ cov) +
                mean_diff @ prev_cov_inv @ mean_diff -
                dim +
                np.linalg.slogdet(cov)[1] - np.linalg.slogdet(prev_cov)[1]
            )
            
            # KL from current to prev
            kl_backward = 0.5 * (
                np.trace(cov_inv @ prev_cov) +
                mean_diff @ cov_inv @ mean_diff -
                dim +
                np.linalg.slogdet(prev_cov)[1] - np.linalg.slogdet(cov)[1]
            )
            
            kl_div = max(0.0, (kl_forward + kl_backward) / 2)
        except np.linalg.LinAlgError:
            kl_div = 0.0
    
    self._kl_history.append(kl_div)
    if len(self._kl_history) > 15:
        self._kl_history.pop(0)
    
    self._prev_population = pop.copy()
    
    # Analyze eigenvalue spread of covariance
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) >= 2:
            eig_ratio = eigenvalues.min() / (eigenvalues.max() + 1e-12)
        else:
            eig_ratio = 1.0
    except np.linalg.LinAlgError:
        eig_ratio = 1.0
    
    # Compute stagnation signals
    stagnation = False
    
    # Signal 1: Entropy collapsed (population concentrated)
    if len(self._entropy_history) >= 3:
        entropy_trend = self._entropy_history[-1] - self._entropy_history[-3]
        if entropy_trend < -0.05 * self._entropy_history[0] and entropy < self._entropy_history[0] * 0.5:
            stagnation = True
    
    # Signal 2: KL divergence too low (population not changing)
    if len(self._kl_history) >= 3:
        recent_kl_mean = np.mean(self._kl_history[-3:])
        if recent_kl_mean < 0.1:
            stagnation = True
    
    # Signal 3: Covariance degenerate (exploration limited)
    if eig_ratio < 1e-6:
        stagnation = True
    
    # Signal 4: Fitness stagnation combined with distribution stagnation
    if best_fitness >= self.prev_best_fitness - 1e-10:
        self._stagnation_counter += 1
    else:
        self._stagnation_counter = 0
    
    self.prev_best_fitness = best_fitness
    
    if stagnation or self._stagnation_counter > max(5, NP // 3):
        self._stagnation_counter = 0
        return True
    
    return False
```