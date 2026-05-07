**Idea: Entropy-Driven Inertia via Population Distribution Fitting**

This approach treats the population as a multivariate Gaussian distribution and modulates inertia weight based on Shannon entropy of the fitted distribution. High entropy → population is well-spread → higher inertia for exploration. Low entropy → population is clustered → lower inertia for exploitation. Entropy rate (change across generations) detects stagnation.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using information-theoretic entropy of population distribution.
    
    Treats the population as a multivariate Gaussian and computes its differential
    entropy. High entropy indicates exploration; low entropy indicates exploitation.
    Entropy rate across generations detects stagnation for adaptive reset.
    """
    # Fit Gaussian to population and compute differential entropy
    centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(centered.T)
    
    # Add small regularization for numerical stability
    cov_reg = cov + np.eye(self.dim) * 1e-8
    
    try:
        eigenvalues, _ = np.linalg.eigh(cov_reg)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        
        # Differential entropy of Gaussian: H = 0.5 * log((2πe)^d * det(Σ))
        # = 0.5 * d * log(2πe) + 0.5 * sum(log(λ_i))
        log_det = np.sum(np.log(eigenvalues))
        entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + log_det)
        
    except np.linalg.LinAlgError:
        entropy = 0.5 * self.dim  # Fallback to maximum entropy (identity covariance)
    
    # Track entropy history for stagnation detection
    if not hasattr(self, '_entropy_history'):
        self._entropy_history = []
    self._entropy_history.append(entropy)
    if len(self._entropy_history) > 50:
        self._entropy_history.pop(0)
    
    # Compute entropy rate (recent change) for stagnation detection
    if len(self._entropy_history) >= 10:
        recent_std = np.std(self._entropy_history[-10:])
        entropy_rate = recent_std / (np.mean(self._entropy_history[-10:]) + 1e-10)
    else:
        entropy_rate = 0.1
    
    # Normalize entropy to [0, 1] using approximate bounds
    # Max entropy: uniform distribution over search space
    search_range = self.upper_bound - self.lower_bound
    max_entropy = self.dim * np.log(search_range + 1e-10)
    # Min entropy: delta distribution
    min_entropy = self.dim * np.log(1e-6)
    
    entropy_norm = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0.0, 1.0)
    
    # Map normalized entropy to inertia weight
    # High entropy (exploration) → high inertia (0.9)
    # Low entropy (exploitation) → low inertia (0.4)
    target_inertia = 0.4 + 0.5 * entropy_norm
    
    # Smooth inertia update with momentum
    if not hasattr(self, '_prev_entropy_inertia'):
        self._prev_entropy_inertia = 0.729
    
    # Gradual convergence also factors in
    generation_decay = 0.05 * (self.generation / 1000)
    
    # Combine entropy-driven target with generation-based decay
    combined_target = target_inertia - generation_decay
    
    # Apply smoothing for stable transitions
    self.inertia_weight = 0.7 * self._prev_entropy_inertia + 0.3 * combined_target
    
    # Stagnation recovery: if entropy rate is very low, boost exploration
    if entropy_rate < 0.01 and len(self._entropy_history) >= 20:
        self.inertia_weight = min(0.95, self.inertia_weight * 1.1)
    
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    self._prev_entropy_inertia = self.inertia_weight
```