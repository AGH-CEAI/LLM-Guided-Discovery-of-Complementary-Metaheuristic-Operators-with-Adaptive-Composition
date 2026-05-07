Looking at the error patterns, I see that tasks 10-23 are stuck at errors of 10^1 to 10^2 — this indicates the algorithm is getting trapped in local optima and not exploring effectively. The current variants all use CMA-ES-style smooth covariance adaptation, which lacks explicit mechanisms to escape local optima.

**Idea: Stagnation-Triggered Eigenvalue Perturbation**
This variant detects stagnation by monitoring fitness plateaus, then injects diversity by perturbing the mean along minor eigenvectors of the covariance matrix (the directions of highest uncertainty). This forces exploration in underrepresented regions of the search space.
```python
def _adapt_covariance_variant_01(self):
    """Stagnation-triggered eigenvalue perturbation for escaping local optima."""
    # Detect stagnation by monitoring fitness improvement
    if not hasattr(self, 'stagnation_history'):
        self.stagnation_history = []
    
    self.stagnation_history.append(self.f_opt)
    if len(self.stagnation_history) > 20:
        self.stagnation_history.pop(0)
    
    # Compute stagnation metrics
    is_stagnant = False
    if len(self.stagnation_history) >= 20:
        recent_improvement = self.stagnation_history[0] - self.stagnation_history[-1]
        relative_improvement = recent_improvement / (abs(self.stagnation_history[0]) + 1e-10)
        is_stagnant = (relative_improvement < 0.01) or (self.stagnation_counter > self.max_stagnation // 2)
    
    # Evolution path update
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Stagnation-triggered perturbation along minor eigenvectors
    if is_stagnant:
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
            
            # Perturb along 3 minor eigenvectors (highest uncertainty directions)
            num_perturb = min(3, self.dim)
            for k in range(num_perturb):
                minor_dir = eigvecs[:, k]
                scale = np.sqrt(eigvals[k]) * self.sigma * 0.3
                perturbation = minor_dir * np.random.randn() * scale
                self.mean = self.mean + perturbation
            
            # Reset evolution path to encourage new direction
            self.pc *= 0.1
            
            # Inject fresh diversity into population
            for i in range(min(10, self.NP)):
                idx = self.NP - 1 - i
                random_dir = np.random.randn(self.dim)
                random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-15)
                self.population[idx] = self._clip_to_bounds(
                    self.mean + random_dir * np.random.uniform(0.5, 2.0) * self.sigma * np.sqrt(self.dim)
                )
        except np.linalg.LinAlgError:
            pass
    
    # Rank-one and rank-μ updates
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rate based on stagnation
    if is_stagnant:
        ccov_effective = min(self.ccov * 3.0, 0.5)
    else:
        ccov_effective = self.ccov
    
    self.C = ((1.0 - ccov_effective) * self.C + 
              ccov_effective * (rank_one + (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```