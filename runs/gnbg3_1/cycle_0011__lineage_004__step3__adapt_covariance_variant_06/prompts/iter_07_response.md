**Idea: Fitness Gradient-Directed Covariance Adaptation**

Instead of scaling learning rates by generic diversity metrics, this approach explicitly computes a fitness landscape gradient by tracking how each population member's fitness deviates from the mean, then uses the resulting gradient matrix to direct covariance adaptation along the most promising eigendirections. On stagnation, it injects directional perturbations aligned with the gradient rather than uniform noise.

```python
def _adapt_covariance_variant_06(self):
    """Fitness gradient-directed covariance adaptation with stagnation escape."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute fitness gradient matrix: captures which directions improve fitness
    fit_diff = self.fitness - np.mean(self.fitness)
    fit_diff = fit_diff / (np.std(fit_diff) + 1e-10)
    
    grad_matrix = np.zeros((self.dim, self.dim))
    for i in range(min(self.mu, len(fit_diff))):
        diff = (self.population[i] - self.old_mean) / self.sigma
        grad_matrix += fit_diff[i] * np.outer(diff, diff)
    
    # Eigendecomposition of covariance
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    
    # Project gradient onto eigendirections to find most promising axes
    grad_proj = np.zeros(self.dim)
    for j in range(self.dim):
        proj = 0.0
        for i in range(min(self.mu, len(fit_diff))):
            diff = (self.population[i] - self.old_mean) / self.sigma
            proj += fit_diff[i] * np.dot(eigvecs[:, j], diff)
        grad_proj[j] = proj
    
    # Compute directional weights: positive = improving direction
    dir_weights = np.zeros(self.dim)
    for j in range(self.dim):
        if eigvals[j] > 1e-10:
            dir_weights[j] = np.tanh(grad_proj[j] / (np.sqrt(eigvals[j]) * self.mu + 1e-10))
    
    # Scale eigendirections by directional weights
    eig_scale = np.ones(self.dim)
    for j in range(self.dim):
        if dir_weights[j] < -0.2:
            eig_scale[j] = 0.5 + 0.5 * dir_weights[j]
        elif dir_weights[j] > 0.2:
            eig_scale[j] = 1.0 + 0.3 * dir_weights[j]
    
    # Adapt cc based on gradient strength
    grad_strength = np.linalg.norm(grad_proj) / np.sqrt(self.dim)
    cc_adapt = self.cc * np.clip(1.0 + 0.5 * np.tanh(grad_strength - 1.0), 0.01, 0.3)
    
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Compute rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adapt ccov based on gradient strength
    ccov_adapt = self.ccov * np.clip(1.0 + 0.5 * np.tanh(grad_strength - 0.5), 0.1, 5.0)
    ccov_adapt = np.clip(ccov_adapt, 1e-10, 0.5)
    
    self.C = ((1.0 - ccov_adapt) * self.C + 
              ccov_adapt * rank_one + 
              (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
    
    # Apply directional scaling to eigendirections
    for j in range(self.dim):
        self.C += (eig_scale[j] - 1.0) * eigvals[j] * np.outer(eigvecs[:, j], eigvecs[:, j])
    
    # Check for stagnation
    stagnation_threshold = max(20, self.dim * 2)
    is_stagnant = self.stagnation_counter > stagnation_threshold
    
    if is_stagnant:
        self.stagnation_counter = 0
        
        # Inject directional perturbation aligned with gradient
        for j in range(self.dim):
            if grad_proj[j] > 0:
                strength = 1.0 + 0.5 * np.tanh(grad_proj[j])
            else:
                strength = 1.0 - 0.3 * np.tanh(-grad_proj[j])
            self.C += strength * eigvals[j] * np.outer(eigvecs[:, j], eigvecs[:, j])
        
        # Add exploration noise
        self.C += 0.1 * np.eye(self.dim)
        
        self.pc *= 0.1
        self.L = None
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Reset on extreme ill-conditioning
    eigvals_check = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals_check) / np.min(eigvals_check)
    if cond > 1e8:
        self.C = np.eye(self.dim) * np.mean(eigvals_check)
        self.pc = np.zeros(self.dim)
```