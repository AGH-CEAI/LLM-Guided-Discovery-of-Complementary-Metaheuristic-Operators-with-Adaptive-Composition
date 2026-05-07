**Idea: Covariance-Adaptive DE Mutation with Success Memory**
Uses CMA-ES-style covariance tracking and success-weighted mutation direction to escape local optima on severely multimodal/deceptive tasks (17, 16, 11, 19, 23).

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using covariance-adaptive mutation with success memory."""
    np_pop = len(population)
    dim = self.dim
    
    # Initialize covariance tracking on first call
    if not hasattr(self, 'cov_mean'):
        self.cov_mean = np.mean(population, axis=0)
        self.cov_matrix = np.eye(dim) * ((self.upper_bound - self.lower_bound) * 0.1) ** 2
        self.cov_success_vec = np.zeros(dim)
        self.cov_success_count = 0
        self.step_size = (self.upper_bound - self.lower_bound) * 0.1
    
    # Update covariance mean with elite individuals
    n_elite = max(1, int(np_pop * 0.2))
    elite_idx = np.argsort(fitness)[:n_elite]
    elite_mean = np.mean(population[elite_idx], axis=0)
    
    # Track successful mutation directions
    if self.cov_success_count > 0:
        # Blend in successful direction
        alpha = 0.1
        self.cov_mean = (1 - alpha) * self.cov_mean + alpha * elite_mean
        self.cov_success_count = 0
    
    # Eigendecomposition for sampling (cached for efficiency)
    if not hasattr(self, 'cov_eigen_ready') or self.cov_eigen_ready != generation % 10:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self._cov_B = eigenvectors
            self._cov_D = np.sqrt(eigenvalues)
            self.cov_eigen_ready = generation % 10
        except np.linalg.LinAlgError:
            self._cov_B = np.eye(dim)
            self._cov_D = np.ones(dim) * self.step_size
    
    # Generate covariance-sampled mutation direction
    z = np.random.randn(np_pop, dim)
    cov_direction = z * self._cov_D  # Scale by eigenvalues
    cov_direction = (self._cov_B @ cov_direction.T).T  # Rotate by eigenvectors
    cov_direction = cov_direction / (np.linalg.norm(cov_direction, axis=1, keepdims=True) + 1e-10)
    
    # Build ring topology indices
    indices = np.arange(np_pop)
    ring_prev = np.roll(indices, 1)
    ring_next = np.roll(indices, -1)
    
    # Sample indices for classic DE components
    r1 = np.array([np.random.choice(np.delete(indices, i)) for i in range(np_pop)])
    r2 = np.array([np.random.choice(np.delete(indices, [i, r1[i]])) for i in range(np_pop)])
    r3 = np.array([np.random.choice(np.delete(indices, [i, r1[i], r2[i]])) for i in range(np_pop)])
    
    # Get pbest indices
    sorted_idx = np.argsort(fitness)
    pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]
    pbest = population[np.random.choice(pbest_idx, np_pop)]
    
    # Adaptive F with larger range for escaping local optima
    f_base = 0.6 + 0.4 * np.random.rand(np_pop)  # [0.6, 1.0]
    f_cov = 0.3 * np.random.rand(np_pop)  # Weight for covariance direction
    
    # Track success for covariance update
    best_idx = np.argmin(fitness)
    
    # Initialize mutants array
    mutants = np.empty_like(population)
    
    # Operator 0: Covariance-guided rand/1
    if selected_operator == 0:
        mutants = (population[r1] + 
                   f_base[:, np.newaxis] * (population[r2] - population[r3]) +
                   f_cov[:, np.newaxis] * cov_direction * self.step_size)
    
    # Operator 1: Covariance-guided best/1
    elif selected_operator == 1:
        mutants = (population[best_idx] + 
                   f_base[:, np.newaxis] * (population[r1] - population[r2]) +
                   f_cov[:, np.newaxis] * cov_direction * self.step_size)
    
    # Operator 2: Covariance-guided current-to-pbest
    elif selected_operator == 2:
        mutants = (population + 
                   f_base[:, np.newaxis] * (pbest - population[r1] + population[r1] - population[r2]) +
                   f_cov[:, np.newaxis] * cov_direction * self.step_size)
    
    # Operator 3: Covariance-guided rand-to-best
    else:
        mutants = (population[r1] + 
                   f_base[:, np.newaxis] * (population[best_idx] - population[r1] + population[r2] - population[r1]) +
                   f_cov[:, np.newaxis] * cov_direction * self.step_size)
    
    # Update covariance matrix based on elite spread
    if len(elite_idx) >= 2:
        elite_diff = population[elite_idx] - self.cov_mean
        cov_update = np.dot(elite_diff.T, elite_diff) / max(1, len(elite_idx))
        self.cov_matrix = 0.9 * self.cov_matrix + 0.1 * cov_update
    
    # Adapt step size
    if self.cov_success_count > 5:
        self.step_size *= 1.1
    else:
        self.step_size *= 0.95
    self.step_size = np.clip(self.step_size, 1e-6, (self.upper_bound - self.lower_bound) * 0.5)
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```