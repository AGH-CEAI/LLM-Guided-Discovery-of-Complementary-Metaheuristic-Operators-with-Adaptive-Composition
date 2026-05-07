**Idea: Covariance-Tracking Velocity with Success-Based Adaptation**

A CMA-ES-inspired velocity update that learns from successful particle movements by maintaining a covariance matrix of recent successful velocity directions. This allows the optimizer to adaptively shape the search distribution, escaping local optima by discovering and exploiting correlated search directions. Unlike the linear combination in the original (which uses fixed uncorrelated random coefficients), this approach learns which velocity directions have historically led to improvements and biases future exploration accordingly.

```python
def _update_velocity_batch(self, inertia):
    """Update velocity with covariance-matrix adaptation from successful moves."""
    # Track successful moves for covariance adaptation
    if not hasattr(self, '_success_buffer'):
        self._success_buffer = np.zeros((self.pop_size, self.dim))
        self._cov_matrix = np.eye(self.dim)
        self._success_count = np.zeros(self.pop_size)
    
    # Detect improvements (successful moves)
    current_pos = self.population.copy()
    improved = self.personal_best_fit < self._prev_fitness if hasattr(self, '_prev_fitness') else np.zeros(self.pop_size, dtype=bool)
    
    # Update success buffer with successful velocity directions
    if hasattr(self, '_prev_population'):
        successful_vel = current_pos - self._prev_population
        for i in range(self.pop_size):
            if improved[i] and np.any(np.isfinite(successful_vel[i])):
                self._success_buffer[i] = 0.9 * self._success_buffer[i] + 0.1 * successful_vel[i]
                self._success_count[i] += 1
            else:
                self._success_count[i] = max(0, self._success_count[i] - 0.1)
    
    self._prev_population = current_pos.copy()
    self._prev_fitness = self.personal_best_fit.copy()
    
    # Build covariance from successful directions (weighted by success count)
    total_weight = np.sum(self._success_count) + 1e-10
    mean_success = np.sum(self._success_buffer * self._success_count[:, np.newaxis], axis=0) / total_weight
    
    # Rank-1 update: direction of mean successful moves
    if np.linalg.norm(mean_success) > 1e-15:
        cov_update = 0.01 * np.outer(mean_success, mean_success) / (np.dot(mean_success, mean_success) + 1e-10)
    else:
        cov_update = np.zeros((self.dim, self.dim))
    
    # Rank-d update: spread successful moves
    if np.any(self._success_count > 0.5):
        weighted_buffer = self._success_buffer * np.sqrt(self._success_count[:, np.newaxis])
        cov_diag = 0.001 * np.mean(weighted_buffer ** 2, axis=0)
        cov_update += np.diag(cov_diag)
    
    # Adapt covariance matrix
    self._cov_matrix = 0.95 * self._cov_matrix + cov_update
    eigvals, eigvecs = np.linalg.eigh(self._cov_matrix)
    eigvals = np.clip(eigvals, 1e-10, None)
    self._cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Standard random components
    r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
    
    cognitive = self.c1 * r1 * (self.personal_best - self.population)
    social = self.c2 * r2 * (self.global_best - self.population)
    neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)
    
    # Base velocity update
    new_velocity = inertia * self.velocity + cognitive + social + neighborhood
    
    # Add covariance-guided perturbation when stagnant (escape local optima)
    stagnation_boost = np.clip(self.stagnation_counter / (self.stagnation_limit + 1), 0, 1)
    if np.any(stagnation_boost > 0.3):
        try:
            cov_sample = np.random.multivariate_normal(
                np.zeros(self.dim), self._cov_matrix, size=self.pop_size
            )
            escape_strength = stagnation_boost[:, np.newaxis] * (self.upper - self.lower) * 0.15 * self.step_size
            new_velocity += escape_strength * cov_sample
        except np.linalg.LinAlgError:
            pass  # Keep original if covariance sampling fails
    
    # Clamp velocity magnitude
    max_vel = (self.upper - self.lower) * 0.25 * self.step_size
    vel_mag = np.linalg.norm(new_velocity, axis=1, keepdims=True)
    scale = np.clip(vel_mag / max_vel, 1.0, None)
    self.velocity = new_velocity / scale
```