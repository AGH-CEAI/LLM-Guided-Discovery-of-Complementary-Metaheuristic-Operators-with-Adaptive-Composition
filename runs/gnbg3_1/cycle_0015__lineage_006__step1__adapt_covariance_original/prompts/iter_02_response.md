**Idea: Directional Memory Covariance with Stagnation Eigenspace Injection**

This variant maintains an explicit archive of successful search direction vectors (normalized step differences weighted by fitness improvement), computes a directional covariance matrix from this archive, and injects exploration energy along the covariance's principal eigendirections when stagnation is detected. This directly attacks the root cause of the worst tasks being stuck ~70 decades above target: the covariance matrix becoming too focused on exploitation directions while ignoring unexplored regions of the search space.
```python
def _adapt_covariance_original(self):
    """Directional memory with eigenspace injection for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Track successful search directions with momentum
    if not hasattr(self, 'step_memory'):
        self.step_memory = []
        self.step_memory_fitness = []
        self.step_memory_size = 5
    
    # Store recent successful steps
    for i in range(min(3, self.NP)):
        if i < len(self.trial_fitness) and i < len(self.fitness):
            if self.trial_fitness[i] < self.fitness[i]:
                step = (self.trials[i] - self.population[i]) / self.sigma
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-10:
                    step_normalized = step / step_norm
                    self.step_memory.append(step_normalized)
                    self.step_memory_fitness.append(self.fitness[i] - self.trial_fitness[i])
    
    # Maintain bounded memory of directions
    while len(self.step_memory) > self.step_memory_size * 3:
        if self.step_memory_fitness:
            min_idx = np.argmin(self.step_memory_fitness)
            self.step_memory.pop(min_idx)
            self.step_memory_fitness.pop(min_idx)
        else:
            self.step_memory.pop(0)
    
    # Compute directional covariance from step memory
    C_dir = np.zeros((self.dim, self.dim))
    if len(self.step_memory) >= 3:
        steps = np.array(self.step_memory)
        weights = np.array(self.step_memory_fitness) if self.step_memory_fitness else np.ones(len(self.step_memory))
        weights = np.maximum(weights, 1e-10)
        weights /= np.sum(weights)
        centered = steps - np.mean(steps, axis=0)
        C_dir = np.dot(centered.T * weights, centered) + 0.1 * np.eye(self.dim)
    
    # Detect stagnation
    improvement = max(0.0, self.f_opt_prev - self.f_opt)
    improvement_threshold = 1e-8 * max(1.0, abs(self.f_opt))
    is_stagnant = improvement < improvement_threshold
    
    # Adaptive learning rates
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    ccov_base = min(0.5, 0.01 / (self.dim + 1.0))
    ccov_adapt = ccov_base * max(0.1, min(10.0, rel_var * 100.0 + 1.0))
    ccov_dir_weight = min(0.3, ccov_adapt * 2.0) if is_stagnant else ccov_adapt * 0.5
    
    # Evolution path update
    cc_adapt = min(self.cc * 2.0, (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim))
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Combine updates with directional memory
    self.C = ((1.0 - ccov_adapt - ccov_dir_weight) * self.C +
              ccov_adapt * rank_one +
              (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
    
    if len(self.step_memory) >= 3:
        self.C = (1.0 - ccov_dir_weight) * self.C + ccov_dir_weight * C_dir
    
    # Inject exploration along eigendirections when stagnant
    if is_stagnant:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        perturbation = np.zeros((self.dim, self.dim))
        for k in range(min(3, self.dim)):
            if k < len(eigvals):
                v = eigvecs[:, k]
                perturbation += 0.05 * eigvals[k] * np.outer(v, v)
        self.C = self.C + perturbation
    
    self.C = self._ensure_positive_definite(self.C)
```