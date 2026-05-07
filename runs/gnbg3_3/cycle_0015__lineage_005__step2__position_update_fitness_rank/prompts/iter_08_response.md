**Idea: Stagnation-Triggered Exploration-Exploitation Hybrid**
Combine graph Laplacian spectral gap (spatial clustering detection) with temporal fitness improvement rate (convergence detection). The stagnation score = (1 - spectral_gap) × (1 - norm_improvement_rate) drives a data-driven switch between exploration and exploitation modes. Targets worst tasks 17, 16, 6, 5 where deceptive landscapes trap the swarm.

```python
def _position_update_fitness_rank(self):
    """Hybrid: Graph spectral gap + temporal improvement rate (Category H).

    Combines TWO distinct mechanisms with principled data-driven weighting:
    1. Graph-Laplacian spectral gap: detects spatial clustering of population
    2. Temporal fitness improvement rate: detects convergence/stagnation over time

    Stagnation score = (1 - spectral_gap) * (1 - norm_improvement_rate)
    determines exploration vs exploitation balance:
      - High stagnation → exploration mode (higher velocity scale, larger random perturbations)
      - Low stagnation → exploitation mode (lower velocity scale, stronger directional pull)

    Targets worst tasks (17, 16, 6, 5) where deceptive landscapes trap the swarm.
    """
    # --- Mechanism 1: Graph-Laplacian spectral gap ---
    k = min(5, self.np - 1)
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

    # Connected component analysis
    visited = np.zeros(self.np, dtype=bool)
    components = []
    for start in range(self.np):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in knn_indices[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        components.append(component)

    component_sizes = np.array([len(c) for c in components])
    particle_component_size = np.array([component_sizes[np.argmax([i in c for c in components])] for i in range(self.np)])

    # Spectral gap from normalized Laplacian
    try:
        row_indices = np.repeat(np.arange(self.np), k)
        col_indices = knn_indices.ravel()
        data = np.ones(len(row_indices))
        from scipy.sparse import csr_matrix
        adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
        adj = adj + adj.T
        adj.data[:] = 1.0

        degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
        lap = dspy.identity(self.np) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

        from scipy.sparse.linalg import eigsh
        eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)

        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
        spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
    except:
        spectral_gap = 1.0

    # --- Mechanism 2: Temporal fitness improvement rate ---
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    self._fitness_history.append(float(np.min(self.current_fitness)))
    if len(self._fitness_history) > 20:
        self._fitness_history.pop(0)

    # Compute improvement rate over sliding window
    if len(self._fitness_history) >= 5:
        recent = np.array(self._fitness_history[-5:])
        oldest, newest = recent[0], recent[-1]
        if abs(oldest) > 1e-10:
            improvement_rate = (oldest - newest) / (abs(oldest) + 1e-10)
        else:
            improvement_rate = 0.0
    else:
        improvement_rate = 0.0

    norm_improvement = np.clip(improvement_rate, 0.0, 1.0)

    # --- Principled weighting: stagnation score drives switch ---
    stagnation_score = (1.0 - spectral_gap) * (1.0 - norm_improvement)
    stagnation_score = np.clip(stagnation_score, 0.0, 1.0)

    # Exploration mode when stagnation is high
    is_exploring = stagnation_score > 0.3
    is_stuck = stagnation_score > 0.6

    # Global scale: exploration bias when stuck
    if is_stuck:
        global_scale = 1.5
    elif is_exploring:
        global_scale = 1.15
    else:
        global_scale = 0.85

    # Fitness rank signal
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    # Success history
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)

    # Velocity scaling combines stagnation-driven global scale with per-particle signals
    vel_scale = global_scale * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
    vel_scale = np.clip(vel_scale, 0.3, 2.5)

    # Directional perturbation toward global best (fitness-guided)
    fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        directional = fitness_perturb * to_best_dir
    else:
        directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

    # Random perturbation: stronger when stuck/exploring
    random_scale = 0.5 if is_exploring else 0.25
    random_perturb = np.random.uniform(-random_scale, random_scale, (self.np, self.dim))

    # Component-size-based nudge: small component particles get extra push toward larger components
    size_factor = particle_component_size / max(1, self.np)
    component_nudge = np.zeros((self.np, self.dim))
    for comp_idx, comp in enumerate(components):
        if len(comp) < self.np * 0.3:
            centroid = np.mean(self.population[comp], axis=0)
            for p_idx in comp:
                to_centroid = centroid - self.population[p_idx]
                to_centroid_norm = np.linalg.norm(to_centroid) + 1e-10
                component_nudge[p_idx] = 0.2 * (1.0 - size_factor[p_idx]) * (to_centroid / to_centroid_norm)

    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb + component_nudge
    self.population = self._clip_to_bounds(new_population)
```