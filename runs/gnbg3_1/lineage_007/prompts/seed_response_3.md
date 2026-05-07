```python
import numpy as np


class AdaptiveTopologyPSO:
    """
    Particle Swarm Optimizer with adaptive ring topology and covariance-guided perturbation.
    Key features:
    - Dynamic neighborhood size adapting to population diversity
    - Covariance-weighted mutation for exploration
    - Adaptive inertia and acceleration coefficients
    - Success-history based perturbation mechanism
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = kwargs.get('NP', 5 * dim)
        self.NP = min(self.NP, 400)
        self.max_neighbors = kwargs.get('max_neighbors', min(10, self.NP // 2))
        self.min_neighbors = kwargs.get('min_neighbors', 3)
        self.w_max = kwargs.get('w_max', 0.9)
        self.w_min = kwargs.get('w_min', 0.1)
        self.c1_max = kwargs.get('c1_max', 2.5)
        self.c1_min = kwargs.get('c1_min', 0.5)
        self.c2_max = kwargs.get('c2_max', 2.5)
        self.c2_min = kwargs.get('c2_min', 0.5)
        self.f_pert = kwargs.get('f_pert', 0.1)
        self.cr = kwargs.get('cr', 0.9)
        self.LB = -100.0
        self.UB = 100.0
        self._reset_state()

    def _reset_state(self):
        self.population = None
        self.velocities = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fit = None
        self.global_best = None
        self.global_best_fit = np.inf
        self.informants_best = None
        self.informants_best_fit = None
        self.neighbor_indices = None
        self.neighbor_counts = None
        self.success_history = None
        self.failure_history = None
        self.iteration = 0
        self.stagnation_counter = 0
        self.covariance_matrix = None
        self.mean_position = None

    def __call__(self, func, stopping_condition):
        self._reset_state()
        self._initialize_population(func)
        self._initialize_topology()
        self._initialize_covariance()
        self._initialize_success_history()

        while not stopping_condition():
            self.iteration += 1
            self._evaluate_batch(func)
            if stopping_condition():
                break
            self._update_personal_best()
            self._update_global_best()
            self._compute_diversity()
            self._adapt_topology()
            self._update_informants_best()
            self._adapt_inertia_weight()
            self._adapt_acceleration_coefficients()
            self._compute_covariance_adaptation()
            self._move_particles_batch()
            self._apply_bounds_batch()
            self._perturb_velocities_batch()
            self._apply_bounds_batch()
            self._history_mutation_batch(func)
            if stopping_condition():
                break
            self._adapt_success_history()
            self._check_stagnation()

        return self.global_best_fit, self.global_best.copy()

    def _initialize_population(self, func):
        low = self.LB * np.ones(self.dim)
        range_arr = (self.UB - self.LB) * np.ones(self.dim)
        self.population = np.random.uniform(low, low + range_arr, (self.NP, self.dim))
        self.velocities = np.random.uniform(-range_arr * 0.1, range_arr * 0.1, (self.NP, self.dim))
        self.personal_best = self.population.copy()
        self.fitness = func(self.population)
        self._handle_budget_exhaustion(len(self.fitness), self.NP)
        self.personal_best_fit = self.fitness.copy()
        self._update_global_best()

    def _handle_budget_exhaustion(self, returned_len, expected_len):
        if returned_len < expected_len:
            self.population = self.population[:returned_len]
            self.velocities = self.velocities[:returned_len]
            self.personal_best = self.personal_best[:returned_len]
            self.personal_best_fit = self.personal_best_fit[:returned_len]
            self.fitness = self.fitness[:returned_len]
            if self.neighbor_indices is not None:
                self.neighbor_indices = self.neighbor_indices[:returned_len]
            if self.informants_best is not None:
                self.informants_best = self.informants_best[:returned_len]
            if self.informants_best_fit is not None:
                self.informants_best_fit = self.informants_best_fit[:returned_len]
            self.NP = returned_len

    def _initialize_topology(self):
        self.neighbor_counts = np.full(self.NP, self.max_neighbors, dtype=int)

    def _initialize_covariance(self):
        self.covariance_matrix = np.eye(self.dim)
        self.mean_position = np.mean(self.population, axis=0)

    def _initialize_success_history(self):
        self.success_history = np.zeros(self.NP)
        self.failure_history = np.zeros(self.NP)

    def _evaluate_batch(self, func):
        clipped = np.clip(self.population, self.LB, self.UB)
        if not np.allclose(clipped, self.population):
            self.population = clipped
        self.fitness = func(self.population)
        self._handle_budget_exhaustion(len(self.fitness), self.NP)

    def _update_personal_best(self):
        improved = self.fitness < self.personal_best_fit
        improved_idx = np.where(improved)[0]
        if len(improved_idx) > 0:
            self.personal_best[improved_idx] = self.population[improved_idx]
            self.personal_best_fit[improved_idx] = self.fitness[improved_idx]

    def _update_global_best(self):
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.global_best_fit:
            self.global_best_fit = self.fitness[best_idx]
            self.global_best = self.population[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        self.population_diversity = np.mean(distances)
        self.max_diversity = np.sqrt(self.dim) * (self.UB - self.LB) / 2

    def _adapt_topology(self):
        diversity_ratio = self.population_diversity / (self.max_diversity + 1e-10)
        target_neighbors = int(self.min_neighbors + (self.max_neighbors - self.min_neighbors) * (1 - diversity_ratio))
        target_neighbors = np.clip(target_neighbors, self.min_neighbors, self.max_neighbors)
        self.neighbor_counts[:] = target_neighbors
        self._build_ring_topology()

    def _build_ring_topology(self):
        self.neighbor_indices = np.zeros((self.NP, self.max_neighbors), dtype=int)
        for i in range(self.NP):
            k = self.neighbor_counts[i]
            indices = [(i + j) % self.NP for j in range(1, k + 1)]
            indices += [(i - j) % self.NP for j in range(1, self.max_neighbors - k + 1)]
            self.neighbor_indices[i] = indices[:self.max_neighbors]

    def _update_informants_best(self):
        self.informants_best = np.zeros((self.NP, self.dim))
        self.informants_best_fit = np.full(self.NP, np.inf)
        for i in range(self.NP):
            k = self.neighbor_counts[i]
            neighbors = self.neighbor_indices[i, :k]
            best_neighbor = neighbors[np.argmin(self.personal_best_fit[neighbors])]
            self.informants_best[i] = self.personal_best[best_neighbor]
            self.informants_best_fit[i] = self.personal_best_fit[best_neighbor]

    def _adapt_inertia_weight(self):
        progress_ratio = 1.0 - min(1.0, self.stagnation_counter / 50.0)
        self.current_w = self.w_max - (self.w_max - self.w_min) * progress_ratio

    def _adapt_acceleration_coefficients(self):
        progress_ratio = min(1.0, self.stagnation_counter / 30.0)
        self.current_c1 = self.c1_max - (self.c1_max - self.c1_min) * progress_ratio
        self.current_c2 = self.c2_max - (self.c2_max - self.c2_min) * progress_ratio

    def _compute_covariance_adaptation(self):
        self.mean_position = 0.1 * self.mean_position + 0.9 * np.mean(self.population, axis=0)
        centered = self.population - self.mean_position
        emp_cov = np.dot(centered.T, centered) / max(1, self.NP - 1)
        self.covariance_matrix = 0.9 * self.covariance_matrix + 0.1 * emp_cov
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        self.covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _move_particles_batch(self):
        r1 = np.random.uniform(0, 1, (self.NP, self.dim))
        r2 = np.random.uniform(0, 1, (self.NP, self.dim))
        r3 = np.random.uniform(0, 1, (self.NP, self.dim))
        cognitive = self.current_c1 * r1 * (self.personal_best - self.population)
        social_local = self.current_c2 * r2 * (self.informants_best - self.population)
        social_global = 0.5 * r3 * (self.global_best - self.population)
        self.velocities = self.current_w * self.velocities + cognitive + social_local + social_global

    def _apply_bounds_batch(self):
        self.population = np.clip(self.population, self.LB, self.UB)
        max_vel = 0.2 * (self.UB - self.LB)
        self.velocities = np.clip(self.velocities, -max_vel, max_vel)

    def _perturb_velocities_batch(self):
        success_rate = self.success_history / (self.success_history + self.failure_history + 1e-10)
        high_success = success_rate > 0.6
        if np.any(high_success):
            chol = np.linalg.cholesky(self.covariance_matrix + 1e-6 * np.eye(self.dim))
            perturbation = chol @ np.random.randn(self.dim)
            self.velocities[high_success] += self.f_pert * perturbation

    def _history_mutation_batch(self, func):
        n_trials = max(1, self.NP // 5)
        indices = np.random.choice(self.NP, size=n_trials, replace=False)
        trials = np.copy(self.population[indices])
        for i, idx in enumerate(indices):
            k = self.neighbor_counts[idx]
            neighbors = self.neighbor_indices[idx, :k]
            r1, r2, r3 = np.random.choice(neighbors, 3, replace=False)
            mutant = self.population[idx] + self.f_pert * (self.personal_best[r1] - self.population[idx]) + \
                     self.f_pert * (self.global_best - self.personal_best[r2])
            if np.random.rand() < self.cr:
                trials[i] = mutant
        trials = np.clip(trials, self.LB, self.UB)
        trial_fitness = func(trials)
        returned_len = len(trial_fitness)
        if returned_len < n_trials:
            trials = trials[:returned_len]
            trial_fitness = trial_fitness[:returned_len]
        accepted = trial_fitness < self.fitness[indices[:returned_len]]
        for j, acc in enumerate(accepted):
            if acc:
                orig_idx = indices[j]
                self.population[orig_idx] = trials[j]
                self.fitness[orig_idx] = trial_fitness[j]
                if trial_fitness[j] < self.personal_best_fit[orig_idx]:
                    self.personal_best[orig_idx] = trials[j]
                    self.personal_best_fit[orig_idx] = trial_fitness[j]
                self.success_history[orig_idx] += 1
            else:
                self.failure_history[indices[j]] += 1

    def _adapt_success_history(self):
        decay = 0.95
        self.success_history *= decay
        self.failure_history *= decay

    def _check_stagnation(self):
        if self.stagnation_counter > 100:
            self._reinitialize_worst_particles()

    def _reinitialize_worst_particles(self):
        n_reinit = max(1, self.NP // 10)
        worst_indices = np.argsort(self.fitness)[-n_reinit:]
        for idx in worst_indices:
            self.population[idx] = np.random.uniform(self.LB, self.UB, self.dim)
            self.velocities[idx] = np.random.uniform(-10, 10, self.dim)
        self.stagnation_counter = 0
```