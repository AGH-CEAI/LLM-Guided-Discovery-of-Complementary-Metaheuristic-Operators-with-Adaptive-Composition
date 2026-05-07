import numpy as np


class AdaptiveNeighborhoodCMAES:
    """
    Adaptive Neighborhood CMA-ES: Combines ring topology local search with
    CMA-ES-style covariance adaptation. Uses multiple mutation strategies
    with adaptive operator selection and archive-based diversity maintenance.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        self.NP = min(150, max(40, 5 * dim))
        self.history_size = 10 + dim // 5
        self.max_archive_size = 2 * self.NP
        self.neighbor_range = max(2, dim // 10)

        self.population = None
        self.fitness = None
        self.trial_batch = None
        self.trial_fitness = None
        self._mean = None
        self._cov_eigenvalues = None
        self._cov_eigenvectors = None
        self.sigma = None
        self.p_sigma = None
        self.p_cov = None
        self.archive = None
        self.operator_weights = None
        self.operator_success_counts = None
        self.operator_total_counts = None
        self.fitness_best_history = None
        self.improvement_history = None
        self.generation = 0
        self.stagnation_counter = 0
        self.cumulated_success_ratio = 0.5
        self.successful_mutations = []
        self._init_flag = True
        self._operator_pool_size = 4

    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling."""
        from collections import deque
        samples = np.random.uniform(self.lower, self.upper, size=(self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(self.lower, self.upper, self.NP + 1)
            offsets = np.random.uniform(0, 1, self.NP)
            samples[:, d] = bins[:-1] + offsets * (bins[1] - bins[0])
        np.random.shuffle(samples)
        self.population = np.clip(samples, self.lower, self.upper)
        self.fitness = np.full(self.NP, np.inf)
        self.trial_batch = np.zeros((self.NP, self.dim))
        self.trial_fitness = np.zeros(self.NP)
        self._mean = np.mean(self.population, axis=0)
        self._init_mean()
        self.fitness_best_history = deque(maxlen=self.history_size)
        self.improvement_history = deque(maxlen=self.history_size)
        self.archive = []
        self._init_flag = True

    def _init_mean(self):
        """Initialize CMA-ES parameters."""
        self.sigma = (self.upper - self.lower) / 6.0
        self.p_sigma = np.zeros(self.dim)
        self.p_cov = np.zeros(self.dim)
        self.cov_matrix = np.eye(self.dim) * (self.upper - self.lower) ** 2 / 12.0
        self._eigen_decompose_covariance()
        self.operator_weights = np.ones(self._operator_pool_size) / self._operator_pool_size
        self.operator_success_counts = np.zeros(self._operator_pool_size)
        self.operator_total_counts = np.zeros(self._operator_pool_size)

    def _eigen_decompose_covariance(self):
        """Perform eigendecomposition of covariance matrix."""
        if self.cov_matrix.ndim == 1:
            self.cov_matrix = np.diag(self.cov_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        if eigenvalues[-1] / max(eigenvalues[0], 1e-10) > 1e9:
            eigenvalues = eigenvalues[-1] / 1e9 + (eigenvalues - eigenvalues[0]) * 0.9
            self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        self._cov_eigenvalues = eigenvalues
        self._cov_eigenvectors = eigenvectors

    def _sample_trials_batch(self):
        """Sample trial vectors using adaptive operator selection."""
        if self._init_flag:
            self._adapt_operator_pool()
            self._init_flag = False
        op_indices = np.random.choice(
            self._operator_pool_size, size=self.NP, p=self.operator_weights
        )
        self.trial_batch = np.zeros((self.NP, self.dim))
        for op_id in range(self._operator_pool_size):
            mask = op_indices == op_id
            if np.any(mask):
                self.trial_batch[mask] = self._generate_operator_trial(mask, op_id)

    def _adapt_operator_pool(self):
        """Adapt operator weights based on historical performance."""
        self.operator_weights *= 0.9
        self.operator_weights += 0.1 * (
            self.operator_success_counts / (self.operator_total_counts + 1e-10)
        )
        self.operator_weights = np.maximum(self.operator_weights, 0.05)
        self.operator_weights /= self.operator_weights.sum()
        self.operator_success_counts *= 0.95
        self.operator_total_counts *= 0.95

    def _generate_operator_trial(self, mask, op_id):
        """Generate trials for specified operator."""
        indices = np.where(mask)[0]
        n_select = len(indices)
        if op_id == 0:
            return self._current_to_pbest_mutation_batch(indices, n_select)
        elif op_id == 1:
            return self._ring_topology_mutation_batch(indices, n_select)
        elif op_id == 2:
            return self._cma_mutation_batch(indices, n_select)
        else:
            return self._combined_mutation_batch(indices, n_select)

    def _current_to_pbest_mutation_batch(self, indices, n_select):
        """Current-to-pbest mutation with adaptive F."""
        p_best_size = max(1, int(0.1 * self.NP))
        p_best = np.argsort(self.fitness)[:p_best_size]
        f_scale = 0.5 + 0.3 * np.random.random(n_select)
        rand_indices = np.random.randint(0, self.NP, size=(n_select, 3))
        x_pbest = self.population[np.random.choice(p_best, n_select)]
        mutants = (
            self.population[indices]
            + f_scale[:, None] * (x_pbest - self.population[indices])
            + f_scale[:, None] * (self.population[rand_indices[:, 0]] - self.population[rand_indices[:, 1]])
        )
        return np.clip(mutants, self.lower, self.upper)

    def _ring_topology_mutation_batch(self, indices, n_select):
        """Ring topology mutation with neighborhood search."""
        neighbors = self.neighbor_range
        mutants = np.zeros((n_select, self.dim))
        for i, idx in enumerate(indices):
            left = (idx - neighbors) % self.NP
            right = (idx + neighbors) % self.NP
            candidates = [left, right, (left + right) // 2 % self.NP]
            r1, r2, r3 = [candidates[k % len(candidates)] for k in range(3)]
            f_scale = 0.6 + 0.4 * np.random.random()
            mutants[i] = (
                self.population[idx]
                + f_scale * (self.population[r1] - self.population[r2])
                + f_scale * (self._mean - self.population[r3])
            )
        return np.clip(mutants, self.lower, self.upper)

    def _cma_mutation_batch(self, indices, n_select):
        """CMA-ES style mutation using eigendecomposed covariance."""
        z = np.random.standard_normal((n_select, self.dim))
        if self._cov_eigenvalues is not None:
            z = z @ (self._cov_eigenvectors * np.sqrt(self._cov_eigenvalues))
        f_scale = 0.3 + 0.4 * np.random.random(n_select)
        mutants = self._mean + self.sigma * f_scale[:, None] * z
        return np.clip(mutants, self.lower, self.upper)

    def _combined_mutation_batch(self, indices, n_select):
        """Combined mutation: DE + CMA-ES influence."""
        z = np.random.standard_normal((n_select, self.dim))
        if self._cov_eigenvalues is not None:
            z = z @ (self._cov_eigenvectors * np.sqrt(self._cov_eigenvalues))
        f_scale = 0.4 + 0.3 * np.random.random(n_select)
        rand_idx = np.random.randint(0, self.NP, size=(n_select, 2))
        mutants = (
            self._mean
            + 0.5 * (self.population[rand_idx[:, 0]] - self.population[rand_idx[:, 1]])
            + 0.5 * self.sigma * f_scale[:, None] * z
        )
        return np.clip(mutants, self.lower, self.upper)

    def _crossover_batch(self):
        """Binomial crossover between population and trial batch."""
        cr_base = 0.5 + 0.3 * np.random.random(self.NP)
        dim_mask = np.random.random((self.NP, self.dim)) < cr_base[:, None]
        dim_mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        self.trial_batch = np.where(dim_mask, self.trial_batch, self.population)

    def _evaluate_trial_batch(self):
        """Evaluate trial batch with budget handling."""
        self.trial_fitness = func(self.trial_batch)
        if len(self.trial_fitness) < self.NP:
            valid = len(self.trial_fitness)
            self.trial_batch = self.trial_batch[:valid]
            self.trial_fitness = self.trial_fitness[:valid]
            return False
        return True

    def _select_survivors_batch(self):
        """Select survivors using elitist archive and niched selection."""
        improved = self.trial_fitness < self.fitness
        if np.any(improved):
            self.cumulated_success_ratio = 0.9 * self.cumulated_success_ratio + 0.1 * 0.8
            for i in np.where(improved)[0]:
                diff = self.trial_batch[i] - self.population[i]
                self.successful_mutations.append(diff * self.sigma)
                if len(self.successful_mutations) > self.NP * 3:
                    self.successful_mutations.pop(0)
        else:
            self.cumulated_success_ratio *= 0.95
        op_indices = np.random.choice(
            self._operator_pool_size, size=self.NP, p=self.operator_weights
        )
        for i in range(self._operator_pool_size):
            mask = op_indices == i
            self.operator_total_counts[i] += np.sum(mask)
            self.operator_success_counts[i] += np.sum(mask & improved)
        self._update_archive()
        self._elitist_selection()

    def _update_archive(self):
        """Update archive with non-dominated solutions."""
        trial_archive = self.trial_batch[~improved].copy() if np.any(~improved) else np.array([]).reshape(0, self.dim)
        trial_fit_archive = self.trial_fitness[~improved] if np.any(~improved) else np.array([])
        all_candidates = (
            list(zip(self.archive, [f for _, f in self.archive]))
            if self.archive else []
        )
        trial_pairs = list(zip(trial_archive, trial_fit_archive))
        all_candidates.extend(trial_pairs)
        pareto_front = []
        for candidate, fit in all_candidates:
            is_dominated = False
            remaining = []
            for existing, existing_fit in pareto_front:
                if np.all(fit <= existing_fit) and np.any(fit < existing_fit):
                    is_dominated = True
                elif np.all(existing_fit <= fit) and np.any(existing_fit < fit):
                    continue
                remaining.append((existing, existing_fit))
            if not is_dominated:
                remaining.append((candidate, fit))
            pareto_front = remaining
        self.archive = [c for c, _ in pareto_front[: self.max_archive_size]]

    def _elitist_selection(self):
        """Elitist selection combining population and archive."""
        combined = np.vstack([self.population, self.trial_batch])
        combined_fitness = np.concatenate([self.fitness, self.trial_fitness])
        sorted_idx = np.argsort(combined_fitness)[: self.NP]
        self.population = combined[sorted_idx]
        self.fitness = combined_fitness[sorted_idx]
        if len(self.archive) > 0:
            archive_arr = np.array(self.archive)
            archive_fitness = np.array([f for _, f in self.archive])
            combined = np.vstack([self.population, archive_arr])
            combined_fitness = np.concatenate([self.fitness, archive_fitness])
            if len(combined) > self.NP:
                combined_fitness_tile = np.tile(combined_fitness, (len(combined), 1))
                distances = np.abs(combined_fitness_tile - combined_fitness[:, None]) + 1e-10
                diversity_scores = np.mean(distances, axis=1)
                scores = 0.7 * (combined_fitness - combined_fitness.min()) + 0.3 * diversity_scores
                keep_idx = np.argsort(scores)[: self.NP]
            else:
                keep_idx = np.argsort(combined_fitness)[: self.NP]
            self.population = combined[keep_idx]
            self.fitness = combined_fitness[keep_idx]
        self._mean = np.mean(self.population, axis=0)

    def _compute_diversity(self):
        """Compute population diversity as average distance to mean."""
        distances = np.linalg.norm(self.population - self._mean, axis=1)
        return np.mean(distances) / ((self.upper - self.lower) / np.sqrt(12))

    def _adapt_step_size(self):
        """Adapt step size using evolution path."""
        chi_d = self.dim ** 0.5 * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))
        c_sigma = 2.0 / (self.dim + 7)
        d_sigma = 1.0 + c_sigma
        expected_norm = self.dim ** 0.5 * (1 - 1.0 / (4 * self.dim))
        norm_ratio = np.linalg.norm(self.p_sigma) / expected_norm
        self.sigma *= np.exp((norm_ratio - chi_d) / d_sigma)
        self.sigma = np.clip(self.sigma, 1e-10, (self.upper - self.lower) / 3)

    def _adapt_covariance_from_success(self):
        """Adapt covariance matrix from successful mutations."""
        if len(self.successful_mutations) < 2:
            return
        c_cov = 2.0 / (self.dim ** 2 + 6)
        c1 = 1.0 / (self.dim ** 2 + 12)
        c_mu = min(0.6, c_cov * 2)
        successful = np.array(self.successful_mutations[-self.NP :])
        n_succ = len(successful)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(n_succ):
            v = successful[i : i + 1].T
            rank_mu += np.dot(v, v.T)
        if n_succ > 0:
            rank_mu /= n_succ * self.sigma ** 2
        self.cov_matrix = (
            (1 - c1 - c_mu) * self.cov_matrix
            + c1 * np.outer(self.p_cov, self.p_cov)
            + c_mu * rank_mu
        )
        self._eigen_decompose_covariance()

    def _update_evolution_paths(self):
        """Update evolution paths for step size and covariance."""
        c_sigma = 2.0 / (self.dim + 7)
        c_cov = 2.0 / (self.dim ** 2 + 6)
        successful = np.array(self.successful_mutations[-self.NP :]) if self.successful_mutations else np.zeros((0, self.dim))
        if len(successful) > 0:
            mean_succ = np.mean(successful, axis=0)
            self.p_sigma = (1 - c_sigma) * self.p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * mean_succ / self.sigma
        else:
            self.p_sigma *= (1 - c_sigma)
        if len(successful) > 0:
            self.p_cov = (1 - c_cov) * self.p_cov + np.sqrt(c_cov * (2 - c_cov) * min(1.5, 1 + self.cumulated_success_ratio)) * mean_succ / self.sigma
        else:
            self.p_cov *= (1 - c_cov)

    def _check_stagnation(self):
        """Check for stagnation based on fitness history."""
        self.fitness_best_history.append(self.fitness[0])
        if len(self.fitness_best_history) > 1:
            recent_improvement = abs(self.fitness_best_history[0] - self.fitness_best_history[-1])
            self.improvement_history.append(recent_improvement)
        is_stagnant = len(self.improvement_history) >= 5 and np.mean(list(self.improvement_history)[-5:]) < 1e-10
        if is_stagnant:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        return self.stagnation_counter >= 3 * self.dim or self._compute_diversity() < 1e-4

    def _restart_if_needed(self):
        """Restart population if stagnation detected."""
        if self._check_stagnation():
            self.population = np.random.uniform(
                self.lower, self.upper, size=(self.NP, self.dim)
            )
            self.fitness = np.full(self.NP, np.inf)
            self.sigma = (self.upper - self.lower) / 6.0
            self.cov_matrix = np.eye(self.dim) * (self.upper - self.lower) ** 2 / 12.0
            self._eigen_decompose_covariance()
            self.p_sigma = np.zeros(self.dim)
            self.p_cov = np.zeros(self.dim)
            self.successful_mutations = []
            self.generation = 0
            self.stagnation_counter = 0
            self.improvement_history = deque(maxlen=self.history_size)
            self._mean = np.mean(self.population, axis=0)

    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self.fitness = func(self.population)
        if len(self.fitness) < self.NP:
            self.population = self.population[: len(self.fitness)]
            self.NP = len(self.fitness)
            self.fitness = self.fitness[: len(self.fitness)]
        best_idx = np.argmin(self.fitness)
        x_opt = self.population[best_idx].copy()
        f_opt = self.fitness[best_idx]
        while not stopping_condition():
            self._sample_trials_batch()
            self._crossover_batch()
            self.trial_batch = np.clip(self.trial_batch, self.lower, self.upper)
            if not self._evaluate_trial_batch():
                break
            if stopping_condition():
                break
            self._select_survivors_batch()
            self._update_evolution_paths()
            self._adapt_step_size()
            self._adapt_covariance_from_success()
            self.generation += 1
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < f_opt:
                f_opt = self.fitness[current_best_idx]
                x_opt = self.population[current_best_idx].copy()
            if stopping_condition():
                break
            self._restart_if_needed()
        return f_opt, x_opt
