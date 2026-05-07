import numpy as np


class AdaptiveCovarianceDE:
    """
    Adaptive Covariance Differential Evolution (ACDE)
    DE backbone with covariance-guided mutation and adaptive strategy selection.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0

        # Population size: 5*dim capped at 400
        self.np = min(max(5 * dim, 100), 400)

        # DE control parameters
        self.f = 0.6
        self.cr = 0.85

        # Covariance adaptation parameters
        self.cov_matrix = np.eye(dim)
        self.cov_learning_rate = 0.02
        self.mean_vector = np.zeros(dim)
        self.step_size = 1.0

        # Strategy weights for adaptive selection
        self.strategy_weights = np.array([0.5, 0.3, 0.2])

        # Archive for diversity
        self.archive = np.zeros((0, dim))
        self.archive_fitness = np.array([])
        self.archive_max_size = self.np * 2

        # Best tracking
        self.best_fitness = np.inf
        self.best_vector = None

        # Stagnation and diversity tracking
        self.stagnation_count = 0
        self.max_stagnation = 40
        self.diversity_threshold = 1e-4

        # Per-generation success tracking for adaptation
        self.strategy_successes = np.zeros(3)
        self.total_successes = 0

    def _initialize_population(self):
        """Initialize population uniformly in bounds and compute initial statistics."""
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.mean_vector = population.mean(axis=0)
        self.cov_matrix = np.cov(population.T) + 1e-6 * np.eye(self.dim)
        return population

    def _update_covariance_matrix(self, selected_population, selected_fitness):
        """Update covariance matrix using weighted elite individuals."""
        n_elite = max(2, self.np // 5)
        elite_indices = np.argsort(selected_fitness)[:n_elite]
        elite_pop = selected_population[elite_indices]

        # Weighted mean based on fitness
        weights = 1.0 / (selected_fitness[elite_indices] - selected_fitness.min() + 1e-8)
        weights = weights / weights.sum()
        weighted_mean = (elite_pop * weights[:, np.newaxis]).sum(axis=0)

        # Update covariance estimate
        diffs = elite_pop - self.mean_vector
        cov_update = (diffs.T @ (diffs * weights[:, np.newaxis])) / weights.sum()
        self.cov_matrix = (1 - self.cov_learning_rate) * self.cov_matrix + \
                          self.cov_learning_rate * cov_update

        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if eigvals.min() < 1e-8:
            self.cov_matrix += (1e-6 - eigvals.min()) * np.eye(self.dim)

        self.mean_vector = 0.5 * self.mean_vector + 0.5 * weighted_mean

    def _sample_trials_batch(self, population, fitness):
        """Sample trial batch using adaptive weighted strategies."""
        # Select strategy based on weights
        strategy_idx = np.random.choice(3, p=self.strategy_weights)

        if strategy_idx == 0:
            trials = self._mutate_current_to_pbest_batch(population, fitness)
        elif strategy_idx == 1:
            trials = self._mutate_covariance_guided_batch(population, fitness)
        else:
            trials = self._mutate_diversity_guided_batch(population, fitness)

        return trials, strategy_idx

    def _mutate_current_to_pbest_batch(self, population, fitness):
        """Current-to-pbest/1 mutation with archive."""
        n_pbest = max(1, int(0.15 * self.np))
        pbest_indices = np.argsort(fitness)[:n_pbest]
        pbest = population[pbest_indices]

        # Select 3 distinct random indices for each target
        all_indices = np.arange(self.np)
        r_indices = np.random.choice(self.np, size=(self.np, 4), replace=True)

        # Ensure different from target index
        for i in range(self.np):
            while r_indices[i, 0] == i:
                r_indices[i, 0] = np.random.randint(self.np)
            while r_indices[i, 1] == i or r_indices[i, 1] == r_indices[i, 0]:
                r_indices[i, 1] = np.random.randint(self.np)
            while r_indices[i, 2] == i or r_indices[i, 2] in r_indices[i, :2]:
                r_indices[i, 2] = np.random.randint(self.np)
            while r_indices[i, 3] in r_indices[i, :3]:
                r_indices[i, 3] = np.random.randint(self.np)

        r1, r2, r3, r4 = r_indices[:, 0], r_indices[:, 1], r_indices[:, 2], r_indices[:, 3]
        pbest_for_target = pbest[np.random.randint(n_pbest, size=self.np)]

        # DE/current-to-pbest/1/bin with 2 difference vectors
        diff_scale = np.random.uniform(0.5, 1.0)
        trials = population + diff_scale * self.f * (pbest_for_target - population) + \
                 self.f * (population[r1] - population[r2]) + \
                 0.5 * self.f * (population[r3] - population[r4])

        return trials

    def _mutate_covariance_guided_batch(self, population, fitness):
        """Covariance-guided mutation using weighted mean and covariance."""
        # Get weighted mean of top individuals
        n_elite = max(2, self.np // 4)
        elite_indices = np.argsort(fitness)[:n_elite]
        elite_pop = population[elite_indices]

        weights = 1.0 / (fitness[elite_indices] - fitness.min() + 1e-8)
        weights = weights / weights.sum()
        weighted_mean = (elite_pop * weights[:, np.newaxis]).sum(axis=0)

        # Add archive to covariance estimation if available
        if len(self.archive) > self.dim:
            combined = np.vstack([elite_pop, self.archive])
            local_cov = np.cov(combined.T) + 1e-6 * np.eye(self.dim)
        else:
            local_cov = self.cov_matrix

        # Generate multivariate normal samples
        try:
            L = np.linalg.cholesky(local_cov)
            noise = np.random.randn(self.np, self.dim)
            perturbations = noise @ L.T
        except np.linalg.LinAlgError:
            perturbations = np.random.randn(self.np, self.dim)

        # Scale by current step size and F
        trials = weighted_mean + self.step_size * self.f * perturbations

        return trials

    def _mutate_diversity_guided_batch(self, population, fitness):
        """Diversity-guided mutation using random walks in sparse dimensions."""
        # Find dimensions with low variance (potential convergence)
        pop_std = population.std(axis=0)
        threshold = pop_std.mean() * 0.5
        sparse_dims = pop_std < threshold

        # Select indices for differential mutation
        indices = np.random.choice(self.np, size=(self.np, 3), replace=True)
        for i in range(self.np):
            while indices[i, 0] == i:
                indices[i, 0] = np.random.randint(self.np)
            while indices[i, 1] == i or indices[i, 1] == indices[i, 0]:
                indices[i, 1] = np.random.randint(self.np)
            while indices[i, 2] == i or indices[i, 2] in indices[i, :2]:
                indices[i, 2] = np.random.randint(self.np)

        r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]

        # Standard DE/rand/1 mutation
        base = population[r1]
        diff = self.f * (population[r2] - population[r3])

        # Add exploration term in sparse dimensions
        exploration = np.zeros_like(population)
        exploration[:, sparse_dims] = np.random.randn(self.np, sparse_dims.sum()) * 10.0

        trials = base + diff + 0.3 * exploration

        return trials

    def _crossover_batch(self, population, trials):
        """Binomial crossover with guaranteed at least one parameter from mutant."""
        rand_mask = np.random.rand(self.np, self.dim) < self.cr
        enforce_idx = np.random.randint(self.dim, size=self.np)
        enforce_mask = np.zeros_like(rand_mask)
        enforce_mask[np.arange(self.np), enforce_idx] = True
        crossed = np.where(rand_mask | enforce_mask, trials, population)
        return crossed

    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection between parent and trial vectors."""
        improved_mask = trial_fitness < fitness
        n_improved = np.sum(improved_mask)

        new_pop = np.where(improved_mask[:, np.newaxis], trials, population)
        new_fit = np.where(improved_mask, trial_fitness, fitness)

        return new_pop, new_fit, n_improved, improved_mask

    def _update_archive(self, population, fitness, trials, trial_fitness):
        """Update archive with successful solutions."""
        improved_mask = trial_fitness < fitness
        if not np.any(improved_mask):
            return

        improved_solutions = trials[improved_mask]
        improved_fitness = trial_fitness[improved_mask]

        # Add to archive
        self.archive = np.vstack([self.archive, improved_solutions])
        self.archive_fitness = np.concatenate([self.archive_fitness, improved_fitness])

        # Trim archive if too large
        if len(self.archive) > self.archive_max_size:
            # Keep best half
            keep_indices = np.argsort(self.archive_fitness)[:self.archive_max_size // 2]
            self.archive = self.archive[keep_indices]
            self.archive_fitness = self.archive_fitness[keep_indices]

    def _compute_diversity(self, population):
        """Compute population diversity using average pairwise distance."""
        sample_size = min(50, len(population))
        if sample_size < 2:
            return 0.0

        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]

        diffs = sample[:, np.newaxis] - sample[np.newaxis, :]
        distances = np.sqrt(np.sum(diffs**2, axis=2))

        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        return distances[mask].mean() if np.any(mask) else 0.0

    def _update_best(self, fitness, population):
        """Update best solution tracking."""
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_vector = population[best_idx].copy()
            return True
        return False

    def _adapt_step_size(self, improvement_rate):
        """Adapt DE step size F based on improvement rate."""
        if improvement_rate > 0.2:
            self.f = min(1.0, self.f * 1.1)
        elif improvement_rate < 0.05:
            self.f = max(0.3, self.f * 0.9)
        return self.f

    def _adapt_crossover_rate(self, improvement_rate):
        """Adapt crossover rate based on success."""
        if improvement_rate > 0.15:
            self.cr = min(0.95, self.cr + 0.05)
        elif improvement_rate < 0.03:
            self.cr = max(0.3, self.cr - 0.05)
        return self.cr

    def _adapt_strategy_weights(self):
        """Adapt mutation strategy weights based on success rates."""
        if self.total_successes == 0:
            return

        success_rates = self.strategy_successes / (self.total_successes + 1e-10)
        self.strategy_weights = 0.7 * self.strategy_weights + 0.3 * success_rates
        self.strategy_weights = np.maximum(self.strategy_weights, 0.05)
        self.strategy_weights /= self.strategy_weights.sum()

    def _adapt_covariance_variant_01(self):
        """Adapt covariance matrix using rank-1 update."""
        if len(self.archive) > self.dim:
            # Use difference between current mean and archive mean
            archive_mean = self.archive.mean(axis=0)
            diff = archive_mean - self.mean_vector
            rank1_update = np.outer(diff, diff)
            self.cov_matrix = (1 - 0.01) * self.cov_matrix + 0.01 * rank1_update

    def _adapt_covariance_variant_08(self):
        """Adapt step size based on covariance eigenvalues."""
        try:
            eigvals = np.linalg.eigvalsh(self.cov_matrix)
            condition_number = eigvals.max() / (eigvals.min() + 1e-10)
            if condition_number > 1e6:
                self.step_size = max(0.5, self.step_size * 0.9)
            else:
                self.step_size = min(2.0, self.step_size * 1.02)
        except np.linalg.LinAlgError:
            pass

    def _adapt_covariance_variant_09(self):
        """Adapt covariance learning rate based on diversity."""
        diversity = self._compute_diversity(self.archive) if len(self.archive) > 0 else 1.0
        if diversity < self.diversity_threshold:
            self.cov_learning_rate = min(0.1, self.cov_learning_rate * 1.1)
        else:
            self.cov_learning_rate = max(0.005, self.cov_learning_rate * 0.95)

    def _check_stagnation(self, n_improved, diversity):
        """Check if optimization has stagnated."""
        if n_improved > 0:
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        return (self.stagnation_count >= self.max_stagnation or
                diversity < self.diversity_threshold)

    def _perform_restart(self, population, fitness):
        """Perform adaptive restart with diversity injection."""
        # Keep best 20% of population
        n_keep = max(2, self.np // 5)
        keep_indices = np.argsort(fitness)[:n_keep]
        survivors = population[keep_indices].copy()

        # Generate new random population
        new_pop = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np - n_keep, self.dim)
        )

        # Combine survivors with new random individuals
        population = np.vstack([survivors, new_pop])
        fitness = np.full(self.np, np.inf)

        # Reset tracking
        self.stagnation_count = 0
        self.archive = np.zeros((0, self.dim))
        self.archive_fitness = np.array([])
        self.strategy_weights = np.array([0.5, 0.3, 0.2])
        self.strategy_successes[:] = 0
        self.total_successes = 0

        return population, fitness

    def __call__(self, func, stopping_condition):
        """Run the optimizer with batched operations."""
        population = self._initialize_population()
        fitness = func(population)

        if len(fitness) < self.np:
            fitness = np.pad(fitness, (0, self.np - len(fitness)), constant_values=np.inf)

        self._update_best(fitness, population)
        self.strategy_successes[:] = 0
        self.total_successes = 0

        while not stopping_condition():
            # Sample trial vectors using adaptive strategy
            trials, strategy_idx = self._sample_trials_batch(population, fitness)
            trials = self._clip_to_bounds(trials)

            # Crossover
            trials = self._crossover_batch(population, trials)
            trials = self._clip_to_bounds(trials)

            # Evaluate trials
            trial_fitness = func(trials)

            # Handle truncated output
            if len(trial_fitness) < len(trials):
                actual_len = len(trial_fitness)
                trial_fitness = np.pad(trial_fitness, (0, len(trials) - actual_len),
                                       constant_values=np.inf)

            if stopping_condition():
                break

            # Selection
            population, fitness, n_improved, improved_mask = \
                self._select_survivors_batch(population, fitness, trials, trial_fitness)

            # Update archive
            self._update_archive(population, fitness, trials, trial_fitness)

            # Track strategy success
            self.strategy_successes[strategy_idx] += n_improved
            self.total_successes += n_improved

            # Update best solution
            self._update_best(fitness, population)

            # Compute diversity
            diversity = self._compute_diversity(population)

            # Check stagnation
            if self._check_stagnation(n_improved, diversity):
                population, fitness = self._perform_restart(population, fitness)
                fitness = func(population)
                if len(fitness) < self.np:
                    break
                continue

            # Adaptation methods
            improvement_rate = n_improved / self.np
            self._adapt_step_size(improvement_rate)
            self._adapt_crossover_rate(improvement_rate)

            if self.total_successes > 0:
                self._adapt_strategy_weights()

            self._adapt_covariance_variant_01()
            self._adapt_covariance_variant_08()
            self._adapt_covariance_variant_09()

            # Update covariance matrix periodically
            if len(self.archive) > self.dim:
                self._update_covariance_matrix(population, fitness)

        return self.best_fitness, self.best_vector
