import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, intelligent restart mechanism, and adaptive operator selection.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(8 * dim, 300)
        self.archive_max = self.np_size
        self.memory_size = 5
        self.p_best_rate = 0.15
        self.restart_threshold = 50
        self.min_pop_size = max(4, dim // 2)

        # Adaptive operator selection: 3 operators
        # 0 = original, 1 = variant_01 (rank-weighted), 2 = variant_09 (eigenvector-guided)
        self.n_operators = 3
        self.op_successes = np.ones(self.n_operators, dtype=float)
        self.op_failures = np.ones(self.n_operators, dtype=float)
        self.op_window_size = 50
        self.op_reward_history = [[] for _ in range(self.n_operators)]
        self.current_op_assignments = None

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset operator stats
        self.op_successes = np.ones(self.n_operators, dtype=float)
        self.op_failures = np.ones(self.n_operators, dtype=float)
        self.op_reward_history = [[] for _ in range(self.n_operators)]

        population, fitness = self._initialize_population(func, stopping_condition)
        if stopping_condition():
            return self.f_opt, self.x_opt

        self._update_best(population, fitness)
        archive = np.empty((0, self.dim))
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        prev_best_fitness = self.f_opt

        while not stopping_condition():
            n = len(population)
            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr, n)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch(n)

            # Select operator for each individual using Thompson Sampling
            op_assignments = self._select_operators(n)
            self.current_op_assignments = op_assignments

            # Generate mutant vectors using adaptive operator selection
            mutants = self._mutate_batch_adaptive(population, fitness, archive, f_values, strategy_mask, op_assignments)

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values)

            # Clip to bounds
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                self._safe_update_best(trials, trial_fitness)
                break

            # Handle truncated results
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                f_values = f_values[:len(trial_fitness)]
                cr_values = cr_values[:len(trial_fitness)]
                op_assignments = op_assignments[:len(trial_fitness)]

            valid = ~np.isnan(trial_fitness)

            # Selection and archive update, also track operator performance
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid, op_assignments
                )
            )

            # Update parameter memory
            memory_f, memory_cr, memory_idx = self._update_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta
            )

            self._update_best(population, fitness)

            # Detect stagnation
            stagnant = self._detect_stagnation(prev_best_fitness)
            if stagnant:
                population, fitness, archive, memory_f, memory_cr, memory_idx = (
                    self._restart(func, stopping_condition, population, fitness)
                )
                if stopping_condition():
                    break

            prev_best_fitness = self.f_opt
            self.generation += 1

            # Periodically try opposition-based learning
            if self.generation % 25 == 0 and not stopping_condition():
                population, fitness = self._opposition_based_jump(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
                self._update_best(population, fitness)

        return self.f_opt, self.x_opt

    def _select_operators(self, n):
        """Select operator for each individual using Thompson Sampling."""
        samples = np.zeros((n, self.n_operators))
        for k in range(self.n_operators):
            alpha = max(self.op_successes[k], 0.01)
            beta = max(self.op_failures[k], 0.01)
            samples[:, k] = np.random.beta(alpha, beta, size=n)
        return np.argmax(samples, axis=1)

    def _update_operator_stats(self, op_idx, reward):
        """Update operator statistics with a sliding window decay."""
        op_idx = int(op_idx)
        reward = float(reward)
        if reward > 0:
            self.op_successes[op_idx] += reward
        else:
            self.op_failures[op_idx] += 1.0

        self.op_reward_history[op_idx].append(reward)
        # Trim history
        if len(self.op_reward_history[op_idx]) > self.op_window_size:
            self.op_reward_history[op_idx] = self.op_reward_history[op_idx][-self.op_window_size:]

        # Periodic decay to allow adaptation
        total = self.op_successes[op_idx] + self.op_failures[op_idx]
        if total > 100:
            decay = 0.95
            self.op_successes[op_idx] *= decay
            self.op_failures[op_idx] *= decay
            self.op_successes[op_idx] = max(self.op_successes[op_idx], 1.0)
            self.op_failures[op_idx] = max(self.op_failures[op_idx], 1.0)

    def _mutate_batch_adaptive(self, population, fitness, archive, f_values, strategy_mask, op_assignments):
        """Dispatch to different mutation operators based on op_assignments."""
        n = len(population)
        dim = self.dim
        mutants = np.empty((n, dim))

        mask0 = op_assignments == 0
        mask1 = op_assignments == 1
        mask2 = op_assignments == 2

        if np.any(mask0):
            idx = np.where(mask0)[0]
            mutants[idx] = self._mutate_original(population, fitness, archive, f_values, strategy_mask, idx)

        if np.any(mask1):
            idx = np.where(mask1)[0]
            mutants[idx] = self._mutate_variant01(population, fitness, archive, f_values, strategy_mask, idx)

        if np.any(mask2):
            idx = np.where(mask2)[0]
            mutants[idx] = self._mutate_variant09(population, fitness, archive, f_values, strategy_mask, idx)

        return mutants

    def _mutate_original(self, population, fitness, archive, f_values, strategy_mask, indices):
        """Original current-to-pbest/1 and rand-to-pbest/1."""
        n = len(population)
        dim = self.dim
        ni = len(indices)

        sorted_idx = np.argsort(fitness)
        p = max(2, int(np.ceil(self.p_best_rate * n)))

        pbest_idx = sorted_idx[np.random.randint(0, p, size=ni)]
        r1 = self._random_indices_not_equal_subset(n, indices)

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(ni, dtype=int)
        for j in range(ni):
            for _ in range(100):
                idx = np.random.randint(0, union_size)
                if idx != indices[j] and idx != r1[j]:
                    r2[j] = idx
                    break

        f_col = f_values[indices, np.newaxis]
        result = np.empty((ni, dim))

        sm = strategy_mask[indices]
        mask1 = sm
        mask2 = ~sm

        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            result[idx1] = (
                population[indices[idx1]]
                + f_col[idx1] * (population[pbest_idx[idx1]] - population[indices[idx1]])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            )

        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal_subset(n, indices[idx2])
            result[idx2] = (
                population[r_base]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            )

        return result

    def _mutate_variant01(self, population, fitness, archive, f_values, strategy_mask, indices):
        """Rank-weighted current-to-pbest/1 with adaptive p (variant_01)."""
        n = len(population)
        dim = self.dim
        ni = len(indices)

        sorted_idx = np.argsort(fitness)

        # Rank-based F scaling
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)
        rank_ratio = ranks / max(n - 1, 1)

        # Scale F for selected indices
        f_scaled = f_values[indices] * (0.5 + 0.5 * rank_ratio[indices])
        f_scaled = np.clip(f_scaled, 0.01, 1.0)

        # Adaptive p-best rate
        generation = getattr(self, 'generation', 0)
        progress = min(1.0, generation / 500.0)
        p_min = max(2, int(np.ceil(0.05 * n)))
        p_max = max(2, int(np.ceil(0.25 * n)))

        p_values = np.clip(
            (p_min + (p_max - p_min) * rank_ratio[indices] * (1.0 - 0.5 * progress)).astype(int),
            2, n
        )

        pbest_idx = np.array([sorted_idx[np.random.randint(0, p_values[j])] for j in range(ni)])

        r1 = self._random_indices_not_equal_subset(n, indices)

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(ni, dtype=int)
        for j in range(ni):
            for _ in range(100):
                idx = np.random.randint(0, union_size)
                if idx != indices[j] and idx != r1[j]:
                    r2[j] = idx
                    break

        f_col = f_scaled[:, np.newaxis]

        result = (
            population[indices]
            + f_col * (population[pbest_idx] - population[indices])
            + f_col * (population[r1] - union[r2])
        )

        # For strategy_mask == False, add second difference vector
        sm = strategy_mask[indices]
        mask2 = ~sm
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r3 = self._random_indices_not_equal_subset(n, indices[idx2])
            r4 = self._random_indices_not_equal_subset(n, np.array([r1[j] for j in idx2]))
            weight2 = 0.5 * f_col[idx2]
            result[idx2] += weight2 * (population[r3] - population[r4])

        return result

    def _mutate_variant09(self, population, fitness, archive, f_values, strategy_mask, indices):
        """Eigenvector-guided mutation (variant_09)."""
        n = len(population)
        dim = self.dim
        ni = len(indices)
        f_col = f_values[indices, np.newaxis]

        sorted_idx = np.argsort(fitness)

        # Compute covariance direction from top individuals
        n_top = max(3, n // 4)
        top_pop = population[sorted_idx[:n_top]]
        center = np.mean(top_pop, axis=0)

        try:
            if dim <= 100 and n_top >= 3:
                diffs = top_pop - center
                cov = np.dot(diffs.T, diffs) / max(1, n_top - 1) + 1e-10 * np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_eigvals = np.sqrt(eigvals / np.max(eigvals))
            else:
                eigvecs = np.eye(dim)
                sqrt_eigvals = np.ones(dim)
        except Exception:
            eigvecs = np.eye(dim)
            sqrt_eigvals = np.ones(dim)

        p1 = max(2, int(0.1 * n))
        p2 = max(3, int(0.3 * n))

        donor1_idx = sorted_idx[np.random.randint(0, p1, size=ni)]
        donor2_idx = sorted_idx[np.random.randint(0, p2, size=ni)]

        r1 = self._random_indices_not_equal_subset(n, indices)

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(ni, dtype=int)
        for j in range(ni):
            for _ in range(100):
                idx = np.random.randint(0, union_size)
                if idx != indices[j] and idx != r1[j]:
                    r2[j] = idx
                    break

        result = np.empty((ni, dim))
        sm = strategy_mask[indices]
        mask1 = sm
        mask2 = ~sm

        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            w1, w2 = 0.7, 0.3
            direction = (w1 * (population[donor1_idx[idx1]] - population[indices[idx1]]) +
                         w2 * (population[donor2_idx[idx1]] - population[indices[idx1]]))
            diff_vec = population[r1[idx1]] - union[r2[idx1]]
            noise = np.random.normal(0, 1, (len(idx1), dim))
            rotated_noise = noise @ eigvecs.T * sqrt_eigvals[np.newaxis, :]
            rotated_noise = rotated_noise @ eigvecs
            noise_scale = 0.1 * f_col[idx1]

            result[idx1] = (population[indices[idx1]] + f_col[idx1] * direction +
                            f_col[idx1] * diff_vec + noise_scale * rotated_noise)

        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal_subset(n, indices[idx2])
            r3 = self._random_indices_not_equal_subset(n, indices[idx2])

            result[idx2] = (population[r_base] +
                            f_col[idx2] * (population[donor1_idx[idx2]] - population[r3]) +
                            f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]]))

        return result

    def _random_indices_not_equal_subset(self, n, exclude_indices):
        """Generate random indices in [0, n) that differ from exclude_indices."""
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _generate_parameters_batch(self, memory_f, memory_cr, n=None):
        """Generate F and CR values with generation-adaptive scaling."""
        if n is None:
            n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        base_scale = 0.1
        generation = getattr(self, 'generation', 0)
        scale_f = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))
        scale_cr = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))

        mu_f = memory_f[indices]
        f_values = np.zeros(n)
        remaining = np.ones(n, dtype=bool)
        max_attempts = 100
        attempt = 0

        while np.any(remaining) and attempt < max_attempts:
            count = np.sum(remaining)
            u = np.random.uniform(0, 1, size=count)
            cauchy_samples = np.tan(np.pi * (u - 0.5))
            candidates = mu_f[remaining] + scale_f * cauchy_samples

            valid = candidates > 0
            idx_remaining = np.where(remaining)[0]
            accepted = idx_remaining[valid]
            f_values[accepted] = np.minimum(candidates[valid], 1.0)
            remaining[accepted] = False
            attempt += 1

        if np.any(remaining):
            f_values[remaining] = np.clip(mu_f[remaining], 0.01, 1.0)

        cr_values = np.random.normal(memory_cr[indices], scale_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        explore_mask = np.random.random(n) < 0.1
        if np.any(explore_mask):
            n_explore = np.sum(explore_mask)
            f_values[explore_mask] = np.clip(
                np.random.uniform(0.5, 1.0, size=n_explore), 0.1, 1.0
            )
            cr_values[explore_mask] = np.clip(
                np.random.uniform(0.8, 1.0, size=n_explore), 0.0, 1.0
            )

        return f_values, cr_values

    def _select_strategies_batch(self, n=None):
        """Select mutation strategy for each individual. Returns boolean mask."""
        if n is None:
            n = self.np_size
        return np.random.random(n) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Fallback: original mutation (not used in adaptive path)."""
        n = len(population)
        dim = self.dim

        sorted_idx = np.argsort(fitness)
        p = max(2, int(np.ceil(self.p_best_rate * n)))

        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]
        r1 = self._random_indices_not_equal(n, np.arange(n))

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            mutants[idx1] = (
                population[idx1]
                + f_col[idx1] * (population[pbest_idx[idx1]] - population[idx1])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            )

        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            mutants[idx2] = (
                population[r_base]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            )

        return mutants

    def _random_indices_not_equal(self, n, exclude_indices):
        """Generate random indices in [0, n) that differ from exclude_indices."""
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to whole population at once."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds."""
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask, op_assignments=None):
        """Greedy selection with operator credit assignment."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                delta = float(abs(fitness[i] - trial_fitness[i]))
                if trial_fitness[i] < fitness[i]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(delta)

                    # Update operator stats
                    if op_assignments is not None and i < len(op_assignments):
                        op_id = int(op_assignments[i])
                        # Reward proportional to improvement (normalized)
                        reward = float(max(delta, 1e-30))
                        self._update_operator_stats(op_id, reward)
                else:
                    # Equal fitness replacement - small reward
                    if op_assignments is not None and i < len(op_assignments):
                        op_id = int(op_assignments[i])
                        self._update_operator_stats(op_id, 0.01)

                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                # Failure for this operator
                if op_assignments is not None and i < len(op_assignments) and valid_mask[i]:
                    op_id = int(op_assignments[i])
                    self._update_operator_stats(op_id, 0.0)

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    def _update_memory(self, memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta):
        """Update parameter memory using weighted Lehmer mean."""
        if len(success_f) == 0:
            return memory_f, memory_cr, memory_idx

        weights = success_delta / (np.sum(success_delta) + 1e-30)

        lehmer_f = np.sum(weights * success_f ** 2) / (np.sum(weights * success_f) + 1e-30)
        memory_f[memory_idx] = lehmer_f

        mean_cr = np.sum(weights * success_cr)
        memory_cr[memory_idx] = mean_cr

        memory_idx = (memory_idx + 1) % self.memory_size
        return memory_f, memory_cr, memory_idx

    def _detect_stagnation(self, prev_best_fitness):
        """Detect if the algorithm is stagnating."""
        improvement = prev_best_fitness - self.f_opt
        if improvement < 1e-12 * (abs(prev_best_fitness) + 1e-30):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        return self.stagnation_counter >= self.restart_threshold

    def _restart(self, func, stopping_condition, population, fitness):
        """Partial restart: keep best individuals, reinitialize rest with opposition."""
        self.stagnation_counter = 0
        n = len(population)
        keep_count = max(2, n // 5)

        sorted_idx = np.argsort(fitness)
        elite = population[sorted_idx[:keep_count]].copy()
        elite_fit = fitness[sorted_idx[:keep_count]].copy()

        reinit_count = n - keep_count
        new_pop = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

        center = np.mean(elite, axis=0)
        opp_pop = 2.0 * center - new_pop
        opp_pop = np.clip(opp_pop, self.lb, self.ub)

        mix_mask = np.random.random(reinit_count) < 0.5
        combined_new = np.where(mix_mask[:, np.newaxis], opp_pop, new_pop)

        if stopping_condition():
            population = np.vstack([elite, combined_new])
            fitness_new = np.full(reinit_count, np.inf)
            return population, np.concatenate([elite_fit, fitness_new]), np.empty((0, self.dim)), np.full(self.memory_size, 0.5), np.full(self.memory_size, 0.5), 0

        combined_new = self._clip_to_bounds(combined_new)
        new_fitness = func(combined_new)
        if len(new_fitness) < len(combined_new):
            combined_new = combined_new[:len(new_fitness)]

        self._safe_update_best(combined_new, new_fitness)

        population = np.vstack([elite, combined_new[:len(new_fitness)]])
        fitness_all = np.concatenate([elite_fit, new_fitness])

        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        archive = np.empty((0, self.dim))

        self.np_size = len(population)

        # Soft reset operator stats (keep some knowledge)
        self.op_successes = 0.5 * self.op_successes + 1.0
        self.op_failures = 0.5 * self.op_failures + 1.0

        return population, fitness_all, archive, memory_f, memory_cr, memory_idx

    def _opposition_based_jump(self, population, fitness, func, stopping_condition):
        """Apply opposition-based learning to diversify population."""
        center = (self.lb + self.ub) / 2.0
        opp_population = 2.0 * center - population
        opp_population = self._clip_to_bounds(opp_population)

        noise = np.random.normal(0, 1.0, opp_population.shape)
        opp_population = self._clip_to_bounds(opp_population + noise)

        if stopping_condition():
            return population, fitness

        opp_fitness = func(opp_population)
        if len(opp_fitness) < len(opp_population):
            opp_population = opp_population[:len(opp_fitness)]

        self._safe_update_best(opp_population, opp_fitness)

        n = min(len(population), len(opp_population))
        combined_pop = np.vstack([population[:n], opp_population[:n]])
        combined_fit = np.concatenate([fitness[:n], opp_fitness[:n]])

        sorted_idx = np.argsort(combined_fit)
        best_idx = sorted_idx[:len(population)]
        return combined_pop[best_idx], combined_fit[best_idx]

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance."""
        center = np.mean(population, axis=0)
        diffs = population - center
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _update_best(self, population, fitness):
        """Update global best from population."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fitness = fitness[valid]
        valid_pop = population[valid]
        best_idx = np.argmin(valid_fitness)
        if valid_fitness[best_idx] < self.f_opt:
            self.f_opt = float(valid_fitness[best_idx])
            self.x_opt = valid_pop[best_idx].copy()

    def _safe_update_best(self, candidates, fitnesses):
        """Safely update best considering possible NaN values."""
        if len(fitnesses) == 0:
            return
        valid = ~np.isnan(fitnesses)
        if not np.any(valid):
            return
        best_idx = np.nanargmin(fitnesses)
        if fitnesses[best_idx] < self.f_opt:
            self.f_opt = float(fitnesses[best_idx])
            self.x_opt = candidates[best_idx].copy()
