import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, intelligent restart mechanism, and adaptive
    crossover operator selection using sliding window credit assignment.
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

        # Adaptive crossover operator selection
        self.num_operators = 5  # number of crossover strategies
        self.op_window_size = 50  # sliding window size
        self.op_rewards = [[] for _ in range(self.num_operators)]
        self.op_usage = np.zeros(self.num_operators, dtype=int)
        self.op_successes = np.zeros(self.num_operators, dtype=float)
        self.op_alpha = np.ones(self.num_operators, dtype=float)  # Thompson sampling params
        self.op_beta = np.ones(self.num_operators, dtype=float)
        self._eigenvectors = None
        self._eigenvalues = None
        self._cov_center = None
        self._cov_valid = False

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset operator stats
        self.op_alpha = np.ones(self.num_operators, dtype=float)
        self.op_beta = np.ones(self.num_operators, dtype=float)

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
            # Update covariance decomposition periodically
            if self.generation % 5 == 0:
                self._update_covariance(population)

            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch()

            # Generate mutant vectors
            mutants = self._mutate_batch(population, fitness, archive, f_values, strategy_mask)

            # Select crossover operator via Thompson Sampling
            op_idx = self._select_crossover_operator()

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values, op_idx)

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

            valid = ~np.isnan(trial_fitness)

            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Update operator reward
            self._update_operator_reward(op_idx, success_delta, len(trial_fitness))

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

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _update_covariance(self, population):
        """Update eigendecomposition of population covariance."""
        n, dim = population.shape
        try:
            if dim < 3 or n < dim // 2:
                self._cov_valid = False
                return
            if dim > 200:
                self._cov_valid = False
                return
            center = np.mean(population, axis=0)
            centered = population - center
            cov = np.dot(centered.T, centered) / max(n - 1, 1)
            cov += 1e-10 * np.eye(dim)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            self._eigenvalues = eigenvalues
            self._eigenvectors = eigenvectors
            self._cov_center = center
            self._cov_valid = True
        except Exception:
            self._cov_valid = False

    def _select_crossover_operator(self):
        """Select crossover operator using Thompson Sampling."""
        samples = np.random.beta(self.op_alpha, self.op_beta)
        return int(np.argmax(samples))

    def _update_operator_reward(self, op_idx, success_delta, n_trials):
        """Update Thompson Sampling parameters based on operator performance."""
        if n_trials == 0:
            return
        # Compute success rate as reward signal
        total_improvement = float(np.sum(success_delta)) if len(success_delta) > 0 else 0.0
        n_successes = len(success_delta)
        success_rate = float(n_successes) / float(n_trials)

        # Update with decay to allow adaptation
        decay = 0.995
        self.op_alpha *= decay
        self.op_beta *= decay

        # Ensure minimums
        self.op_alpha = np.maximum(self.op_alpha, 0.1)
        self.op_beta = np.maximum(self.op_beta, 0.1)

        # Update selected operator
        # Use both success rate and improvement magnitude
        reward = success_rate + min(1.0, total_improvement / (abs(self.f_opt) + 1e-30))
        reward = float(np.clip(reward, 0.0, 2.0))

        self.op_alpha[op_idx] += reward
        self.op_beta[op_idx] += (2.0 - reward)

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values with generation-adaptive scaling."""
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

    def _select_strategies_batch(self):
        """Select mutation strategy for each individual."""
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using mixed strategies."""
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

    def _crossover_batch(self, population, mutants, cr_values, op_idx=0):
        """Adaptive crossover: dispatches to selected operator."""
        if op_idx == 0:
            return self._crossover_original(population, mutants, cr_values)
        elif op_idx == 1:
            return self._crossover_eigenvector_block(population, mutants, cr_values)
        elif op_idx == 2:
            return self._crossover_eigenvector_binomial(population, mutants, cr_values)
        elif op_idx == 3:
            return self._crossover_rotated_exponential(population, mutants, cr_values)
        elif op_idx == 4:
            return self._crossover_block_sectored(population, mutants, cr_values)
        else:
            return self._crossover_original(population, mutants, cr_values)

    def _crossover_original(self, population, mutants, cr_values):
        """Original binomial crossover."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True
        trials = np.where(cross_mask, mutants, population)
        return trials

    def _crossover_eigenvector_block(self, population, mutants, cr_values):
        """Eigenvector-guided rotational crossover with block transfer (variant_06 style)."""
        n, dim = population.shape

        if not self._cov_valid or dim < 3:
            return self._crossover_original(population, mutants, cr_values)

        try:
            eigenvectors = self._eigenvectors
            pop_rot = population @ eigenvectors
            mut_rot = mutants @ eigenvectors

            trials_rot = pop_rot.copy()

            for i in range(n):
                cr = cr_values[i]
                block_size = max(1, int(np.ceil(cr * dim * 0.5)))
                start = np.random.randint(0, dim)
                mask = np.zeros(dim, dtype=bool)
                indices = np.arange(start, start + block_size) % dim
                mask[indices] = True
                rand_vals = np.random.random(dim)
                scatter_mask = rand_vals < (cr * 0.3)
                mask = mask | scatter_mask
                if not np.any(mask):
                    mask[np.random.randint(0, dim)] = True
                trials_rot[i, mask] = mut_rot[i, mask]

            trials = trials_rot @ eigenvectors.T
            trials = np.clip(trials, self.lb, self.ub)
            return trials
        except Exception:
            return self._crossover_original(population, mutants, cr_values)

    def _crossover_eigenvector_binomial(self, population, mutants, cr_values):
        """Eigenvector-based binomial crossover (variant_08 style)."""
        n, dim = population.shape

        if not self._cov_valid:
            return self._crossover_original(population, mutants, cr_values)

        try:
            eigenvectors = self._eigenvectors
            eigenvalues = self._eigenvalues
            center = self._cov_center

            pop_rot = np.dot(population - center, eigenvectors)
            mut_rot = np.dot(mutants - center, eigenvectors)

            rand_matrix = np.random.random((n, dim))
            cr_matrix = cr_values[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True

            trials_rot = np.where(cross_mask, mut_rot, pop_rot)

            # Block crossover along top eigenvectors for some individuals
            block_mask = np.random.random(n) < 0.15
            if np.any(block_mask) and dim > 2:
                block_idx = np.where(block_mask)[0]
                k = max(1, dim // 3)
                top_k = np.argsort(eigenvalues)[-k:]
                for i in block_idx:
                    trial_rot_i = pop_rot[i].copy()
                    trial_rot_i[top_k] = mut_rot[i][top_k]
                    trials_rot[i] = trial_rot_i

            trials = np.dot(trials_rot, eigenvectors.T) + center
            trials = np.clip(trials, self.lb, self.ub)
            return trials
        except Exception:
            return self._crossover_original(population, mutants, cr_values)

    def _crossover_rotated_exponential(self, population, mutants, cr_values):
        """Covariance-rotated exponential crossover (variant_04 style)."""
        n, dim = population.shape

        if not self._cov_valid:
            # Fallback: exponential crossover in original space
            trials = population.copy()
            for i in range(n):
                cr = cr_values[i]
                j_start = np.random.randint(0, dim)
                L = 0
                while L < dim and (np.random.random() < cr or L == 0):
                    j = (j_start + L) % dim
                    trials[i, j] = mutants[i, j]
                    L += 1
            return trials

        try:
            eigenvectors = self._eigenvectors
            center = self._cov_center

            pop_rot = np.dot(population - center, eigenvectors)
            mut_rot = np.dot(mutants - center, eigenvectors)

            trials_rot = pop_rot.copy()
            for i in range(n):
                cr = cr_values[i]
                j_start = np.random.randint(0, dim)
                L = 0
                while L < dim and (np.random.random() < cr or L == 0):
                    j = (j_start + L) % dim
                    trials_rot[i, j] = mut_rot[i, j]
                    L += 1

            trials = np.dot(trials_rot, eigenvectors.T) + center

            # Mix with standard binomial for diversity
            mix_prob = 0.2
            use_standard = np.random.random(n) < mix_prob
            if np.any(use_standard):
                idx = np.where(use_standard)[0]
                rand_matrix = np.random.random((len(idx), dim))
                cr_matrix = cr_values[idx, np.newaxis]
                j_rand = np.random.randint(0, dim, size=len(idx))
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(len(idx)), j_rand] = True
                standard_trials = np.where(cross_mask, mutants[idx], population[idx])
                trials[idx] = standard_trials

            trials = np.clip(trials, self.lb, self.ub)
            return trials
        except Exception:
            return self._crossover_original(population, mutants, cr_values)

    def _crossover_block_sectored(self, population, mutants, cr_values):
        """Block-sectored crossover (variant_10 style)."""
        n, dim = population.shape
        trials = population.copy()

        for i in range(n):
            cr = cr_values[i]
            inject_random = np.random.random() < 0.05

            if dim <= 4:
                rand_vals = np.random.random(dim)
                j_rand = np.random.randint(0, dim)
                mask = rand_vals < cr
                mask[j_rand] = True
                trials[i] = np.where(mask, mutants[i], population[i])
            else:
                n_blocks = max(2, int(np.sqrt(dim)))
                perm = np.random.permutation(dim)
                splits = np.array_split(perm, n_blocks)

                forced_block = np.random.randint(0, len(splits))

                for b_idx, block_dims in enumerate(splits):
                    if len(block_dims) == 0:
                        continue
                    take_from_mutant = (np.random.random() < cr) or (b_idx == forced_block)

                    if take_from_mutant:
                        if inject_random and np.random.random() < 0.3:
                            trials[i, block_dims] = np.random.uniform(self.lb, self.ub, size=len(block_dims))
                        else:
                            trials[i, block_dims] = mutants[i, block_dims]

                if np.random.random() < 0.03 and self.x_opt is not None:
                    scale = 0.01 * (self.ub - self.lb) * np.random.random()
                    perturbation = np.random.normal(0, scale, dim)
                    candidate = self.x_opt + perturbation
                    candidate = np.clip(candidate, self.lb, self.ub)
                    trials[i] = candidate

        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds."""
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """Greedy selection: replace if trial is better. Track successful params."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(fitness[i] - trial_fitness[i])))
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

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
