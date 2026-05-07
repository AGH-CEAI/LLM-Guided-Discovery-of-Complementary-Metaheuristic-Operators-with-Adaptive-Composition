

```python
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

        # Adaptive operator selection setup
        self.num_operators = 6  # number of crossover strategies
        self.operator_window_size = 50  # sliding window size
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        self.operator_counts = np.ones(self.num_operators, dtype=float)
        self.operator_successes = np.zeros(self.num_operators, dtype=float)
        self.operator_alpha = np.ones(self.num_operators, dtype=float)  # Thompson sampling alpha
        self.operator_beta = np.ones(self.num_operators, dtype=float)   # Thompson sampling beta
        self.current_operator = 0

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset operator tracking
        self.operator_alpha = np.ones(self.num_operators, dtype=float)
        self.operator_beta = np.ones(self.num_operators, dtype=float)

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
            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch()

            # Generate mutant vectors
            mutants = self._mutate_batch(population, fitness, archive, f_values, strategy_mask)

            # Select crossover operator via Thompson Sampling
            self.current_operator = self._select_operator()

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

            valid = ~np.isnan(trial_fitness)

            # Compute reward for operator before selection modifies fitness
            reward = self._compute_operator_reward(fitness, trial_fitness, valid)
            self._update_operator_stats(self.current_operator, reward)

            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
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

    def _select_operator(self):
        """Select crossover operator using Thompson Sampling."""
        samples = np.array([
            np.random.beta(max(self.operator_alpha[i], 0.01), max(self.operator_beta[i], 0.01))
            for i in range(self.num_operators)
        ])
        return int(np.argmax(samples))

    def _compute_operator_reward(self, old_fitness, new_fitness, valid_mask):
        """Compute reward as the fraction of improved individuals weighted by improvement magnitude."""
        n = min(len(old_fitness), len(new_fitness))
        if n == 0:
            return 0.0
        valid = valid_mask[:n]
        if not np.any(valid):
            return 0.0
        improvements = old_fitness[:n][valid] - new_fitness[:n][valid]
        n_improved = float(np.sum(improvements > 0))
        total_improvement = float(np.sum(np.maximum(improvements, 0)))
        # Normalize reward to [0, 1] range
        reward = n_improved / max(float(np.sum(valid)), 1.0)
        return float(reward)

    def _update_operator_stats(self, op_idx, reward):
        """Update Thompson Sampling parameters for the selected operator."""
        reward = float(reward)
        # Decay existing stats slightly for adaptivity
        decay = 0.995
        self.operator_alpha *= decay
        self.operator_beta *= decay
        # Ensure minimums
        self.operator_alpha = np.maximum(self.operator_alpha, 0.5)
        self.operator_beta = np.maximum(self.operator_beta, 0.5)

        # Update selected operator
        if reward > 0.1:
            self.operator_alpha[op_idx] += reward * 2.0
        else:
            self.operator_beta[op_idx] += (1.0 - reward)

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

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

    def _crossover_batch(self, population, mutants, cr_values):
        """Adaptive crossover that selects among multiple strategies."""
        op = self.current_operator
        if op == 0:
            return self._crossover_standard_binomial(population, mutants, cr_values)
        elif op == 1:
            return self._crossover_eigenvector(population, mutants, cr_values)
        elif op == 2:
            return self._crossover_eigenvector_boosted(population, mutants, cr_values)
        elif op == 3:
            return self._crossover_exponential_rotated(population, mutants, cr_values)
        elif op == 4:
            return self._crossover_eigenvector_segment(population, mutants, cr_values)
        elif op == 5:
            return self._crossover_exponential_directional(population, mutants, cr_values)
        else:
            return self._crossover_standard_binomial(population, mutants, cr_values)

    def _crossover_standard_binomial(self, population, mutants, cr_values):
        """Original: standard binomial crossover."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _crossover_eigenvector(self, population, mutants, cr_values):
        """Eigenvector-based crossover in rotated coordinate system (variant_05)."""
        n, dim = population.shape
        cr_matrix = cr_values[:, np.newaxis]

        try:
            if n > dim + 1 and dim >= 2:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                cov += 1e-12 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                pop_rotated = np.dot(population - center, eigenvectors)
                mut_rotated = np.dot(mutants - center, eigenvectors)

                rand_matrix = np.random.random((n, dim))
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(n), j_rand] = True

                trial_rotated = np.where(cross_mask, mut_rotated, pop_rotated)
                trials = np.dot(trial_rotated, eigenvectors.T) + center

                bad = np.any(~np.isfinite(trials), axis=1)
                if np.any(bad):
                    rand_matrix2 = np.random.random((np.sum(bad), dim))
                    j_rand2 = np.random.randint(0, dim, size=np.sum(bad))
                    cross_mask2 = rand_matrix2 < cr_matrix[bad]
                    cross_mask2[np.arange(np.sum(bad)), j_rand2] = True
                    trials[bad] = np.where(cross_mask2, mutants[bad], population[bad])

                return trials
            else:
                raise ValueError("Not enough population for covariance")
        except (np.linalg.LinAlgError, ValueError):
            return self._crossover_standard_binomial(population, mutants, cr_values)

    def _crossover_eigenvector_boosted(self, population, mutants, cr_values):
        """Eigenvector-rotated crossover with boosted CR and mixed exponential (variant_08)."""
        n, dim = population.shape

        try:
            if n > dim and dim > 1:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                use_rotation = True
            else:
                use_rotation = False
        except Exception:
            use_rotation = False

        generation = getattr(self, 'generation', 0)
        cr_boost = min(0.3, generation / 500.0)
        boosted_cr = np.clip(cr_values + cr_boost, 0.0, 1.0)

        if use_rotation and dim > 1:
            pop_rotated = np.dot(population - center, eigenvectors)
            mut_rotated = np.dot(mutants - center, eigenvectors)

            rand_matrix = np.random.random((n, dim))
            cr_matrix = boosted_cr[:, np.newaxis]

            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True

            exp_mask = np.random.random(n) < 0.2
            if np.any(exp_mask):
                exp_idx = np.where(exp_mask)[0]
                for i in exp_idx:
                    L = np.random.randint(0, dim)
                    cross_mask[i, :] = False
                    j = L
                    while True:
                        cross_mask[i, j % dim] = True
                        j += 1
                        if np.random.random() >= boosted_cr[i] or (j - L) >= dim:
                            break

            trial_rotated = np.where(cross_mask, mut_rotated, pop_rotated)
            trials = np.dot(trial_rotated, eigenvectors.T) + center
        else:
            rand_matrix = np.random.random((n, dim))
            cr_matrix = boosted_cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)

        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _crossover_exponential_rotated(self, population, mutants, cr_values):
        """Exponential crossover with eigenvector rotation (variant_06)."""
        n, dim = population.shape

        try:
            if n > dim and dim <= 100 and dim >= 2:
                centered = population - np.mean(population, axis=0)
                cov = np.cov(centered, rowvar=False)
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                pop_rot = population @ eigenvectors
                mut_rot = mutants @ eigenvectors
                use_rotation = True
            else:
                pop_rot = population
                mut_rot = mutants
                use_rotation = False
        except Exception:
            pop_rot = population
            mut_rot = mutants
            use_rotation = False

        trials_rot = pop_rot.copy()

        for i in range(n):
            cr = cr_values[i]

            if np.random.random() < 0.15:
                trials_rot[i] = mut_rot[i]
                continue

            start = np.random.randint(0, dim)
            L = 0
            while L < dim:
                trials_rot[i, (start + L) % dim] = mut_rot[i, (start + L) % dim]
                L += 1
                if np.random.random() >= cr:
                    break

        if use_rotation:
            trials = trials_rot @ eigenvectors.T
        else:
            trials = trials_rot

        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _crossover_eigenvector_segment(self, population, mutants, cr_values):
        """Eigenvector-guided crossover with segment-based approach (variant_10)."""
        n, dim = population.shape
        cr_matrix = cr_values[:, np.newaxis]

        trials = population.copy()
        use_rotated = dim >= 2 and n >= dim + 1

        if use_rotated:
            try:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                cov += 1e-10 * np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(cov)

                pop_rot = np.dot(population - center, eigvecs)
                mut_rot = np.dot(mutants - center, eigvecs)

                for i in range(n):
                    cr_i = cr_values[i]
                    trial_rot = pop_rot[i].copy()
                    j_start = np.random.randint(0, dim)

                    seg_len = min(np.random.geometric(p=max(0.1, 1.0 - cr_i + 0.01)), dim)
                    indices = [(j_start + k) % dim for k in range(seg_len)]
                    trial_rot[indices] = mut_rot[i][indices]

                    remaining = np.ones(dim, dtype=bool)
                    remaining[indices] = False
                    rand_vals = np.random.random(dim)
                    binomial_mask = (rand_vals < cr_i * 0.5) & remaining
                    trial_rot[binomial_mask] = mut_rot[i][binomial_mask]

                    trials[i] = np.dot(trial_rot, eigvecs.T) + center

            except (np.linalg.LinAlgError, ValueError):
                rand_matrix = np.random.random((n, dim))
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(n), j_rand] = True
                trials = np.where(cross_mask, mutants, population)
        else:
            for i in range(n):
                L = 0
                j = np.random.randint(0, dim)
                while np.random.random() < cr_values[i] and L < dim:
                    trials[i, j] = mutants[i, j]
                    j = (j + 1) % dim
                    L += 1
                if L == 0:
                    trials[i, j] = mutants[i, j]

        return trials

    def _crossover_exponential_directional(self, population, mutants, cr_values):
        """Exponential crossover with directional perturbation (variant_04)."""
        n, dim = population.shape
        trials = population.copy()

        for i in range(n):
            cr = cr_values[i]
            start = np.random.randint(0, dim)
            L = 0
            while True:
                L += 1
                if L >= dim or np.random.random() >= cr:
                    break

            indices = [(start + j) % dim for j in range(L)]
            trials[i, indices] = mutants[i, indices]

        direction = mutants - population
        dir_norms = np.sqrt(np.sum(direction ** 2, axis=1, keepdims=True)) + 1e-30
        direction_normalized = direction / dir_norms

        generation = getattr(self, 'generation', 0)
        base_scale = 0.05 * (self.ub - self.lb)
        scale = base_scale / (1.0 + generation / 100.0)

        perturb_mask = np.random.random(n) < 0.3
        if np.any(perturb_mask):
            n_perturb = np.sum(perturb_mask)
            magnitudes = np.random.exponential(scale, size=(n_perturb, 1))
            random_component = np.random.normal(0, scale * 0.1, size=(n_perturb, dim))

            trials[perturb_mask] = (
                trials[perturb_mask]
                + magnitudes * direction_normalized[perturb_mask]
                + random_component
            )

        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds."""
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """Greedy selection: replace if trial is better."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(float(fitness[i]) - float(trial_fitness[i])))
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
```