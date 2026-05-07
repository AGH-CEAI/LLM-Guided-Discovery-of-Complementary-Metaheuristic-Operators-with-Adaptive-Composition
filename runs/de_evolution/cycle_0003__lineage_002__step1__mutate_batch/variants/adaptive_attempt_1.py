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

        # Adaptive operator selection (7 strategies)
        self.n_operators = 7
        # Thompson Sampling parameters
        self.op_alpha = np.ones(self.n_operators)  # successes + 1
        self.op_beta = np.ones(self.n_operators)   # failures + 1
        # Sliding window for credit assignment
        self.window_size = 50
        self.op_rewards = [[] for _ in range(self.n_operators)]
        self.op_usage = np.zeros(self.n_operators, dtype=int)
        # Decay factor for old observations
        self.decay = 0.95

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset operator stats
        self.op_alpha = np.ones(self.n_operators)
        self.op_beta = np.ones(self.n_operators)
        self.op_rewards = [[] for _ in range(self.n_operators)]
        self.op_usage = np.zeros(self.n_operators, dtype=int)

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
            self.np_size = n

            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select operator for each individual using Thompson Sampling
            op_indices = self._select_operators_batch(n)

            # Generate mutant vectors using adaptive operator selection
            mutants = self._mutate_batch_adaptive(population, fitness, archive, f_values, op_indices)

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
                op_indices = op_indices[:len(trial_fitness)]

            valid = ~np.isnan(trial_fitness)

            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta, success_ops, fail_ops = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid, op_indices
                )
            )

            # Update operator stats
            self._update_operator_stats(success_ops, fail_ops, success_delta)

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
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _generate_parameters_batch(self, memory_f, memory_cr):
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        f_values = np.zeros(n)
        for i in range(n):
            while True:
                f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    break
            f_values[i] = min(f_val, 1.0)

        cr_values = np.clip(
            np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0
        )

        return f_values, cr_values

    def _select_operators_batch(self, n):
        """Select operator for each individual using Thompson Sampling."""
        op_indices = np.empty(n, dtype=int)
        for i in range(n):
            samples = np.random.beta(self.op_alpha, self.op_beta)
            op_indices[i] = np.argmax(samples)
        return op_indices

    def _update_operator_stats(self, success_ops, fail_ops, success_delta):
        """Update Thompson Sampling parameters with decay."""
        # Apply decay
        self.op_alpha = 1.0 + self.decay * (self.op_alpha - 1.0)
        self.op_beta = 1.0 + self.decay * (self.op_beta - 1.0)

        # Update successes weighted by improvement
        if len(success_ops) > 0 and len(success_delta) > 0:
            max_delta = np.max(success_delta) + 1e-30
            for k in range(len(success_ops)):
                op = int(success_ops[k])
                if 0 <= op < self.n_operators:
                    reward = float(success_delta[k]) / max_delta
                    self.op_alpha[op] += reward

        # Update failures
        for op in fail_ops:
            op = int(op)
            if 0 <= op < self.n_operators:
                self.op_beta[op] += 0.5

        # Keep parameters bounded to avoid extremes
        self.op_alpha = np.clip(self.op_alpha, 1.0, 50.0)
        self.op_beta = np.clip(self.op_beta, 1.0, 50.0)

    def _mutate_batch_adaptive(self, population, fitness, archive, f_values, op_indices):
        """Dispatch mutation to different strategies based on op_indices."""
        n = len(population)
        dim = self.dim
        mutants = np.empty((n, dim))

        sorted_idx = np.argsort(fitness)
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)

        p = max(2, int(np.ceil(self.p_best_rate * n)))
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Adaptive p for greedy/explorative
        p_greedy = max(2, int(np.ceil(0.05 * n)))
        p_explore = max(2, int(np.ceil(0.25 * n)))

        best_idx = sorted_idx[0]

        r1 = self._random_indices_not_equal(n, np.arange(n))

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            attempts = 0
            while attempts < 100:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break
                attempts += 1
            else:
                r2[i] = np.random.randint(0, union_size)

        # Additional random indices for strategies that need them
        r3 = self._random_indices_not_equal(n, np.arange(n))
        for i in range(n):
            if r3[i] == r1[i]:
                r3[i] = (r3[i] + 1) % n
                if r3[i] == i:
                    r3[i] = (r3[i] + 1) % n

        r4 = np.zeros(n, dtype=int)
        r5 = np.zeros(n, dtype=int)
        for i in range(n):
            used = {i, r1[i], r3[i]}
            attempts = 0
            while attempts < 100:
                idx = np.random.randint(0, union_size)
                if idx not in used:
                    r4[i] = idx
                    break
                attempts += 1
            else:
                r4[i] = np.random.randint(0, union_size)

            used.add(r4[i])
            attempts = 0
            while attempts < 100:
                idx = np.random.randint(0, n)
                if idx not in used:
                    r5[i] = idx
                    break
                attempts += 1
            else:
                r5[i] = np.random.randint(0, n)

        f_col = f_values[:, np.newaxis]

        # Rank-based scaling
        rank_ratio = ranks / max(n - 1, 1)
        f_scaled = f_values * (0.5 + rank_ratio)
        f_scaled = np.clip(f_scaled, 0.1, 1.5)
        f_scaled_col = f_scaled[:, np.newaxis]

        rank_scale = (0.5 + 0.5 * (ranks / max(n - 1, 1)))[:, np.newaxis]

        # Weighted centroid of top individuals
        n_top = max(3, n // 4)
        top_idx_arr = sorted_idx[:n_top]
        top_pop = population[top_idx_arr]
        weights = np.log(n_top + 0.5) - np.log(np.arange(1, n_top + 1))
        weights = weights / np.sum(weights)
        weighted_mean = np.sum(weights[:, np.newaxis] * top_pop, axis=0)

        # Centroid for strategy 5
        top_k = max(3, n // 3)
        top_pop_c = population[sorted_idx[:top_k]]
        weights_c = np.log(top_k + 0.5) - np.log(np.arange(1, top_k + 1))
        weights_c /= weights_c.sum()
        centroid = weights_c @ top_pop_c

        # Covariance for strategies that need it
        sqrt_cov = None
        eigvecs = np.eye(dim)
        D = np.ones(dim)
        if dim <= 100 and n_top >= max(2, dim // 2):
            try:
                diffs = top_pop - weighted_mean
                cov = np.dot((weights[:, np.newaxis] * diffs).T, diffs)
                cov += 1e-10 * np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except:
                sqrt_cov = None
                eigvecs = np.eye(dim)
                D = np.ones(dim)

        for op_id in range(self.n_operators):
            mask = (op_indices == op_id)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]

            if op_id == 0:
                # Strategy 0: variant_01 - current-to-pbest/1 with adaptive p
                pbest_local = np.empty(len(idx), dtype=int)
                for j, ii in enumerate(idx):
                    if np.random.random() < 0.7:
                        pbest_local[j] = sorted_idx[np.random.randint(0, p_greedy)]
                    else:
                        pbest_local[j] = sorted_idx[np.random.randint(0, p_explore)]
                mutants[idx] = (
                    population[idx]
                    + f_col[idx] * (population[pbest_local] - population[idx])
                    + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                )

            elif op_id == 1:
                # Strategy 1: original current-to-pbest/1
                mutants[idx] = (
                    population[idx]
                    + f_col[idx] * (population[pbest_idx[idx]] - population[idx])
                    + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                )

            elif op_id == 2:
                # Strategy 2: original rand-to-pbest/1
                r_base = self._random_indices_not_equal(n, np.arange(n))[idx]
                mutants[idx] = (
                    population[r_base]
                    + f_col[idx] * (population[pbest_idx[idx]] - population[r_base])
                    + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                )

            elif op_id == 3:
                # Strategy 3: variant_02 rank-based multi-strategy
                third = n // 3
                for ii in idx:
                    r = ranks[ii]
                    if r < third:
                        # exploitation
                        mutants[ii] = (
                            population[ii]
                            + f_scaled_col[ii].flatten() * (population[pbest_idx[ii]] - population[ii])
                            + f_scaled_col[ii].flatten() * (population[r1[ii]] - union[r2[ii]])
                        )
                    elif r < 2 * third:
                        # balanced current-to-pbest/2
                        mutants[ii] = (
                            population[ii]
                            + f_scaled_col[ii].flatten() * (population[pbest_idx[ii]] - population[ii])
                            + f_scaled_col[ii].flatten() * (population[r1[ii]] - union[r2[ii]])
                            + 0.5 * f_scaled_col[ii].flatten() * (union[r4[ii]] - population[r3[ii]])
                        )
                    else:
                        # exploration
                        rb = np.random.randint(0, n)
                        if rb == ii:
                            rb = (rb + 1) % n
                        f_explore = np.clip(f_scaled[ii] * 1.5, 0.4, 2.0)
                        mutants[ii] = (
                            population[rb]
                            + f_explore * (population[pbest_idx[ii]] - union[r2[ii]])
                            + f_explore * (population[r1[ii]] - union[r4[ii]])
                        )

            elif op_id == 4:
                # Strategy 4: variant_04 current-to-best/2 with rank scaling
                mutants[idx] = (
                    population[idx]
                    + rank_scale[idx] * f_col[idx] * (population[best_idx] - population[idx])
                    + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                    + 0.5 * f_col[idx] * (population[r3[idx]] - union[r4[idx]])
                )

            elif op_id == 5:
                # Strategy 5: variant_08 centroid-based + eigenvector rotation
                diff_vec = population[r1[idx]] - population[r3[idx]]
                if sqrt_cov is not None:
                    try:
                        rotated = (diff_vec @ eigvecs) * (D / (D.mean() + 1e-30))
                        rotated = rotated @ eigvecs.T
                    except:
                        rotated = diff_vec
                else:
                    rotated = diff_vec
                mutants[idx] = (
                    population[idx]
                    + f_col[idx] * (centroid - population[idx])
                    + 0.5 * f_col[idx] * rotated
                )

            elif op_id == 6:
                # Strategy 6: variant_10 weighted mean guidance + eigen perturbation
                base = population[idx] + f_col[idx] * (weighted_mean - population[idx])
                diff = f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                if sqrt_cov is not None:
                    z = np.random.standard_normal((len(idx), dim))
                    eigen_perturb = 0.1 * f_col[idx] * (z @ sqrt_cov)
                    use_eigen = np.random.random(len(idx)) < 0.4
                    eigen_perturb[~use_eigen] = 0.0
                    mutants[idx] = base + diff + eigen_perturb
                else:
                    mutants[idx] = base + diff

        return mutants

    def _select_strategies_batch(self):
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Fallback: Generate mutant vectors using mixed strategies."""
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
        result = np.random.randint(0, max(n - 1, 1), size=len(exclude_indices))
        if n > 1:
            mask = result >= exclude_indices
            result[mask] += 1
            result = np.clip(result, 0, n - 1)
        else:
            result[:] = 0
        return result

    def _crossover_batch(self, population, mutants, cr_values):
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask, op_indices=None):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []
        success_ops = []
        fail_ops = []

        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    delta = float(abs(fitness[i] - trial_fitness[i]))
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(delta)
                    if op_indices is not None:
                        success_ops.append(int(op_indices[i]))
                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                if op_indices is not None and valid_mask[i]:
                    fail_ops.append(int(op_indices[i]))

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta),
                np.array(success_ops, dtype=int) if len(success_ops) > 0 else np.array([], dtype=int),
                np.array(fail_ops, dtype=int) if len(fail_ops) > 0 else np.array([], dtype=int))

    def _update_memory(self, memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta):
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
        improvement = prev_best_fitness - self.f_opt
        if improvement < 1e-12 * (abs(prev_best_fitness) + 1e-30):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        return self.stagnation_counter >= self.restart_threshold

    def _restart(self, func, stopping_condition, population, fitness):
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

        # Partially reset operator stats on restart (keep some memory)
        self.op_alpha = 1.0 + 0.5 * (self.op_alpha - 1.0)
        self.op_beta = 1.0 + 0.5 * (self.op_beta - 1.0)

        return population, fitness_all, archive, memory_f, memory_cr, memory_idx

    def _opposition_based_jump(self, population, fitness, func, stopping_condition):
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
        center = np.mean(population, axis=0)
        diffs = population - center
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _update_best(self, population, fitness):
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
        if len(fitnesses) == 0:
            return
        valid = ~np.isnan(fitnesses)
        if not np.any(valid):
            return
        best_idx = np.nanargmin(fitnesses)
        if fitnesses[best_idx] < self.f_opt:
            self.f_opt = float(fitnesses[best_idx])
            self.x_opt = candidates[best_idx].copy()
