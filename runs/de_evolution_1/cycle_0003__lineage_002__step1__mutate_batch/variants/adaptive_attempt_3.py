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

        # Adaptive operator selection: 7 mutation strategies
        self.n_operators = 7
        # Thompson Sampling parameters
        self.op_alpha = np.ones(self.n_operators)  # successes + 1
        self.op_beta = np.ones(self.n_operators)   # failures + 1
        # Sliding window for credit assignment
        self.window_size = 50
        self.op_history = []  # list of (op_idx, reward)
        # Decay factor for old observations
        self.decay = 0.995

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset operator stats
        self.op_alpha = np.ones(self.n_operators)
        self.op_beta = np.ones(self.n_operators)
        self.op_history = []

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

            # Select mutation operator for each individual using Thompson Sampling
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

            # Selection and archive update (with operator tracking)
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch_adaptive(
                    population, fitness, trials, trial_fitness, archive,
                    f_values, cr_values, valid, op_indices
                )
            )

            # Update parameter memory
            memory_f, memory_cr, memory_idx = self._update_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta
            )

            self._update_best(population, fitness)

            # Decay operator statistics
            self._decay_operator_stats()

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

    def _decay_operator_stats(self):
        """Apply decay to operator statistics to forget old performance."""
        self.op_alpha = 1.0 + (self.op_alpha - 1.0) * self.decay
        self.op_beta = 1.0 + (self.op_beta - 1.0) * self.decay
        # Ensure minimum values
        self.op_alpha = np.maximum(self.op_alpha, 1.0)
        self.op_beta = np.maximum(self.op_beta, 1.0)

    def _update_operator_stats(self, op_idx, reward):
        """Update Thompson Sampling parameters for an operator."""
        op_idx = int(op_idx)
        reward = float(reward)
        if reward > 0:
            self.op_alpha[op_idx] += reward
        else:
            self.op_beta[op_idx] += 1.0

    def _mutate_batch_adaptive(self, population, fitness, archive, f_values, op_indices):
        """Generate mutant vectors using adaptively selected operators."""
        n = len(population)
        dim = self.dim
        mutants = np.empty((n, dim))

        sorted_idx = np.argsort(fitness)
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)

        p = max(2, int(np.ceil(self.p_best_rate * n)))

        # Precompute common data
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Precompute random indices
        arange_n = np.arange(n)
        pbest_idx_greedy = sorted_idx[np.random.randint(0, max(2, int(np.ceil(0.05 * n))), size=n)]
        pbest_idx_explore = sorted_idx[np.random.randint(0, max(2, int(np.ceil(0.25 * n))), size=n)]
        pbest_idx_normal = sorted_idx[np.random.randint(0, p, size=n)]

        r1 = self._random_indices_not_equal(n, arange_n)
        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        r3 = self._random_indices_not_equal(n, r1)
        for i in range(n):
            while r3[i] == i:
                r3[i] = np.random.randint(0, n)

        r4 = np.zeros(n, dtype=int)
        r5 = np.zeros(n, dtype=int)
        for i in range(n):
            attempts = 0
            while attempts < 50:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i] and idx != r2[i]:
                    r4[i] = idx
                    break
                attempts += 1
            else:
                r4[i] = np.random.randint(0, union_size)

            attempts = 0
            while attempts < 50:
                idx = np.random.randint(0, n)
                if idx != i and idx != r1[i] and idx != r3[i]:
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

        # Best individual
        best_idx = sorted_idx[0]

        # Weighted centroid of top individuals
        n_top = max(3, n // 4)
        top_idx_arr = sorted_idx[:n_top]
        top_pop = population[top_idx_arr]
        weights_top = np.log(n_top + 0.5) - np.log(np.arange(1, n_top + 1))
        weights_top = weights_top / np.sum(weights_top)
        weighted_mean = np.sum(weights_top[:, np.newaxis] * top_pop, axis=0)

        # Centroid for operator 5
        top_k = max(3, n // 3)
        top_pop_k = population[sorted_idx[:top_k]]
        weights_k = np.log(top_k + 0.5) - np.log(np.arange(1, top_k + 1))
        weights_k /= weights_k.sum()
        centroid = weights_k @ top_pop_k

        # Covariance for operator 5
        try:
            diff_cov = top_pop_k - centroid
            cov = (diff_cov * weights_k[:, np.newaxis]).T @ diff_cov
            cov += 1e-10 * np.eye(dim)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            has_cov = True
        except:
            eigvecs = np.eye(dim)
            D = np.ones(dim)
            has_cov = False

        # Covariance for operator 3 (rotation-aware)
        try:
            n_elite = max(4, n // 2)
            elite_pop = population[sorted_idx[:n_elite]]
            cov2 = np.cov(elite_pop.T) + 1e-10 * np.eye(dim)
            eigvals2, eigvecs2 = np.linalg.eigh(cov2)
            eigvals2 = np.maximum(eigvals2, 1e-20)
            sqrt_eigvals2 = np.sqrt(eigvals2 / np.max(eigvals2))
            transform = eigvecs2 * sqrt_eigvals2[np.newaxis, :]
            inv_transform = np.linalg.inv(transform + 1e-12 * np.eye(dim))
            has_transform = True
        except:
            transform = np.eye(dim)
            inv_transform = np.eye(dim)
            has_transform = False

        # Sqrt cov for operator 6
        try:
            if dim <= 100 and n_top >= dim // 2:
                diffs_sc = top_pop - weighted_mean
                cov_sc = np.dot((weights_top[:, np.newaxis] * diffs_sc).T, diffs_sc)
                cov_sc += 1e-10 * np.eye(dim)
                eigvals_sc, eigvecs_sc = np.linalg.eigh(cov_sc)
                eigvals_sc = np.maximum(eigvals_sc, 1e-20)
                sqrt_cov = eigvecs_sc @ np.diag(np.sqrt(eigvals_sc)) @ eigvecs_sc.T
                has_sqrt_cov = True
            else:
                has_sqrt_cov = False
                sqrt_cov = None
        except:
            has_sqrt_cov = False
            sqrt_cov = None

        # Process each operator
        for op in range(self.n_operators):
            mask = op_indices == op
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]

            if op == 0:
                # Operator 0: Original current-to-pbest/1 (from original.py)
                mutants[idx] = (
                    population[idx]
                    + f_col[idx] * (population[pbest_idx_normal[idx]] - population[idx])
                    + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                )

            elif op == 1:
                # Operator 1: variant_01 - adaptive p greediness current-to-pbest/1
                # Use greedy pbest for top half, explore for bottom half
                pbest_sel = np.where(ranks[idx] < n // 2,
                                     pbest_idx_greedy[idx],
                                     pbest_idx_explore[idx])
                mutants[idx] = (
                    population[idx]
                    + f_col[idx] * (population[pbest_sel] - population[idx])
                    + f_col[idx] * (population[r1[idx]] - union[r2[idx]])
                )

            elif op == 2:
                # Operator 2: variant_02 - rank-based multi-strategy
                third = n // 3
                for ii in idx:
                    if ranks[ii] < third:
                        # Exploitation
                        mutants[ii] = (
                            population[ii]
                            + f_scaled_col[ii] * (population[pbest_idx_normal[ii]] - population[ii])
                            + f_scaled_col[ii] * (population[r1[ii]] - union[r2[ii]])
                        )
                    elif ranks[ii] < 2 * third:
                        # Balanced with extra diff
                        mutants[ii] = (
                            population[ii]
                            + f_scaled_col[ii] * (population[pbest_idx_normal[ii]] - population[ii])
                            + f_scaled_col[ii] * (population[r1[ii]] - union[r2[ii]])
                            + 0.5 * f_scaled_col[ii] * (population[r3[ii]] - union[r4[ii]])
                        )
                    else:
                        # Exploration
                        r_base_ii = np.random.randint(0, n)
                        if r_base_ii == ii:
                            r_base_ii = (r_base_ii + 1) % n
                        f_explore = np.clip(f_scaled[ii] * 1.5, 0.4, 2.0)
                        mutants[ii] = (
                            population[r_base_ii]
                            + f_explore * (population[pbest_idx_normal[ii]] - union[r2[ii]])
                            + f_explore * (population[r1[ii]] - population[r3[ii]])
                        )

            elif op == 3:
                # Operator 3: variant_07 - covariance-rotated mutation
                for ii in idx:
                    if ranks[ii] < n // 2:
                        diff1 = population[pbest_idx_normal[ii]] - population[ii]
                        diff2 = population[r1[ii]] - union[r2[ii]]
                        diff3 = population[r3[ii]] - population[ii]
                        if has_transform:
                            rot_diff = (diff1 + 0.5 * diff2) @ inv_transform @ transform
                        else:
                            rot_diff = diff1 + 0.5 * diff2
                        mutants[ii] = population[ii] + f_scaled[ii] * rot_diff + 0.3 * f_scaled[ii] * diff3
                    else:
                        r_base_ii = np.random.randint(0, n)
                        if r_base_ii == ii:
                            r_base_ii = (r_base_ii + 1) % n
                        diff1 = population[pbest_idx_normal[ii]] - population[r_base_ii]
                        diff2 = population[r1[ii]] - union[r2[ii]]
                        noise = np.random.normal(0, 0.1, dim) @ transform
                        mutants[ii] = population[r_base_ii] + f_scaled[ii] * (diff1 + diff2) + noise

            elif op == 4:
                # Operator 4: variant_04 - current-to-best/2 with rank scaling
                for ii in idx:
                    if np.random.random() < 0.5:
                        # current-to-best/2
                        mutants[ii] = (
                            population[ii]
                            + rank_scale[ii] * f_col[ii].flatten() * (population[best_idx] - population[ii])
                            + f_col[ii].flatten() * (population[r1[ii]] - union[r2[ii]])
                            + 0.5 * f_col[ii].flatten() * (population[r3[ii]] - union[r4[ii]])
                        )
                    else:
                        # rand-to-pbest/2
                        r_base_ii = np.random.randint(0, n)
                        if r_base_ii == ii:
                            r_base_ii = (r_base_ii + 1) % n
                        mutants[ii] = (
                            population[r_base_ii]
                            + rank_scale[ii] * f_col[ii].flatten() * (population[pbest_idx_normal[ii]] - population[r_base_ii])
                            + f_col[ii].flatten() * (population[r1[ii]] - union[r2[ii]])
                            + 0.5 * f_col[ii].flatten() * (population[r3[ii]] - union[r4[ii]])
                        )

            elif op == 5:
                # Operator 5: variant_08 - centroid-based with eigenvector rotation
                for ii in idx:
                    if ranks[ii] < n // 2:
                        diff_vec = population[r1[ii]] - union[r2[ii]]
                        if has_cov:
                            rotated = (diff_vec @ eigvecs) * (D / (D.mean() + 1e-30))
                            rotated = rotated @ eigvecs.T
                        else:
                            rotated = diff_vec
                        mutants[ii] = (
                            population[ii]
                            + f_values[ii] * (centroid - population[ii])
                            + 0.5 * f_values[ii] * rotated
                        )
                    else:
                        mutants[ii] = (
                            population[ii]
                            + f_values[ii] * (population[pbest_idx_normal[ii]] - population[ii])
                            + 0.5 * f_values[ii] * (population[r1[ii]] - union[r2[ii]])
                            + 0.5 * f_values[ii] * (union[r2[ii]] - population[r3[ii]])
                        )

            elif op == 6:
                # Operator 6: variant_10 - weighted mean guidance + eigen perturbation
                for ii in idx:
                    if ranks[ii] < n // 2:
                        base = population[ii] + f_values[ii] * (weighted_mean - population[ii])
                        diff = f_values[ii] * (population[r1[ii]] - union[r2[ii]])
                        if has_sqrt_cov and np.random.random() < 0.4:
                            z = np.random.standard_normal(dim)
                            eigen_perturb = 0.1 * f_values[ii] * sqrt_cov @ z
                            mutants[ii] = base + diff + eigen_perturb
                        else:
                            mutants[ii] = base + diff
                    else:
                        mutants[ii] = (
                            population[ii]
                            + f_values[ii] * (population[pbest_idx_normal[ii]] - population[ii])
                            + f_values[ii] * (population[r1[ii]] - union[r2[ii]])
                            + 0.5 * f_values[ii] * (population[r3[ii]] - union[r4[ii]])
                        )

        return mutants

    def _random_indices_not_equal(self, n, exclude_indices):
        result = np.random.randint(0, max(n - 1, 1), size=len(exclude_indices))
        if n > 1:
            mask = result >= exclude_indices
            result[mask] += 1
            result = np.clip(result, 0, n - 1)
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

    def _select_survivors_batch_adaptive(self, population, fitness, trials, trial_fitness,
                                          archive, f_values, cr_values, valid_mask, op_indices):
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
                    # Reward operator: use normalized improvement
                    reward = float(delta / (abs(fitness[i]) + 1e-30))
                    reward = min(reward, 10.0)  # cap reward
                    self._update_operator_stats(op_indices[i], reward)
                else:
                    # Tie: small reward
                    self._update_operator_stats(op_indices[i], 0.1)

                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                # Failure for the operator
                if i < len(op_indices):
                    self._update_operator_stats(op_indices[i], 0.0)

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))
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

        # Partially reset operator stats on restart (soft reset)
        self.op_alpha = 1.0 + (self.op_alpha - 1.0) * 0.5
        self.op_beta = 1.0 + (self.op_beta - 1.0) * 0.5

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
