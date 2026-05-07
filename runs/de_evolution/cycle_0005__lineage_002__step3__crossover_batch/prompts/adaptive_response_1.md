

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
        self.num_operators = 7  # number of crossover strategies
        self.op_window_size = 50  # sliding window size
        self.op_rewards = [[] for _ in range(self.num_operators)]
        self.op_successes = np.zeros(self.num_operators)
        self.op_trials_count = np.zeros(self.num_operators)
        self.op_alpha = np.ones(self.num_operators)  # Thompson sampling alpha
        self.op_beta = np.ones(self.num_operators)   # Thompson sampling beta
        self.op_min_prob = 0.02  # minimum selection probability per operator

        # Eigenvector cache
        self._eigenvectors = None
        self._eigenvalues = None
        self._eigen_center = None
        self._eigen_gen = -10  # last generation eigenvectors were computed

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset operator tracking
        self.op_alpha = np.ones(self.num_operators)
        self.op_beta = np.ones(self.num_operators)
        self.op_successes = np.zeros(self.num_operators)
        self.op_trials_count = np.zeros(self.num_operators)
        self.op_rewards = [[] for _ in range(self.num_operators)]

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

            # Select crossover operator for each individual
            op_indices = self._select_crossover_operators(len(population))

            # Crossover with adaptive operator selection
            trials = self._crossover_batch_adaptive(population, mutants, cr_values, op_indices)

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
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Update operator credits based on success/failure
            self._update_operator_credits(op_indices, trial_fitness, fitness, valid, len(trial_fitness))

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

            valid_c = candidates > 0
            idx_remaining = np.where(remaining)[0]
            accepted = idx_remaining[valid_c]
            f_values[accepted] = np.minimum(candidates[valid_c], 1.0)
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
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
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
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    # ---- Adaptive Operator Selection ----

    def _select_crossover_operators(self, n):
        """Select crossover operator for each individual using Thompson Sampling."""
        # Sample from Beta distributions
        samples = np.array([
            np.random.beta(max(self.op_alpha[i], 0.01), max(self.op_beta[i], 0.01))
            for i in range(self.num_operators)
        ])

        # Convert to probabilities with minimum exploration
        probs = np.maximum(samples, 0.0)
        probs = probs / (probs.sum() + 1e-30)

        # Enforce minimum probability
        probs = np.maximum(probs, self.op_min_prob)
        probs = probs / probs.sum()

        # Select operators
        op_indices = np.random.choice(self.num_operators, size=n, p=probs)
        return op_indices

    def _update_operator_credits(self, op_indices, trial_fitness, old_fitness_after_selection, valid, n):
        """Update Thompson Sampling parameters based on improvement."""
        # We need to track which trials improved. Since selection already happened,
        # we track based on whether trial was accepted (trial_fitness <= old fitness).
        # We stored old fitness before selection, but after _select_survivors_batch
        # the fitness array is already updated. So we need a different approach.
        # Instead, we pass info differently. We'll use a simpler heuristic:
        # compare trial_fitness with a threshold.

        # Actually, let's just use the trial_fitness values directly
        # and give credit based on how good the trial is relative to median
        if n == 0:
            return

        valid_tf = trial_fitness[:n]
        valid_mask = valid[:n]

        for op_id in range(self.num_operators):
            op_mask = (op_indices[:n] == op_id) & valid_mask
            n_op = np.sum(op_mask)
            if n_op == 0:
                continue

            # We don't have the old fitness per individual easily here,
            # so we use a decay-based Thompson Sampling update
            # Count how many were valid (we'll refine in the main loop)
            self.op_trials_count[op_id] += float(n_op)

    def _update_operator_credits_detailed(self, op_indices, improvements, n):
        """Update Thompson Sampling with actual improvement data."""
        decay = 0.995  # slow decay to forget old data

        # Decay existing parameters
        self.op_alpha = 1.0 + (self.op_alpha - 1.0) * decay
        self.op_beta = 1.0 + (self.op_beta - 1.0) * decay

        for i in range(n):
            op_id = int(op_indices[i])
            if op_id < 0 or op_id >= self.num_operators:
                continue
            imp = float(improvements[i])
            if imp > 0:
                # Success: improvement happened
                self.op_alpha[op_id] += min(imp * 1e3, 5.0)  # scale reward
            else:
                # Failure
                self.op_beta[op_id] += 1.0

    def _compute_eigenvectors(self, population):
        """Compute and cache eigenvectors of population covariance."""
        n, dim = population.shape
        if (self.generation - self._eigen_gen < 5 and
                self._eigenvectors is not None and
                self._eigenvectors.shape[0] == dim):
            return self._eigenvectors, self._eigenvalues, self._eigen_center

        try:
            if dim <= 200 and n >= max(dim // 2, 3):
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                self._eigenvectors = eigenvectors
                self._eigenvalues = eigenvalues
                self._eigen_center = center
                self._eigen_gen = self.generation
                return eigenvectors, eigenvalues, center
        except (np.linalg.LinAlgError, ValueError):
            pass

        self._eigenvectors = None
        self._eigenvalues = None
        self._eigen_center = None
        return None, None, None

    # ---- The 7 crossover operators ----

    def _crossover_op0_binomial(self, population, mutants, cr_values, indices):
        """Standard binomial crossover (original)."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim
        pop_sub = population[indices]
        mut_sub = mutants[indices]
        cr_sub = cr_values[indices]

        rand_matrix = np.random.random((n_op, dim))
        cr_matrix = cr_sub[:, np.newaxis]
        j_rand = np.random.randint(0, dim, size=n_op)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n_op), j_rand] = True
        return np.where(cross_mask, mut_sub, pop_sub)

    def _crossover_op1_exponential(self, population, mutants, cr_values, indices):
        """Exponential crossover."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim
        trials = population[indices].copy()
        mut_sub = mutants[indices]
        cr_sub = cr_values[indices]

        for i in range(n_op):
            cr = cr_sub[i]
            j_start = np.random.randint(0, dim)
            L = 0
            while L < dim and (np.random.random() < cr or L == 0):
                j = (j_start + L) % dim
                trials[i, j] = mut_sub[i, j]
                L += 1
        return trials

    def _crossover_op2_eigen_binomial(self, population, mutants, cr_values, indices):
        """Eigenvector-based binomial crossover (variant_08)."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim

        eigenvectors, eigenvalues, center = self._eigenvectors, self._eigenvalues, self._eigen_center

        if eigenvectors is not None and eigenvectors.shape[0] == dim:
            pop_sub = population[indices]
            mut_sub = mutants[indices]
            cr_sub = cr_values[indices]

            pop_rot = np.dot(pop_sub - center, eigenvectors)
            mut_rot = np.dot(mut_sub - center, eigenvectors)

            rand_matrix = np.random.random((n_op, dim))
            cr_matrix = cr_sub[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n_op)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n_op), j_rand] = True

            trials_rot = np.where(cross_mask, mut_rot, pop_rot)

            # Block crossover for a fraction
            block_mask = np.random.random(n_op) < 0.15
            if np.any(block_mask) and dim > 2:
                block_idx = np.where(block_mask)[0]
                k = max(1, dim // 3)
                top_k = np.argsort(eigenvalues)[-k:]
                for bi in block_idx:
                    trials_rot[bi, top_k] = mut_rot[bi, top_k]

            trials = np.dot(trials_rot, eigenvectors.T) + center
            trials = np.clip(trials, self.lb, self.ub)
            return trials
        else:
            # Fallback to binomial
            return self._crossover_op0_binomial(population, mutants, cr_values, indices)

    def _crossover_op3_eigen_block(self, population, mutants, cr_values, indices):
        """Eigenvector-guided block crossover (variant_06 - top winner)."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim

        eigenvectors = self._eigenvectors

        if eigenvectors is not None and eigenvectors.shape[0] == dim and dim >= 3:
            pop_sub = population[indices]
            mut_sub = mutants[indices]
            cr_sub = cr_values[indices]

            pop_rot = pop_sub @ eigenvectors
            mut_rot = mut_sub @ eigenvectors

            trials_rot = pop_rot.copy()

            for i in range(n_op):
                cr = cr_sub[i]
                block_size = max(1, int(np.ceil(cr * dim * 0.5)))
                start = np.random.randint(0, dim)

                mask = np.zeros(dim, dtype=bool)
                idx_block = np.arange(start, start + block_size) % dim
                mask[idx_block] = True

                rand_vals = np.random.random(dim)
                scatter_mask = rand_vals < (cr * 0.3)
                mask = mask | scatter_mask

                if not np.any(mask):
                    mask[np.random.randint(0, dim)] = True

                trials_rot[i, mask] = mut_rot[i, mask]

            trials = trials_rot @ eigenvectors.T
            trials = np.clip(trials, self.lb, self.ub)
            return trials
        else:
            return self._crossover_op0_binomial(population, mutants, cr_values, indices)

    def _crossover_op4_covariance_exponential(self, population, mutants, cr_values, indices):
        """Covariance-rotated exponential crossover (variant_04)."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim

        eigenvectors, eigenvalues, center = self._eigenvectors, self._eigenvalues, self._eigen_center

        if eigenvectors is not None and eigenvectors.shape[0] == dim:
            pop_sub = population[indices]
            mut_sub = mutants[indices]
            cr_sub = cr_values[indices]

            pop_rot = np.dot(pop_sub - center, eigenvectors)
            mut_rot = np.dot(mut_sub - center, eigenvectors)

            trials_rot = pop_rot.copy()
            for i in range(n_op):
                cr = cr_sub[i]
                j_start = np.random.randint(0, dim)
                L = 0
                while L < dim and (np.random.random() < cr or L == 0):
                    j = (j_start + L) % dim
                    trials_rot[i, j] = mut_rot[i, j]
                    L += 1

            # Mix with standard binomial for 20%
            mix_mask = np.random.random(n_op) < 0.2
            if np.any(mix_mask):
                mix_idx = np.where(mix_mask)[0]
                rand_m = np.random.random((len(mix_idx), dim))
                cr_m = cr_sub[mix_idx, np.newaxis]
                j_rand_m = np.random.randint(0, dim, size=len(mix_idx))
                cross_m = rand_m < cr_m
                cross_m[np.arange(len(mix_idx)), j_rand_m] = True
                std_trials = np.where(cross_m, mut_sub[mix_idx], pop_sub[mix_idx])
                # Replace in final result
                result = np.dot(trials_rot, eigenvectors.T) + center
                result[mix_idx] = std_trials
                result = np.clip(result, self.lb, self.ub)
                return result

            trials = np.dot(trials_rot, eigenvectors.T) + center
            trials = np.clip(trials, self.lb, self.ub)
            return trials
        else:
            return self._crossover_op1_exponential(population, mutants, cr_values, indices)

    def _crossover_op5_eigen_explore(self, population, mutants, cr_values, indices):
        """Eigenvector crossover with weak-direction exploration (variant_05)."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim

        eigenvectors, eigenvalues, center = self._eigenvectors, self._eigenvalues, self._eigen_center

        if eigenvectors is not None and eigenvectors.shape[0] == dim and eigenvalues is not None:
            pop_sub = population[indices]
            mut_sub = mutants[indices]
            cr_sub = cr_values[indices]

            pop_eigen = np.dot(pop_sub - center, eigenvectors)
            mut_eigen = np.dot(mut_sub - center, eigenvectors)

            rand_matrix = np.random.random((n_op, dim))
            cr_matrix = cr_sub[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n_op)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n_op), j_rand] = True

            trial_eigen = np.where(cross_mask, mut_eigen, pop_eigen)

            # Explore weak eigenvectors for 15%
            explore_mask = np.random.random(n_op) < 0.15
            if np.any(explore_mask):
                n_exp = np.sum(explore_mask)
                weak_dims = min(max(1, dim // 4), dim)
                perturbation_eigen = np.zeros((n_exp, dim))
                scales = np.sqrt(np.maximum(eigenvalues[:weak_dims], 1e-20))
                perturbation_eigen[:, :weak_dims] = np.random.normal(0, 1, (n_exp, weak_dims)) * scales * 0.1
                perturbation = np.dot(perturbation_eigen, eigenvectors.T)
                trial_partial = np.dot(trial_eigen[explore_mask], eigenvectors.T) + center
                trial_partial = trial_partial + perturbation
                trial_eigen_back = np.dot(trial_partial - center, eigenvectors)
                trial_eigen[explore_mask] = trial_eigen_back

            trials = np.dot(trial_eigen, eigenvectors.T) + center
            trials = np.clip(trials, self.lb, self.ub)
            return trials
        else:
            return self._crossover_op0_binomial(population, mutants, cr_values, indices)

    def _crossover_op6_block_sectored(self, population, mutants, cr_values, indices):
        """Block-sectored crossover (variant_10)."""
        n_op = len(indices)
        if n_op == 0:
            return
        dim = self.dim
        trials = population[indices].copy()
        mut_sub = mutants[indices]
        cr_sub = cr_values[indices]

        for i in range(n_op):
            cr = cr_sub[i]
            inject_random = np.random.random() < 0.05

            if dim <= 4:
                rand_vals = np.random.random(dim)
                j_rand = np.random.randint(0, dim)
                mask = rand_vals < cr
                mask[j_rand] = True
                trials[i] = np.where(mask, mut_sub[i], trials[i])
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
                            trials[i, block_dims] = mut_sub[i, block_dims]

                if np.random.random() < 0.03 and self.x_opt is not None:
                    scale = 0.01 * (self.ub - self.lb) * np.random.random()
                    perturbation = np.random.normal(0, scale, dim)
                    candidate = self.x_opt + perturbation
                    candidate = np.clip(candidate, self.lb, self.ub)
                    trials[i] = candidate

        return trials

    def _crossover_batch_adaptive(self, population, mutants, cr_values, op_indices):
        """Apply different crossover operators based on adaptive selection."""
        n, dim = population.shape
        trials = population.copy()

        # Compute eigenvectors once per generation (shared by multiple operators)
        self._compute_eigenvectors(population)

        # Group individuals by operator
        operator_methods = [
            self._crossover_op0_binomial,
            self._crossover_op1_exponential,
            self._crossover_op2_eigen_binomial,
            self._crossover_op3_eigen_block,
            self._crossover_op4_covariance_exponential,
            self._crossover_op5_eigen_explore,
            self._crossover_op6_block_sectored,
        ]

        for op_id in range(self.num_operators):
            indices = np.where(op_indices == op_id)[0]
            if len(indices) == 0:
                continue

            try:
                result = operator_methods[op_id](population, mutants, cr_values, indices)
                if result is not None and len(result) == len(indices):
                    trials[indices] = result
                else:
                    # Fallback
                    result = self._crossover_op0_binomial(population, mutants, cr_values, indices)
                    if result is not None:
                        trials[indices] = result
            except Exception:
                # Fallback to binomial on any error
                try:
                    result = self._crossover_op0_binomial(population, mutants, cr_values, indices)
                    if result is not None:
                        trials[indices] = result
                except Exception:
                    pass

        return trials

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover - kept for compatibility but not used in main loop."""
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
                                 archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        # Store old fitness for operator credit
        self._old_fitness = fitness[:n].copy()
        self._trial_fitness = trial_fitness[:n].copy() if len(trial_fitness) >= n else trial_fitness.copy()

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

    def _update_operator_credits(self, op_indices, trial_fitness, current_fitness, valid, n):
        """Update Thompson Sampling parameters based on actual improvements."""
        n_eff = min(n, len(trial_fitness), len(self._old_fitness))

        decay = 0.995
        self.op_alpha = 1.0 + (self.op_alpha - 1.0) * decay
        self.op_beta = 1.0 + (self.op_beta - 1.0) * decay

        for i in range(n_eff):
            if not valid[i]:
                continue
            op_id = int(op_indices[i])
            if op_id < 0 or op_id >= self.num_operators:
                continue

            old_f = float(self._old_fitness[i])
            new_f = float(trial_fitness[i])

            if new_f < old_f:
                # Success - scale reward by improvement
                improvement = float(old_f - new_f)
                reward = min(float(improvement / (abs(old_f) + 1e-30)) * 10.0, 5.0)
                reward = max(reward, 1.0)
                self.op_alpha[op_id] += reward
            else:
                # Failure
                self.op_beta[op_id] += 1.0

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
            population_out = np.vstack([elite, combined_new])
            fitness_new = np.full(reinit_count, np.inf)
            return population_out, np.concatenate([elite_fit, fitness_new]), np.empty((0, self.dim)), np.full(self.memory_size, 0.5), np.full(self.memory_size, 0.5), 0

        combined_new = self._clip_to_bounds(combined_new)
        new_fitness = func(combined_new)
        if len(new_fitness) < len(combined_new):
            combined_new = combined_new[:len(new_fitness)]

        self._safe_update_best(combined_new, new_fitness)

        population_out = np.vstack([elite, combined_new[:len(new_fitness)]])
        fitness_all = np.concatenate([elite_fit, new_fitness])

        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        archive = np.empty((0, self.dim))

        self.np_size = len(population_out)

        # Partially reset operator stats on restart (keep some history)
        self.op_alpha = 1.0 + (self.op_alpha - 1.0) * 0.5
        self.op_beta = 1.0 + (self.op_beta - 1.0) * 0.5

        return population_out, fitness_all, archive, memory_f, memory_cr, memory_idx

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
```