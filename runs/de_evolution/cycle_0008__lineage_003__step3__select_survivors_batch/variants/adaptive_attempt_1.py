import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, intelligent restart mechanism, and adaptive
    survivor selection via Thompson Sampling over multiple strategies.
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

        # Adaptive operator selection: Thompson Sampling
        self.num_strategies = 8  # number of survivor selection strategies
        self.strategy_alpha = np.ones(self.num_strategies)  # Beta dist alpha (successes)
        self.strategy_beta = np.ones(self.num_strategies)   # Beta dist beta (failures)
        self.strategy_window_size = 20
        self.strategy_rewards = [[] for _ in range(self.num_strategies)]
        self.current_strategy_idx = 0

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset Thompson Sampling
        self.strategy_alpha = np.ones(self.num_strategies)
        self.strategy_beta = np.ones(self.num_strategies)
        self.strategy_rewards = [[] for _ in range(self.num_strategies)]

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

            # Select survivor strategy via Thompson Sampling
            self.current_strategy_idx = self._select_survivor_strategy()

            # Track fitness before selection
            best_before = float(np.min(fitness))

            # Selection and archive update using chosen strategy
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Compute reward for the chosen strategy
            best_after = float(np.min(fitness))
            num_improvements = len(success_f)
            reward = self._compute_strategy_reward(best_before, best_after, num_improvements, len(population))
            self._update_strategy_stats(self.current_strategy_idx, reward)

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

    def _select_survivor_strategy(self):
        """Select a survivor strategy using Thompson Sampling."""
        samples = np.array([
            np.random.beta(self.strategy_alpha[i], self.strategy_beta[i])
            for i in range(self.num_strategies)
        ])
        return int(np.argmax(samples))

    def _compute_strategy_reward(self, best_before, best_after, num_improvements, pop_size):
        """Compute reward for strategy based on improvement."""
        improvement = best_before - best_after
        # Normalize by scale
        scale = max(abs(best_before), 1e-30)
        normalized_improvement = improvement / scale
        # Also reward number of successful replacements
        success_rate = num_improvements / max(pop_size, 1)
        reward = float(np.clip(normalized_improvement * 100 + success_rate, 0.0, 1.0))
        return reward

    def _update_strategy_stats(self, strategy_idx, reward):
        """Update Thompson Sampling statistics for the chosen strategy."""
        reward = float(reward)
        self.strategy_rewards[strategy_idx].append(reward)
        # Keep sliding window
        if len(self.strategy_rewards[strategy_idx]) > self.strategy_window_size:
            self.strategy_rewards[strategy_idx] = self.strategy_rewards[strategy_idx][-self.strategy_window_size:]

        # Update alpha/beta based on reward (treat as Bernoulli-like)
        if reward > 0.01:
            self.strategy_alpha[strategy_idx] += reward
        else:
            self.strategy_beta[strategy_idx] += 1.0 - reward

        # Decay to prevent lock-in
        decay = 0.995
        self.strategy_alpha = np.maximum(1.0, self.strategy_alpha * decay)
        self.strategy_beta = np.maximum(1.0, self.strategy_beta * decay)

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
                                 archive, f_values, cr_values, valid_mask):
        """Dispatch to the selected survivor strategy."""
        idx = self.current_strategy_idx
        if idx == 0:
            return self._survivors_original(population, fitness, trials, trial_fitness,
                                            archive, f_values, cr_values, valid_mask)
        elif idx == 1:
            return self._survivors_crowding_sa(population, fitness, trials, trial_fitness,
                                               archive, f_values, cr_values, valid_mask)
        elif idx == 2:
            return self._survivors_crowding_neighbor(population, fitness, trials, trial_fitness,
                                                     archive, f_values, cr_values, valid_mask)
        elif idx == 3:
            return self._survivors_crowding_prob(population, fitness, trials, trial_fitness,
                                                 archive, f_values, cr_values, valid_mask)
        elif idx == 4:
            return self._survivors_sa_crowding_mix(population, fitness, trials, trial_fitness,
                                                    archive, f_values, cr_values, valid_mask)
        elif idx == 5:
            return self._survivors_sa_normalized(population, fitness, trials, trial_fitness,
                                                  archive, f_values, cr_values, valid_mask)
        elif idx == 6:
            return self._survivors_sa_crowding_hybrid(population, fitness, trials, trial_fitness,
                                                      archive, f_values, cr_values, valid_mask)
        elif idx == 7:
            return self._survivors_fitness_sharing(population, fitness, trials, trial_fitness,
                                                    archive, f_values, cr_values, valid_mask)
        else:
            return self._survivors_original(population, fitness, trials, trial_fitness,
                                            archive, f_values, cr_values, valid_mask)

    def _archive_individual(self, archive, individual):
        """Helper to add an individual to the archive."""
        if len(archive) < self.archive_max:
            if len(archive) > 0:
                archive = np.vstack([archive, individual.reshape(1, -1)])
            else:
                archive = individual.reshape(1, -1).copy()
        else:
            replace_idx = np.random.randint(0, self.archive_max)
            archive[replace_idx] = individual
        return archive

    # Strategy 0: Original greedy selection
    def _survivors_original(self, population, fitness, trials, trial_fitness,
                            archive, f_values, cr_values, valid_mask):
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
                archive = self._archive_individual(archive, population[i])
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 1: Probabilistic acceptance with crowding distance (variant_02)
    def _survivors_crowding_sa(self, population, fitness, trials, trial_fitness,
                               archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)
        T = max(1e-10, 1.0 * np.exp(-generation / 100.0))

        # Compute crowding distances
        try:
            from scipy.spatial.distance import cdist
            dists = cdist(population[:n], population[:n])
            np.fill_diagonal(dists, np.inf)
            crowd_dist = np.min(dists, axis=1)
            cd_max = np.max(crowd_dist) + 1e-30
            crowd_scores = crowd_dist / cd_max
        except Exception:
            crowd_scores = np.ones(n) * 0.5

        for i in range(n):
            if not valid_mask[i]:
                continue

            delta = trial_fitness[i] - fitness[i]

            if delta <= 0:
                if delta < 0:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(delta)))
                archive = self._archive_individual(archive, population[i])
                population[i] = trials[i].copy()
                fitness[i] = trial_fitness[i]
            else:
                fit_range = np.max(fitness[:n]) - np.min(fitness[:n]) + 1e-30
                normalized_delta = delta / fit_range
                crowding_boost = 1.0 + 2.0 * (1.0 - crowd_scores[i])
                accept_prob = np.exp(-normalized_delta / (T * crowding_boost + 1e-30))
                accept_prob = min(accept_prob, 0.3)

                if np.random.random() < accept_prob:
                    archive = self._archive_individual(archive, population[i])
                    population[i] = trials[i].copy()
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 2: SA with crowding neighbor replacement (variant_03)
    def _survivors_crowding_neighbor(self, population, fitness, trials, trial_fitness,
                                     archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)
        temperature = max(1e-10, 1.0 * (0.995 ** generation))

        fit_range = np.ptp(fitness[:n])
        if fit_range < 1e-30:
            fit_range = 1.0

        for i in range(n):
            if not valid_mask[i]:
                continue

            if trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(fitness[i] - trial_fitness[i])))
                archive = self._archive_individual(archive, population[i])
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                delta = (trial_fitness[i] - fitness[i]) / (fit_range + 1e-30)
                accept_prob = np.exp(-delta / (temperature + 1e-30))

                if np.random.random() < accept_prob * 0.15:
                    dists = np.sum((population[:n] - trials[i]) ** 2, axis=1)
                    dists[i] = np.inf
                    k = max(2, n // 10)
                    if k < n:
                        nearest_indices = np.argpartition(dists, k)[:k]
                    else:
                        nearest_indices = np.arange(n)
                    worst_neighbor = nearest_indices[np.argmax(fitness[nearest_indices])]

                    if trial_fitness[i] < fitness[worst_neighbor]:
                        archive = self._archive_individual(archive, population[worst_neighbor])
                        population[worst_neighbor] = trials[i]
                        fitness[worst_neighbor] = trial_fitness[i]
                        success_f.append(float(f_values[i]))
                        success_cr.append(float(cr_values[i]))
                        success_delta.append(float(abs(fitness[worst_neighbor] - trial_fitness[i])))

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 3: Crowding with probabilistic worse-acceptance (variant_04)
    def _survivors_crowding_prob(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)
        accept_temp = max(1e-15, 1.0 * np.exp(-generation / 30.0))

        for i in range(n):
            if not valid_mask[i]:
                continue

            dists = np.sum((population[:n] - trials[i]) ** 2, axis=1)
            dists[i] = np.inf
            nearest_idx = np.argmin(dists)

            target_idx = nearest_idx if np.random.random() < 0.5 else i

            if trial_fitness[i] <= fitness[target_idx]:
                if trial_fitness[i] < fitness[target_idx]:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(fitness[target_idx] - trial_fitness[i])))
                archive = self._archive_individual(archive, population[target_idx])
                population[target_idx] = trials[i]
                fitness[target_idx] = trial_fitness[i]
            elif target_idx == i:
                delta = trial_fitness[i] - fitness[i]
                scale = max(abs(fitness[i]), 1e-30)
                rel_delta = delta / scale
                accept_prob = np.exp(-rel_delta / (accept_temp + 1e-30))
                accept_prob = min(accept_prob, 0.05)

                if np.random.random() < accept_prob:
                    archive = self._archive_individual(archive, population[i])
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 4: Crowding with SA-style (variant_05)
    def _survivors_sa_crowding_mix(self, population, fitness, trials, trial_fitness,
                                    archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        for i in range(n):
            if not valid_mask[i]:
                continue

            diffs = population[:n] - trials[i]
            distances = np.sum(diffs ** 2, axis=1)
            nearest_idx = int(np.argmin(distances))

            if np.random.random() < 0.7:
                target_idx = nearest_idx
            else:
                target_idx = i

            target_fit = fitness[target_idx]

            accept = False
            if trial_fitness[i] <= target_fit:
                accept = True
            else:
                delta = trial_fitness[i] - target_fit
                temperature = max(1e-15, 1.0 / (1.0 + generation * 0.1))
                scale = max(abs(target_fit), 1e-30)
                prob = np.exp(-delta / (scale * temperature + 1e-30))
                prob = min(prob, 0.05)
                if np.random.random() < prob:
                    accept = True

            if accept:
                if trial_fitness[i] < fitness[target_idx]:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(fitness[target_idx] - trial_fitness[i])))

                archive = self._archive_individual(archive, population[target_idx])
                population[target_idx] = trials[i].copy()
                fitness[target_idx] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 5: SA with normalized fitness range (variant_06)
    def _survivors_sa_normalized(self, population, fitness, trials, trial_fitness,
                                  archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        fit_valid = fitness[~np.isnan(fitness)]
        if len(fit_valid) > 1:
            fit_range = float(np.max(fit_valid) - np.min(fit_valid))
        else:
            fit_range = 1.0
        fit_range = max(fit_range, 1e-30)

        temperature = fit_range * 0.3 * np.exp(-generation / 150.0)
        temperature = max(temperature, 1e-30)

        for i in range(n):
            if not valid_mask[i]:
                continue

            delta = trial_fitness[i] - fitness[i]

            if delta <= 0:
                if delta < 0:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(delta)))
                archive = self._archive_individual(archive, population[i])
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                normalized_delta = delta / fit_range
                accept_prob = np.exp(-normalized_delta / (temperature / fit_range + 1e-30))
                accept_prob = min(accept_prob, 0.15)

                if np.random.random() < accept_prob:
                    archive = self._archive_individual(archive, population[i])
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 6: SA with crowding replacement hybrid (variant_07)
    def _survivors_sa_crowding_hybrid(self, population, fitness, trials, trial_fitness,
                                      archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        fit_valid = fitness[~np.isnan(fitness)]
        T_init = max(1.0, float(np.max(np.abs(fit_valid))) * 0.1) if len(fit_valid) > 0 else 10.0
        T = T_init * np.exp(-generation / 150.0)
        T = max(T, 1e-15)

        crowding_prob = 0.3

        for i in range(n):
            if not valid_mask[i]:
                continue

            if trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(fitness[i] - trial_fitness[i])))

                archive = self._archive_individual(archive, population[i])

                if np.random.random() < crowding_prob and n > 4:
                    dists = np.sum((population[:n] - trials[i]) ** 2, axis=1)
                    dists[i] = np.inf
                    nearest = int(np.argmin(dists))
                    if trial_fitness[i] <= fitness[nearest]:
                        population[nearest] = trials[i]
                        fitness[nearest] = trial_fitness[i]
                    else:
                        population[i] = trials[i]
                        fitness[i] = trial_fitness[i]
                else:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]
            else:
                delta = trial_fitness[i] - fitness[i]
                scale = max(abs(fitness[i]), 1e-30)
                normalized_delta = delta / scale
                accept_prob = np.exp(-normalized_delta / (T / scale + 1e-30))
                accept_prob = min(accept_prob, 0.15)

                if np.random.random() < accept_prob:
                    archive = self._archive_individual(archive, population[i])
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    # Strategy 7: Fitness sharing SA (variant_10)
    def _survivors_fitness_sharing(self, population, fitness, trials, trial_fitness,
                                    archive, f_values, cr_values, valid_mask):
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        generation = getattr(self, 'generation', 0)

        T_initial = 100.0
        T_min = 1e-10
        decay_rate = 0.02
        temperature = max(T_min, T_initial * np.exp(-decay_rate * generation))

        pop_center = np.mean(population[:n], axis=0)
        pop_dists = np.sqrt(np.sum((population[:n] - pop_center)**2, axis=1))
        sigma_share = max(1e-10, float(np.median(pop_dists)) * 0.5)

        shared_fitness = fitness[:n].copy()
        for i in range(n):
            dists = np.sqrt(np.sum((population[:n] - population[i])**2, axis=1))
            sharing = np.sum(np.maximum(0, 1.0 - (dists / sigma_share)**2))
            sharing = max(1.0, sharing)
            shared_fitness[i] = fitness[i] * sharing

        for i in range(n):
            if not valid_mask[i]:
                continue

            trial_dists = np.sqrt(np.sum((population[:n] - trials[i])**2, axis=1))
            trial_sharing = np.sum(np.maximum(0, 1.0 - (trial_dists / sigma_share)**2))
            trial_sharing = max(1.0, trial_sharing)
            trial_shared = trial_fitness[i] * trial_sharing

            accept = False
            if trial_fitness[i] <= fitness[i]:
                accept = True
            else:
                delta = trial_shared - shared_fitness[i]
                if temperature > T_min:
                    prob = np.exp(-delta / (temperature + 1e-30))
                    prob = min(prob, 0.3)
                    if np.random.random() < prob:
                        accept = True

            if accept:
                if trial_fitness[i] < fitness[i]:
                    success_f.append(float(f_values[i]))
                    success_cr.append(float(cr_values[i]))
                    success_delta.append(float(abs(fitness[i] - trial_fitness[i])))

                archive = self._archive_individual(archive, population[i])
                population[i] = trials[i].copy()
                fitness[i] = trial_fitness[i]
                shared_fitness[i] = trial_shared

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
        return float(np.mean(distances))

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
