

```python
import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, intelligent restart mechanism, and adaptive operator selection
    using Thompson Sampling over multiple parameter generation strategies.
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

        # Adaptive operator selection: 7 strategies
        self.num_strategies = 7
        # Thompson Sampling parameters (Beta distribution: alpha=successes+1, beta=failures+1)
        self.ts_alpha = np.ones(self.num_strategies, dtype=float)
        self.ts_beta = np.ones(self.num_strategies, dtype=float)
        # Sliding window for credit assignment
        self.window_size = 50
        self.strategy_history = []  # list of (strategy_idx, reward)
        # Decay factor for older observations
        self.ts_decay = 0.995

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        # Reset Thompson Sampling
        self.ts_alpha = np.ones(self.num_strategies, dtype=float)
        self.ts_beta = np.ones(self.num_strategies, dtype=float)
        self.strategy_history = []
        self.current_strategy_idx = 0

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
            # Select strategy via Thompson Sampling
            samples = np.array([
                np.random.beta(max(self.ts_alpha[i], 0.01), max(self.ts_beta[i], 0.01))
                for i in range(self.num_strategies)
            ])
            self.current_strategy_idx = int(np.argmax(samples))

            # Generate adaptive parameters using selected strategy
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

            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Compute reward for the selected strategy
            total_improvement = float(np.sum(success_delta)) if len(success_delta) > 0 else 0.0
            n_successes = len(success_f)
            n_total = int(np.sum(valid))

            # Update Thompson Sampling
            if n_total > 0:
                success_rate = float(n_successes) / float(n_total)
                # Weight by improvement magnitude (normalized)
                if total_improvement > 0:
                    reward = success_rate + min(1.0, total_improvement / (abs(self.f_opt) + 1e-30))
                    reward = min(reward, 2.0)  # cap
                else:
                    reward = success_rate

                # Decay old observations
                self.ts_alpha *= self.ts_decay
                self.ts_beta *= self.ts_decay
                # Ensure minimum values
                self.ts_alpha = np.maximum(self.ts_alpha, 0.1)
                self.ts_beta = np.maximum(self.ts_beta, 0.1)

                # Update selected strategy
                sid = self.current_strategy_idx
                self.ts_alpha[sid] += float(reward)
                self.ts_beta[sid] += float(2.0 - reward)

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

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Dispatch to the selected strategy."""
        sid = self.current_strategy_idx
        if sid == 0:
            return self._strategy_original(memory_f, memory_cr)
        elif sid == 1:
            return self._strategy_variant01(memory_f, memory_cr)
        elif sid == 2:
            return self._strategy_variant02(memory_f, memory_cr)
        elif sid == 3:
            return self._strategy_variant04(memory_f, memory_cr)
        elif sid == 4:
            return self._strategy_variant05(memory_f, memory_cr)
        elif sid == 5:
            return self._strategy_variant06(memory_f, memory_cr)
        elif sid == 6:
            return self._strategy_variant07(memory_f, memory_cr)
        else:
            return self._strategy_original(memory_f, memory_cr)

    # === Strategy 0: Original ===
    def _strategy_original(self, memory_f, memory_cr):
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)
        f_values = np.zeros(n)
        for i in range(n):
            while True:
                f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    break
            f_values[i] = min(f_val, 1.0)
        cr_values = np.clip(np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0)
        return f_values, cr_values

    # === Strategy 1: variant_01 (10 wins) ===
    def _strategy_variant01(self, memory_f, memory_cr):
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

    # === Strategy 2: variant_02 (2 wins) ===
    def _strategy_variant02(self, memory_f, memory_cr):
        n = self.np_size
        explore_ratio = 0.4
        n_explore = max(1, int(explore_ratio * n))
        n_exploit = n - n_explore

        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        if n_exploit > 0:
            indices = np.random.randint(0, self.memory_size, size=n_exploit)
            for i in range(n_exploit):
                attempts = 0
                while attempts < 100:
                    f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                    attempts += 1
                else:
                    f_val = 0.5 + 0.1 * abs(np.random.standard_cauchy())
                f_values[i] = min(f_val, 1.0)
            cr_values[:n_exploit] = np.clip(
                np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0
            )

        if n_explore > 0:
            start = n_exploit
            high_mask = np.random.random(n_explore) < 0.6
            n_high = np.sum(high_mask)
            n_mod = n_explore - n_high

            explore_f = np.zeros(n_explore)
            if n_high > 0:
                explore_f[high_mask] = np.random.uniform(0.7, 1.2, size=n_high)
            if n_mod > 0:
                explore_f[~high_mask] = np.random.uniform(0.4, 0.8, size=n_mod)

            explore_f = np.clip(explore_f, 0.1, 1.2)
            f_values[start:] = explore_f

            cr_values[start:] = np.clip(
                np.random.normal(0.9, 0.1, size=n_explore), 0.5, 1.0
            )

        f_values = np.clip(f_values, 0.01, 1.2)
        f_values = np.minimum(f_values, 1.0)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        perm = np.random.permutation(n)
        f_values = f_values[perm]
        cr_values = cr_values[perm]

        return f_values, cr_values

    # === Strategy 3: variant_04 (1 win) ===
    def _strategy_variant04(self, memory_f, memory_cr):
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        f_values = np.zeros(n)
        aggressive_mask = np.random.random(n) < 0.5

        for i in range(n):
            if aggressive_mask[i]:
                while True:
                    f_val = 0.85 + 0.15 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                f_values[i] = min(f_val, 1.0)
            else:
                center = max(memory_f[indices[i]], 0.6)
                while True:
                    f_val = center + 0.2 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                f_values[i] = min(f_val, 1.0)

        cr_high_mask = np.random.random(n) < 0.6
        high_cr = np.random.normal(0.95, 0.05, size=n)
        memory_cr_inflated = np.maximum(memory_cr[indices], 0.7)
        adapted_cr = np.random.normal(memory_cr_inflated, 0.1)
        cr_values = np.where(cr_high_mask, high_cr, adapted_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)
        f_values = np.maximum(f_values, 0.1)

        return f_values, cr_values

    # === Strategy 4: variant_05 (2 wins) ===
    def _strategy_variant05(self, memory_f, memory_cr):
        n = self.np_size
        gen = getattr(self, 'generation', 0)

        explore_ratio = max(0.3, 0.85 - gen * 0.003)

        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        is_explorer = np.random.random(n) < explore_ratio
        n_explore = np.sum(is_explorer)
        n_exploit = n - n_explore

        if n_explore > 0:
            f_explore = np.random.normal(0.85, 0.15, n_explore)
            boost = np.random.random(n_explore) < 0.2
            f_explore[boost] = np.random.uniform(0.9, 1.0, np.sum(boost))
            f_explore = np.clip(f_explore, 0.1, 1.0)
            f_values[is_explorer] = f_explore

            cr_explore = np.zeros(n_explore)
            high_cr_mask = np.random.random(n_explore) < 0.6
            cr_explore[high_cr_mask] = np.random.normal(0.95, 0.05, np.sum(high_cr_mask))
            cr_explore[~high_cr_mask] = np.random.normal(0.1, 0.05, np.sum(~high_cr_mask))
            cr_explore = np.clip(cr_explore, 0.0, 1.0)
            cr_values[is_explorer] = cr_explore

        if n_exploit > 0:
            indices = np.random.randint(0, self.memory_size, size=n_exploit)

            f_exploit = np.zeros(n_exploit)
            for i in range(n_exploit):
                for _ in range(100):
                    f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        f_exploit[i] = min(f_val, 1.0)
                        break
                else:
                    f_exploit[i] = 0.5
            f_values[~is_explorer] = f_exploit

            cr_exploit = np.clip(np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0)
            cr_values[~is_explorer] = cr_exploit

        return f_values, cr_values

    # === Strategy 5: variant_06 (1 win) ===
    def _strategy_variant06(self, memory_f, memory_cr):
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        explore_rate = max(0.3, 0.7 - self.generation * 0.002)
        is_explore = np.random.random(n) < explore_rate

        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        n_explore = np.sum(is_explore)
        n_exploit = n - n_explore

        if n_explore > 0:
            explore_f = np.zeros(n_explore)
            for j in range(n_explore):
                attempts = 0
                while attempts < 100:
                    f_val = 0.9 + 0.2 * np.random.standard_cauchy()
                    if f_val > 0.4:
                        break
                    attempts += 1
                explore_f[j] = min(f_val, 1.5)
            f_values[is_explore] = explore_f
            cr_explore = np.clip(np.random.normal(0.9, 0.05, n_explore), 0.5, 1.0)
            cr_values[is_explore] = cr_explore

        if n_exploit > 0:
            exploit_indices = indices[~is_explore]
            exploit_f = np.zeros(n_exploit)
            for j in range(n_exploit):
                attempts = 0
                while attempts < 100:
                    f_val = memory_f[exploit_indices[j]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                    attempts += 1
                exploit_f[j] = min(f_val, 1.0)
            f_values[~is_explore] = exploit_f
            cr_exploit = np.clip(
                np.random.normal(memory_cr[exploit_indices], 0.1), 0.0, 1.0
            )
            cr_values[~is_explore] = cr_exploit

        n_extreme = max(1, n // 10)
        extreme_idx = np.random.choice(n, n_extreme, replace=False)
        f_values[extreme_idx] = np.clip(1.0 + 0.3 * np.abs(np.random.standard_cauchy(n_extreme)), 1.0, 1.5)
        cr_values[extreme_idx] = np.random.uniform(0.8, 1.0, n_extreme)

        return f_values, cr_values

    # === Strategy 6: variant_07 (1 win) ===
    def _strategy_variant07(self, memory_f, memory_cr):
        n = self.np_size
        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        gen_ratio = min(1.0, self.generation / max(1, 200))
        explore_frac = 0.6 * (1.0 - 0.5 * gen_ratio)

        n_explore = int(n * explore_frac)
        n_refine = n - n_explore

        if n_explore > 0:
            base_f_explore = np.random.uniform(0.6, 1.0, size=n_explore)
            cauchy_perturb = 0.2 * np.random.standard_cauchy(size=n_explore)
            f_explore = base_f_explore + cauchy_perturb
            bad = f_explore <= 0
            while np.any(bad):
                f_explore[bad] = np.random.uniform(0.5, 1.0, size=np.sum(bad))
                bad = f_explore <= 0
            f_explore = np.clip(f_explore, 0.01, 1.5)
            f_explore = np.minimum(f_explore, 1.0)
            cr_explore = np.clip(np.random.normal(0.9, 0.1, size=n_explore), 0.5, 1.0)
            f_values[:n_explore] = f_explore
            cr_values[:n_explore] = cr_explore

        if n_refine > 0:
            indices = np.random.randint(0, self.memory_size, size=n_refine)
            for i in range(n_refine):
                scale = 0.15
                attempts = 0
                while attempts < 50:
                    f_val = memory_f[indices[i]] + scale * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                    attempts += 1
                if attempts >= 50:
                    f_val = memory_f[indices[i]]
                f_values[n_explore + i] = min(f_val, 1.0)
            cr_refine = np.clip(
                np.random.normal(memory_cr[indices], 0.15), 0.0, 1.0
            )
            cr_values[n_explore:] = cr_refine

        perm = np.random.permutation(n)
        f_values = f_values[perm]
        cr_values = cr_values[perm]

        return f_values, cr_values

    def _select_strategies_batch(self):
        """Select mutation strategy for each individual. Returns boolean mask."""
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using mixed strategies for the whole population."""
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
```