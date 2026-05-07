import numpy as np


class AdaptiveMultiStrategyDE:
    """
    Adaptive DE with Thompson Sampling for automatic operator selection.
    Based on benchmark results, includes 9 mutation strategies with adaptive selection.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(kwargs.get('NP', 8 * dim), 300)
        self.min_pop_size = 20
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        # Adaptive parameters
        self.F = 0.7
        self.CR = 0.5
        self.p_best_rate = 0.1

        # Cultural memory
        self.cultural_memory_size = min(30, self.NP)
        self.cultural_memory = np.zeros((self.cultural_memory_size, dim))
        self.cultural_weights = np.ones(self.cultural_memory_size)
        self.culture_idx = 0

        # Local search
        self.local_search_interval = kwargs.get('local_search_interval', 7)
        self.local_radius = kwargs.get('local_radius', 2.0)
        self.local_max_iter = 10

        # Diversity and stagnation
        self.stagnation_count = 0
        self.stagnation_limit = kwargs.get('stagnation_limit', 60)
        self.diversity_threshold = 0.1
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        self.improvement_buffer = []

        # === Adaptive Operator Selection (Thompson Sampling) ===
        # 9 operators: original + 8 winning variants
        self.num_operators = 9
        self.operator_names = [
            'original', 'variant_02', 'variant_05', 'variant_06',
            'variant_07', 'variant_08', 'variant_10',
            'jade_archive', 'multi_donor'
        ]
        # Beta distribution parameters for Thompson Sampling
        self.operator_alpha = np.ones(self.num_operators)
        self.operator_beta = np.ones(self.num_operators)
        # Prior weights based on benchmark win counts
        benchmark_wins = [10, 1, 1, 1, 5, 1, 2, 0, 0]
        for i, wins in enumerate(benchmark_wins):
            self.operator_alpha[i] = 1.0 + wins * 0.5
            self.operator_beta[i] = 1.0 + (sum(benchmark_wins) - wins) * 0.1

        # Sliding window reward tracking
        self.reward_window_size = 40
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        self.operator_successes = [[] for _ in range(self.num_operators)]
        self.current_operator_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []

        # Per-operator fitness tracking for credit assignment
        self.operator_best_fitness = [np.inf] * self.num_operators
        self.operator_calls = [0] * self.num_operators

        # Archive for JADE-style mutation
        self.archive = np.zeros((0, dim))

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on relative improvement quality"""
        fitness = np.array([float(np.clip(f, -1e50, 1e50)) for f in fitness])
        trial_fitness = np.array([float(np.clip(f, -1e50, 1e50)) for f in trial_fitness])

        improvements = fitness - trial_fitness
        valid_improvements = improvements[np.isfinite(improvements)]
        n_improved = int(np.sum(valid_improvements > 0))
        total_valid = max(int(np.sum(np.isfinite(improvements))), 1)

        if n_improved == 0:
            return -0.1

        pos_improvements = valid_improvements[valid_improvements > 0]
        if len(pos_improvements) == 0:
            return -0.1

        # Relative improvement
        abs_fitness = np.abs(fitness[np.isfinite(fitness) & (fitness != 0)])
        if len(abs_fitness) == 0:
            return 0.0
        scale = float(np.mean(abs_fitness)) + 1e-10
        relative_imp = pos_improvements / scale

        # Log-scaled reward
        log_rewards = np.log1p(relative_imp)
        avg_log_reward = float(np.mean(log_rewards))

        # Consistency bonus
        consistency_ratio = float(n_improved) / float(total_valid)
        consistency_bonus = 0.15 * consistency_ratio

        reward = avg_log_reward + consistency_bonus
        return float(np.clip(reward, -0.5, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window and credit assignment"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(float(reward))
        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        successes = self.operator_successes[operator_idx]
        successes.append(1.0 if reward > 0 else 0.0)
        if len(successes) > self.reward_window_size:
            successes.pop(0)

        # Update Beta distribution parameters
        window = min(15, len(rewards))
        if window >= 3:
            recent_rewards = rewards[-window:]
            recent_successes = successes[-window:]

            success_rate = float(np.mean(recent_successes))
            avg_reward = float(np.mean(recent_rewards))

            alpha_update = 1.0 + success_rate * 5.0 + max(0, avg_reward) * 2.0
            beta_update = 1.0 + (1.0 - success_rate) * 5.0 + max(0, -avg_reward) * 2.0

            self.operator_alpha[operator_idx] = float(np.clip(alpha_update, 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(beta_update, 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)
        fitness = np.array([float(np.clip(f, -1e50, 1e50)) for f in fitness])

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self.current_operator_idx = self._select_operator_thompson()

        while not stopping_condition():
            mutants = self._mutate_dispatch(population, fitness)
            mutants = np.clip(mutants, self.lower, self.upper)

            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)
            trial_fitness = np.array([float(np.clip(f, -1e50, 1e50)) for f in trial_fitness])

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            # Compute reward before selection
            reward = self._compute_reward(fitness, trial_fitness)

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            fitness = np.array([float(np.clip(f, -1e50, 1e50)) for f in fitness])

            best_idx = np.argmin(fitness)
            current_best = float(fitness[best_idx])
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Update operator performance
            self._update_operator_rewards(self.current_operator_idx, reward)
            self.operator_calls[self.current_operator_idx] += 1

            # Update JADE archive if applicable
            if self.current_operator_idx == 7:
                self._update_archive(population, fitness, trials, trial_fitness)

            self._update_cultural_memory_on_improvement(
                population[best_idx],
                mutants[best_idx] if best_idx < len(mutants) else mutants[0]
            )
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            # Operator switching with minimum generations between switches
            if self.generation - self.last_operator_switch_gen >= self.operator_min_switch_gens:
                new_op = self._select_operator_thompson()
                if new_op != self.current_operator_idx:
                    self.current_operator_idx = new_op
                    self.last_operator_switch_gen = self.generation
                    self.operator_switches.append((self.generation, new_op))

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

    def _initialize_population_lhs(self):
        samples = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
        for d in range(self.dim):
            edges = np.linspace(self.lower[d], self.upper[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = edges[perm] + np.random.uniform(0, edges[1] - edges[0], self.NP)
        return self._clip_to_bounds_batch(samples)

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lower, self.upper)

    def _eval_wrapper_batch(self, population, func):
        result = func(population)
        if len(result) < len(population):
            result = np.pad(result, (0, len(population) - len(result)), constant_values=np.inf)
        return result

    def _mutate_dispatch(self, population, fitness):
        """Dispatch to selected mutation strategy"""
        dispatch_map = {
            0: self._mutate_original,
            1: self._mutate_variant02,
            2: self._mutate_variant05,
            3: self._mutate_variant06,
            4: self._mutate_variant07,
            5: self._mutate_variant08,
            6: self._mutate_variant10,
            7: self._mutate_jade_archive,
            8: self._mutate_multi_donor,
        }
        return dispatch_map[self.current_operator_idx](population, fitness)

    def _get_pbest_indices(self, population, fitness):
        """Get p_best indices helper"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        return p_best_idx

    def _get_random_indices(self, n_indices=3):
        """Get random distinct indices helper"""
        idx_a = np.random.randint(0, self.NP, (self.NP, n_indices))
        for col in range(1, n_indices):
            idx_a[:, col] = np.where(
                idx_a[:, col] == np.arange(self.NP)[:, None],
                (idx_a[:, col] + 1) % self.NP,
                idx_a[:, col]
            )
        for col in range(1, n_indices):
            for row in range(self.NP):
                for prev_col in range(col):
                    if idx_a[row, col] == idx_a[row, prev_col]:
                        idx_a[row, col] = (idx_a[row, col] + 1) % self.NP
        return idx_a

    def _mutate_original(self, population, fitness):
        """Original mutation: current-to-pbest/1 with cultural memory"""
        p_best_idx = self._get_pbest_indices(population, fitness)
        p_best = population[p_best_idx]
        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            donors = self._blend_with_cultural(population, base, cultural_contribution)
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _blend_with_cultural(self, population, base, cultural_contribution):
        """Blend mutation with cultural memory"""
        similarity = np.array([
            float(np.linalg.norm(population[i] - cultural_contribution))
            for i in range(self.NP)
        ])
        sim_std = float(np.std(similarity)) + 1e-10
        weights_culture = np.exp(-similarity / sim_std)
        weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
        cultural_offset = cultural_contribution * 0.15
        donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        return donors

    def _mutate_variant02(self, population, fitness):
        """Historical Best-Guided Mutation with Convergence-Aware Exploration"""
        p_best_idx = self._get_pbest_indices(population, fitness)
        p_best = population[p_best_idx]
        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Historical best guidance
        if self.best_solution is not None:
            hist_best = self.best_solution
            hist_dist = np.linalg.norm(population - hist_best, axis=1) + 1e-10
            hist_weights = 1.0 / hist_dist
            hist_weights = hist_weights / (np.sum(hist_weights) + 1e-10)

            pop_center = np.mean(population, axis=0)
            convergence = np.linalg.norm(population - pop_center, axis=1)
            denom = float(np.linalg.norm(hist_best - pop_center)) + 1e-10
            convergence = np.clip(convergence / denom, 0, 1)
            hist_strength = 0.3 * (1.0 - convergence)
            hist_direction = (hist_best - population)
            base = base + hist_strength[:, np.newaxis] * hist_direction

        # Convergence-aware perturbation
        pop_center = np.mean(population, axis=0)
        pop_dispersion = np.linalg.norm(population - pop_center, axis=1)
        avg_dispersion = float(np.mean(pop_dispersion)) + 1e-10
        normalized_dispersion = pop_dispersion / avg_dispersion

        low_disp_mask = normalized_dispersion < 0.5
        if np.any(low_disp_mask):
            n_low = int(np.sum(low_disp_mask))
            perturbation_scale = 0.5 * self.F * (1.0 - normalized_dispersion[low_disp_mask])
            perturbation = np.random.randn(n_low, self.dim) * perturbation_scale[:, np.newaxis]
            base[low_disp_mask] = base[low_disp_mask] + perturbation

        return np.clip(base, self.lower, self.upper)

    def _mutate_variant05(self, population, fitness):
        """Opposition-Guided DE with Adaptive Reflection and Quasi-Opposition"""
        p_best_idx = self._get_pbest_indices(population, fitness)
        p_best = population[p_best_idx]
        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Adaptive reflection probability
        decay_rate = 0.01
        reflect_prob = max(0.05, 0.8 * np.exp(-decay_rate * self.generation))

        center = (self.lower + self.upper) / 2.0
        opposite_full = 2.0 * center - base
        quasi_opposite = np.random.uniform(center, opposite_full)
        quasi_reflected = np.random.uniform(center, base)

        dim_perturb = np.random.randn(self.NP, self.dim) * (0.1 + 0.3 * np.exp(-0.005 * self.generation))

        donors = base.copy()
        reflect_mask = np.random.rand(self.NP) < reflect_prob
        reflect_indices = np.where(reflect_mask)[0]
        n_reflect = len(reflect_indices)

        if n_reflect > 0:
            half = n_reflect // 2
            donors[reflect_indices[:half]] = opposite_full[reflect_indices[:half]]
            donors[reflect_indices[half:]] = quasi_opposite[reflect_indices[half:]]

        non_reflect_indices = np.where(~reflect_mask)[0]
        if len(non_reflect_indices) > 0:
            donors[non_reflect_indices] = donors[non_reflect_indices] + dim_perturb[non_reflect_indices]
            mix_mask = np.random.rand(len(non_reflect_indices)) < 0.3
            mix_indices = non_reflect_indices[mix_mask]
            if len(mix_indices) > 0:
                donors[mix_indices] = 0.7 * donors[mix_indices] + 0.3 * quasi_reflected[mix_indices]

        # Cultural memory blending
        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            donors = self._blend_with_cultural(population, donors, cultural_contribution)

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant06(self, population, fitness):
        """Opposition-Guided Escape: basin-hopping via dynamic opposition vectors"""
        p_best_idx = self._get_pbest_indices(population, fitness)
        p_best = population[p_best_idx]
        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        centroid = np.mean(population, axis=0)

        # Adaptive F based on stagnation
        if self.stagnation_count > 10:
            F_escape = min(self.F * (1.0 + 0.5 * np.log1p(self.stagnation_count)), 2.5)
        else:
            F_escape = self.F

        mutation_dir = p_best - population + F_escape * (population[r1] - population[r2])
        opposition_vector = 2.0 * centroid - population
        opposition_dir = opposition_vector - population + F_escape * (population[r2] - population[r1])

        base_opp_prob = 0.1
        stagnation_boost = min(0.6, 0.05 * self.stagnation_count)
        opp_prob = np.clip(base_opp_prob + stagnation_boost, 0.0, 0.7)

        use_opposition = np.random.rand(self.NP) < opp_prob
        donor_exploit = population + mutation_dir
        donor_explore = population + opposition_dir
        donors = np.where(use_opposition[:, np.newaxis], donor_explore, donor_exploit)

        if self.stagnation_count > 20:
            perturbation_scale = min(0.5 * np.log1p(self.stagnation_count - 20), 5.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            donors = donors + random_perturbation

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant07(self, population, fitness):
        """Restart-based mutation: aggressive escape from local optima via random restarts"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        global_best_idx = best_indices[0]
        global_best = population[global_best_idx]

        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (global_best - population) + self.F * (population[r1] - population[r2])

        if self.stagnation_count > 20:
            restart_prob = min(0.50 + 0.05 * (self.stagnation_count - 20), 0.95)
            restart_mask = np.random.rand(self.NP) < restart_prob
            n_restart = int(np.sum(restart_mask))
            if n_restart > 0:
                restart_vectors = np.random.uniform(self.lower, self.upper, (n_restart, self.dim))
                base[restart_mask] = restart_vectors

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            donors = self._blend_with_cultural(population, base, cultural_contribution)
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant08(self, population, fitness):
        """Reflective Diversification: escape local optima by bouncing away from centroid"""
        p_best_idx = self._get_pbest_indices(population, fitness)
        p_best = population[p_best_idx]
        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        centroid = np.mean(population, axis=0)
        diff_from_centroid = population - centroid
        dist_from_centroid = np.linalg.norm(diff_from_centroid, axis=1, keepdims=True) + 1e-10
        direction_from_centroid = diff_from_centroid / dist_from_centroid
        reflected_direction = -direction_from_centroid

        convergence_ratio = np.std(population, axis=0) / (self.upper - self.lower + 1e-10)
        avg_convergence = float(np.mean(convergence_ratio))

        if self.stagnation_count > 10 or avg_convergence < 0.05:
            reflection_strength = min(0.5, 0.25 + 0.025 * self.stagnation_count)
        elif self.stagnation_count > 3:
            reflection_strength = 0.15 + 0.02 * self.stagnation_count
        else:
            reflection_strength = 0.0

        if reflection_strength > 0:
            jump_scale = (self.upper - self.lower) * 0.15
            reflected_points = centroid + reflected_direction * jump_scale
            donors = (1 - reflection_strength) * base + reflection_strength * reflected_points
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant10(self, population, fitness):
        """Restart-Driven Opposition-Based Mutation: escape local optima via reflection"""
        bounds_center = (self.lower + self.upper) / 2.0
        is_stuck = self.stagnation_count > 15

        if is_stuck:
            opposition_pop = 2.0 * bounds_center - population
            opposition_pop = np.clip(opposition_pop, self.lower, self.upper)

            p_best_idx = self._get_pbest_indices(population, fitness)
            p_best = population[p_best_idx]
            idx_a = self._get_random_indices(3)
            r1, r2 = idx_a[:, 1], idx_a[:, 2]

            F_oppose = min(self.F * 1.5, 2.0)
            opp_base = opposition_pop + F_oppose * (p_best - opposition_pop) + F_oppose * (population[r1] - population[r2])

            F_std = self.F
            std_base = population + F_std * (p_best - population) + F_std * (population[r1] - population[r2])

            donors = 0.6 * opp_base + 0.4 * std_base
            return np.clip(donors, self.lower, self.upper)

        # Standard mutation
        p_best_idx = self._get_pbest_indices(population, fitness)
        p_best = population[p_best_idx]
        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            donors = self._blend_with_cultural(population, base, cultural_contribution)
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_jade_archive(self, population, fitness):
        """JADE mutation with archive and bounded random F"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        p_best_indices = sorted_idx[:n_best]
        p_best_idx = p_best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        r1 = np.random.randint(0, self.NP, size=self.NP)
        r2 = np.zeros(self.NP, dtype=int)
        for i in range(self.NP):
            while True:
                r2[i] = np.random.randint(0, self.NP)
                if r2[i] != i and r2[i] != r1[i]:
                    break

        archive_size = len(self.archive)
        total_size = self.NP + archive_size

        if archive_size > 0:
            r3_indices = np.random.randint(0, total_size, size=self.NP)
            r3 = np.where(r3_indices < self.NP, r3_indices, -(r3_indices - self.NP + 1))
            for i in range(self.NP):
                if r3[i] == i:
                    r3[i] = (i + 1) % self.NP
            r3_pop = np.where(r3 >= 0, population[r3], self.archive[-(r3 + 1)])
        else:
            r3 = np.random.randint(0, self.NP, size=self.NP)
            for i in range(self.NP):
                while r3[i] == i or r3[i] == r1[i]:
                    r3[i] = (r3[i] + 1) % self.NP
            r3_pop = population[r3]

        F_i = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)
        donors = population + F_i[:, np.newaxis] * (p_best - population) + F_i[:, np.newaxis] * (r3_pop - population[r1])
        return np.clip(donors, self.lower, self.upper)

    def _update_archive(self, population, fitness, trials, trial_fitness):
        """Update JADE archive with rejected solutions"""
        better_mask = trial_fitness >= fitness
        rejected = population[better_mask]
        if len(rejected) > 0:
            max_archive = self.NP
            if len(self.archive) + len(rejected) > max_archive:
                indices = np.random.choice(len(rejected), max(0, max_archive - len(self.archive)), replace=False)
                rejected = rejected[indices]
            self.archive = np.vstack([self.archive, rejected])[-max_archive:]

    def _mutate_multi_donor(self, population, fitness):
        """Multi-Donor Composite Mutation with Diversity-Weighted Selection"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = self._get_random_indices(3)
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        top_percentile = max(1, self.NP // 5)
        top_indices = sorted_idx[:top_percentile]
        centroid = np.mean(population[top_indices], axis=0)

        donor_best = population + self.F * (p_best - population)
        donor_centroid = population + self.F * (centroid - population)
        donor_rand = population + self.F * (population[r1] - population[r2])

        dist_best = np.linalg.norm(donor_best - population, axis=1) + 1e-10
        dist_centroid = np.linalg.norm(donor_centroid - population, axis=1) + 1e-10
        dist_rand = np.linalg.norm(donor_rand - population, axis=1) + 1e-10

        total_dist = dist_best + dist_centroid + dist_rand + 1e-10
        w_best = dist_best / total_dist
        w_centroid = dist_centroid / total_dist
        w_rand = dist_rand / total_dist

        if self.stagnation_count > 10:
            w_best = np.clip(w_best * 0.5, 0.1, 0.6)
            w_centroid = np.clip(w_centroid * 1.5, 0.2, 0.6)
            w_rand = np.clip(w_rand * 1.2, 0.1, 0.4)
            total_w = w_best + w_centroid + w_rand
            w_best /= total_w
            w_centroid /= total_w
            w_rand /= total_w

        donors = (w_best[:, np.newaxis] * donor_best +
                  w_centroid[:, np.newaxis] * donor_centroid +
                  w_rand[:, np.newaxis] * donor_rand)

        return np.clip(donors, self.lower, self.upper)

    def _crossover_binomial_batch(self, population, donors):
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        trials = np.where(mask, donors, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        new_population = np.copy(population)
        new_population[better] = trials[better]
        new_fitness = np.copy(fitness)
        new_fitness[better] = trial_fitness[better]
        return new_population, new_fitness

    def _update_cultural_memory_on_improvement(self, best_individual, best_mutant):
        if self.stagnation_count == 0:
            pattern = best_mutant - best_individual
            self.cultural_memory[self.culture_idx] = pattern
            self.cultural_weights[self.culture_idx] = 1.0 / (1.0 + float(self.stagnation_count))
            self.culture_idx = (self.culture_idx + 1) % self.cultural_memory_size
            decay = 0.95
            self.cultural_weights *= decay

    def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        if not hasattr(self, '_adapt_weighted_improvement'):
            self._adapt_weighted_improvement = 0.0
            self._adapt_fitness_spread = 1.0
            self._adapt_history_F = []
            self._adapt_history_CR = []

        current_best = float(np.min(fitness))
        if hasattr(self, '_prev_best_fitness') and np.isfinite(self._prev_best_fitness):
            direction = self._prev_best_fitness - current_best
            decay = 0.7
            self._adapt_weighted_improvement = decay * self._adapt_weighted_improvement + (1 - decay) * direction
        self._prev_best_fitness = current_best

        finite_fitness = fitness[np.isfinite(fitness)]
        if len(finite_fitness) > 1:
            spread = float(np.std(finite_fitness)) + 1e-10
            self._adapt_fitness_spread = 0.9 * self._adapt_fitness_spread + 0.1 * spread

        norm_signal = self._adapt_weighted_improvement / (self._adapt_fitness_spread + 1e-10)

        if norm_signal > 0.1:
            F_base = 0.4 + 0.3 * np.random.rand()
            F_scale = 0.05
        elif norm_signal < -0.1:
            F_base = 0.7 + 0.4 * np.random.rand()
            F_scale = 0.15
        else:
            F_base = 0.5 + 0.3 * np.random.rand()
            F_scale = 0.1

        F_candidate = F_base + np.random.randn() * F_scale
        new_F = float(np.clip(F_candidate, 0.1, 2.0))

        if norm_signal > 0.1:
            CR_base = 0.7 + 0.2 * np.random.rand()
        elif norm_signal < -0.1:
            CR_base = 0.3 + 0.2 * np.random.rand()
        else:
            CR_base = 0.5 + 0.2 * np.random.rand()

        CR_candidate = CR_base + np.random.randn() * 0.1
        new_CR = float(np.clip(CR_candidate, 0.05, 0.98))

        if len(self._adapt_history_F) > 0:
            momentum = 0.3
            new_F = momentum * np.mean(self._adapt_history_F[-5:]) + (1 - momentum) * new_F
            new_CR = momentum * np.mean(self._adapt_history_CR[-5:]) + (1 - momentum) * new_CR

        self.F = float(np.clip(new_F, 0.1, 2.0))
        self.CR = float(np.clip(new_CR, 0.05, 0.98))

        self._adapt_history_F.append(self.F)
        self._adapt_history_CR.append(self.CR)
        if len(self._adapt_history_F) > 20:
            self._adapt_history_F.pop(0)
            self._adapt_history_CR.pop(0)

        if self.stagnation_count > 15:
            delta = np.random.uniform(0.02, 0.08)
            self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.05, 0.4))
        elif self.stagnation_count > 5:
            if np.random.rand() < 0.3:
                self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.03, 0.03), 0.05, 0.3))

    def _apply_adaptive_local_search(self, func):
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        for _ in range(self.local_max_iter):
            if radius < 1e-6:
                break

            directions = np.random.randn(self.dim)
            directions = directions / (np.linalg.norm(directions) + 1e-10)
            candidates = np.clip(center + radius * directions, self.lower, self.upper)
            candidates = np.vstack([candidates, np.clip(center - radius * directions, self.lower, self.upper)])
            candidates = np.vstack([candidates, center])

            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = np.argmin(fit_candidates)
            best_local_fitness = float(fit_candidates[best_local_idx])

            if best_local_fitness < current_fitness:
                center = candidates[best_local_idx].copy()
                current_fitness = best_local_fitness
                radius = min(radius * 1.5, 10.0)
            else:
                radius *= 0.4

        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = center.copy()
            self.local_radius = min(radius * 1.2, 10.0)
        else:
            self.local_radius = max(radius * 0.8, 0.1)

    def _compute_diversity_batch(self, population):
        centroids = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroids, axis=1)
        return float(np.mean(distances))

    def _check_diversity_low(self, population):
        diversity = self._compute_diversity_batch(population)
        return (diversity < self.diversity_threshold or
                self.stagnation_count > self.stagnation_limit)

    def _reinitialize_population(self, population, fitness):
        if self.best_solution is not None:
            best_idx = np.argmin(fitness)
            population[0] = self.best_solution.copy()
            fitness[0] = self.best_fitness

        n_random = max(self.min_pop_size, self.NP // 2)
        new_part = self._initialize_population_lhs()[:n_random]
        population[1:1 + n_random] = new_part
        fitness[1:1 + n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5
