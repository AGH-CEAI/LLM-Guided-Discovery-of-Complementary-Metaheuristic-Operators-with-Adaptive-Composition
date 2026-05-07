```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation strategy AND boundary handling strategy works best during optimization.
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
        self.num_operators = 4
        self.operator_names = ['original', 'variant_01', 'variant_07', 'variant_09']
        # Beta distribution parameters for Thompson Sampling
        self.operator_alpha = np.ones(self.num_operators)
        self.operator_beta = np.ones(self.num_operators)
        # Sliding window reward tracking
        self.reward_window_size = 30
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        self.current_operator_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []

        # Archive for variant_01 (JADE-style)
        self.archive = None
        self._pending_archive_additions = None

        # === Adaptive Boundary Handling (Thompson Sampling) ===
        self.num_boundary_operators = 8
        self.boundary_operator_names = ['original', 'variant_01', 'variant_02', 'variant_03',
                                         'variant_04', 'variant_05', 'variant_06', 'variant_07', 'variant_08']
        # Beta distribution parameters for Thompson Sampling
        self.boundary_alpha = np.ones(self.num_boundary_operators)
        self.boundary_beta = np.ones(self.num_boundary_operators)
        # Sliding window reward tracking
        self.boundary_reward_window_size = 30
        self.boundary_operator_rewards = [[] for _ in range(self.num_boundary_operators)]
        self.current_boundary_operator_idx = 0
        self.last_boundary_switch_gen = 0
        self.boundary_min_switch_gens = 5
        self.boundary_switches = []

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _select_boundary_operator_thompson(self):
        """Select boundary handling operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.boundary_alpha, self.boundary_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on improvement quality"""
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            reward = -0.01
        else:
            reward = total_improvement / max(float(n_improved), 1.0)

        return float(np.clip(reward, -1.0, 1.0))

    def _compute_boundary_reward(self, fitness, trial_fitness):
        """Compute reward for boundary handling based on trial performance"""
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            reward = -0.01
        else:
            reward = total_improvement / max(float(n_improved), 1.0)

        return float(np.clip(reward, -1.0, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(float(reward))

        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if float(r) > 0)
            success_rate = float(successes) / float(len(recent))
            avg_reward = float(np.mean(recent))

            self.operator_alpha[operator_idx] = float(np.clip(1.0 + success_rate * 5.0 + avg_reward * 2.0, 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0, 0.1, 100.0))

    def _update_boundary_operator_rewards(self, operator_idx, reward):
        """Update rewards for boundary operator with sliding window"""
        rewards = self.boundary_operator_rewards[operator_idx]
        rewards.append(float(reward))

        if len(rewards) > self.boundary_reward_window_size:
            rewards.pop(0)

        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if float(r) > 0)
            success_rate = float(successes) / float(len(recent))
            avg_reward = float(np.mean(recent))

            self.boundary_alpha[operator_idx] = float(np.clip(1.0 + success_rate * 5.0 + avg_reward * 2.0, 0.1, 100.0))
            self.boundary_beta[operator_idx] = float(np.clip(1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0, 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self._select_operator_thompson()
        self._select_boundary_operator_thompson()

        while not stopping_condition():
            mutants = self._mutate_current_to_pbest_with_culture(population, fitness)
            mutants = np.clip(mutants, self.lower, self.upper)

            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

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
            boundary_reward = self._compute_boundary_reward(fitness, trial_fitness)

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            fitness = np.clip(fitness, -1e50, 1e50)

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
            self._update_boundary_operator_rewards(self.current_boundary_operator_idx, boundary_reward)

            # Update archive if using variant_01
            if self.current_operator_idx == 1 and self.archive is not None and len(self.archive) > 0:
                self._update_archive(population, fitness, trials, trial_fitness)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
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

            # Boundary operator switching with minimum generations between switches
            if self.generation - self.last_boundary_switch_gen >= self.boundary_min_switch_gens:
                new_boundary_op = self._select_boundary_operator_thompson()
                if new_boundary_op != self.current_boundary_operator_idx:
                    self.current_boundary_operator_idx = new_boundary_op
                    self.last_boundary_switch_gen = self.generation
                    self.boundary_switches.append((self.generation, new_boundary_op))

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
        """Dispatch to selected boundary handling strategy"""
        idx = int(self.current_boundary_operator_idx)
        if idx == 0:
            return self._clip_original(population)
        elif idx == 1:
            return self._clip_variant01(population)
        elif idx == 2:
            return self._clip_variant02(population)
        elif idx == 3:
            return self._clip_variant03(population)
        elif idx == 4:
            return self._clip_variant04(population)
        elif idx == 5:
            return self._clip_variant05(population)
        elif idx == 6:
            return self._clip_variant06(population)
        elif idx == 7:
            return self._clip_variant07(population)
        else:
            return self._clip_variant08(population)

    def _clip_original(self, population):
        """Original: Simple clip to bounds"""
        return np.clip(population, self.lower, self.upper)

    def _clip_variant01(self, population):
        """Variant 01: Double reflection with clamping"""
        result = np.copy(population)

        above_upper = result > self.upper
        if np.any(above_upper):
            reflected = 2.0 * self.upper - result[above_upper]
            result[above_upper] = np.where(reflected >= self.lower, reflected,
                                            2.0 * self.lower - reflected)

        below_lower = result < self.lower
        if np.any(below_lower):
            reflected = 2.0 * self.lower - result[below_lower]
            result[below_lower] = np.where(reflected <= self.upper, reflected,
                                            2.0 * self.upper - reflected)

        return result

    def _clip_variant02(self, population):
        """Variant 02: Stagnation-dependent reflection/wrap"""
        lower = self.lower
        upper = self.upper

        if np.any(upper <= lower):
            return np.clip(population, lower, upper)

        result = population.copy()
        range_size = upper - lower

        if self.stagnation_count < 10:
            below = result < lower
            above = result > upper

            reflected_below = lower + (lower - result)
            reflected_below = np.clip(reflected_below, lower, upper)
            result = np.where(below, reflected_below, result)

            reflected_above = upper - (result - upper)
            reflected_above = np.clip(reflected_above, lower, upper)
            result = np.where(above, reflected_above, result)

            still_below = result < lower
            still_above = result > upper
            result = np.where(still_below, lower + (lower - result) * 0.5, result)
            result = np.where(still_above, upper - (result - upper) * 0.5, result)
        else:
            result = lower + (result - lower) % (range_size + 1e-10)

        result = np.clip(result, lower, upper)

        invalid = ~np.isfinite(result)
        if np.any(invalid):
            result[invalid] = (lower + upper)[invalid] * 0.5

        return result

    def _clip_variant03(self, population):
        """Variant 03: Modular reflection"""
        range_width = self.upper - self.lower
        clipped = np.clip(population, self.lower, self.upper)
        overflow = population - clipped
        reflected = clipped - overflow
        reflected = np.where(population > self.upper,
                            self.upper - ((population - self.upper) % (2 * range_width + 1e-10)),
                            reflected)
        reflected = np.where(population < self.lower,
                            self.lower + ((self.lower - population) % (2 * range_width + 1e-10)),
                            reflected)
        return np.clip(reflected, self.lower, self.upper)

    def _clip_variant04(self, population):
        """Variant 04: Adaptive damping reflection"""
        reflected = np.copy(population)
        stagnation_factor = min(1.0 + 0.5 * self.stagnation_count, 2.5)
        damping = 1.0 / stagnation_factor
        upper_bounce = self.upper - (population - self.upper) * damping
        reflected = np.where(population > self.upper, upper_bounce, reflected)
        lower_bounce = self.lower + (self.lower - population) * damping
        reflected = np.where(population < self.lower, lower_bounce, reflected)
        return np.clip(reflected, self.lower, self.upper)

    def _clip_variant05(self, population):
        """Variant 05: Partial reflection with 0.7 factor"""
        reflected = np.copy(population)
        above_upper = population > self.upper
        if np.any(above_upper):
            excess = population[above_upper] - self.upper
            reflected[above_upper] = self.upper - 0.7 * excess
        below_lower = population < self.lower
        if np.any(below_lower):
            deficit = self.lower - population[below_lower]
            reflected[below_lower] = self.lower + 0.7 * deficit
        return reflected

    def _clip_variant06(self, population):
        """Variant 06: Direct reflection with overflow/underflow"""
        result = np.copy(population)
        upper_mask = result > self.upper
        if np.any(upper_mask):
            overflow = result[upper_mask] - self.upper
            result[upper_mask] = self.upper - overflow
        lower_mask = result < self.lower
        if np.any(lower_mask):
            underflow = self.lower - result[lower_mask]
            result[lower_mask] = self.lower + underflow
        return np.clip(result, self.lower, self.upper)

    def _clip_variant07(self, population):
        """Variant 07: Probabilistic multi-bounce reflection"""
        result = np.copy(population)
        below_mask = result < self.lower
        above_mask = result > self.upper
        below_dist = np.where(below_mask, self.lower - result, 0.0)
        above_dist = np.where(above_mask, result - self.upper, 0.0)
        stagnation_factor = min(self.stagnation_count / 30.0, 1.0)
        generation_factor = min(self.generation / 100.0, 0.3)
        boundary_softness = 0.5 + 0.5 * stagnation_factor + generation_factor

        for _ in range(3):
            any_outside = np.any(below_mask) or np.any(above_mask)
            if not any_outside:
                break

            reflect_below = below_mask & (np.random.rand(*result.shape) < boundary_softness)
            result[reflect_below] = 2 * self.lower[reflect_below] - result[reflect_below]

            reflect_above = above_mask & (np.random.rand(*result.shape) < boundary_softness)
            result[reflect_above] = 2 * self.upper[reflect_above] - result[reflect_above]

            below_mask = result < self.lower
            above_mask = result > self.upper

        result = np.clip(result, self.lower, self.upper)
        return result

    def _clip_variant08(self, population):
        """Variant 08: Reflection with jitter"""
        clipped = np.clip(population, self.lower, self.upper)
        hit_lower = population < self.lower
        hit_upper = population > self.upper
        if not (np.any(hit_lower) or np.any(hit_upper)):
            return clipped

        jitter = np.random.randn(*population.shape) * 0.01 * (self.upper - self.lower)
        reflected = np.where(hit_lower, self.lower + (self.lower - population[hit_lower]) + jitter[hit_lower], clipped)
        reflected = np.where(hit_upper, self.upper - (population[hit_upper] - self.upper) + jitter[hit_upper], reflected)
        return np.clip(reflected, self.lower, self.upper)

    def _eval_wrapper_batch(self, population, func):
        result = func(population)
        if len(result) < len(population):
            result = np.pad(result, (0, len(population) - len(result)), constant_values=np.inf)
        return result

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
        """Dispatch to selected mutation strategy"""
        idx = int(self.current_operator_idx)
        if idx == 0:
            return self._mutate_original(population, fitness)
        elif idx == 1:
            return self._mutate_variant01(population, fitness)
        elif idx == 2:
            return self._mutate_variant07(population, fitness)
        else:
            return self._mutate_variant09(population, fitness)

    def _mutate_original(self, population, fitness):
        """Original mutation strategy: current-to-pbest/1 with cultural memory"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant01(self, population, fitness):
        """Variant 01: JADE mutation with archive and bounded random F"""
        if self.archive is None:
            self.archive = np.zeros((0, self.dim))

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
            all_indices = np.concatenate([np.arange(self.NP), self.archive])
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
        rejected = population[trial_fitness >= fitness]
        if len(rejected) > 0:
            max_archive = self.NP
            if len(self.archive) + len(rejected) > max_archive:
                indices = np.random.choice(len(rejected), max_archive - len(self.archive), replace=False)
                rejected = rejected[indices]
            self.archive = np.vstack([self.archive, rejected])[-max_archive:]

    def _mutate_variant07(self, population, fitness):
        """Variant 07: GLOBAL best, adaptive F, escalating perturbation"""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        global_best_idx = best_indices[0]
        global_best = population[global_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        F_adapted = self.F
        if self.stagnation_count > 10:
            F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count