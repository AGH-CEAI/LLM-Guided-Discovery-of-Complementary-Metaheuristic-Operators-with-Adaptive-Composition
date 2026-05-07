```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation strategy works best during optimization.
    Now includes ADAPTIVE _clip_to_bounds_batch strategy selection.
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

        # === Adaptive Operator Selection (Thompson Sampling) for Mutation ===
        self.num_operators = 4
        self.operator_names = ['original', 'variant_01', 'variant_07', 'variant_09']
        self.operator_alpha = np.ones(self.num_operators)
        self.operator_beta = np.ones(self.num_operators)
        self.reward_window_size = 30
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        self.current_operator_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []

        # === Adaptive Operator Selection for _clip_to_bounds_batch (NEW) ===
        self.num_clip_operators = 8
        self.clip_operator_names = ['clip_v01', 'clip_v02', 'clip_v03', 'clip_v04',
                                     'clip_v05', 'clip_v06', 'clip_v07', 'clip_v08']
        # Beta distribution parameters for Thompson Sampling
        self.clip_operator_alpha = np.ones(self.num_clip_operators)
        self.clip_operator_beta = np.ones(self.num_clip_operators)
        # Sliding window reward tracking
        self.clip_reward_window_size = 25
        self.clip_operator_rewards = [[] for _ in range(self.num_clip_operators)]
        self.current_clip_operator_idx = 0
        self.last_clip_switch_gen = 0
        self.clip_min_switch_gens = 8
        self.clip_switches = []
        # Track boundary violations per clip operator
        self.clip_operator_violations = [0] * self.num_clip_operators
        self.clip_operator_total = [0] * self.num_clip_operators

        # Archive for variant_01 (JADE-style)
        self.archive = None
        self._pending_archive_additions = None

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _select_clip_operator_thompson(self):
        """Select clip operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.clip_operator_alpha, self.clip_operator_beta)
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

    def _compute_clip_reward(self, trial_fitness, valid_mask):
        """Compute reward for clip operator based on solution quality after clipping"""
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)
        valid_fitness = trial_fitness[valid_mask] if valid_mask is not None else trial_fitness
        
        if len(valid_fitness) == 0:
            return -0.05
        
        # Reward based on mean fitness quality (lower is better)
        mean_fitness = float(np.mean(valid_fitness))
        # Negative mean fitness is good, so we negate
        reward = float(np.clip(-mean_fitness / 1e4, -1.0, 1.0))
        
        # Bonus for no NaN/Inf
        nan_inf_count = int(np.sum(~np.isfinite(valid_fitness)))
        if nan_inf_count == 0:
            reward += 0.1
        
        return float(np.clip(reward, -1.0, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(reward)

        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = np.mean(recent)

            self.operator_alpha[operator_idx] = 1.0 + success_rate * 5.0 + avg_reward * 2.0
            self.operator_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0

            self.operator_alpha[operator_idx] = float(np.clip(self.operator_alpha[operator_idx], 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(self.operator_beta[operator_idx], 0.1, 100.0))

    def _update_clip_operator_rewards(self, operator_idx, reward):
        """Update rewards for clip operator with sliding window"""
        rewards = self.clip_operator_rewards[operator_idx]
        rewards.append(reward)

        if len(rewards) > self.clip_reward_window_size:
            rewards.pop(0)

        recent = rewards[-min(8, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = np.mean(recent)

            self.clip_operator_alpha[operator_idx] = 1.0 + success_rate * 5.0 + avg_reward * 2.0
            self.clip_operator_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0

            self.clip_operator_alpha[operator_idx] = float(np.clip(self.clip_operator_alpha[operator_idx], 0.1, 100.0))
            self.clip_operator_beta[operator_idx] = float(np.clip(self.clip_operator_beta[operator_idx], 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self._select_operator_thompson()
        self._select_clip_operator_thompson()

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
            
            # Compute clip reward and update clip operator
            valid_mask = np.ones(len(trial_fitness), dtype=bool)
            clip_reward = self._compute_clip_reward(trial_fitness, valid_mask)

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
            self._update_clip_operator_rewards(self.current_clip_operator_idx, clip_reward)

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

            # Clip operator switching with minimum generations between switches
            if self.generation - self.last_clip_switch_gen >= self.clip_min_switch_gens:
                new_clip_op = self._select_clip_operator_thompson()
                if new_clip_op != self.current_clip_operator_idx:
                    self.current_clip_operator_idx = new_clip_op
                    self.last_clip_switch_gen = self.generation
                    self.clip_switches.append((self.generation, new_clip_op))

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
        """Dispatch to selected clip strategy"""
        if self.current_clip_operator_idx == 0:
            return self._clip_v01(population)
        elif self.current_clip_operator_idx == 1:
            return self._clip_v02(population)
        elif self.current_clip_operator_idx == 2:
            return self._clip_v03(population)
        elif self.current_clip_operator_idx == 3:
            return self._clip_v04(population)
        elif self.current_clip_operator_idx == 4:
            return self._clip_v05(population)
        elif self.current_clip_operator_idx == 5:
            return self._clip_v06(population)
        elif self.current_clip_operator_idx == 6:
            return self._clip_v07(population)
        else:
            return self._clip_v08(population)

    # === Clip Strategy Implementations ===

    def _clip_v01(self, population):
        """Variant 01: Reflection-based with double reflection"""
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

    def _clip_v02(self, population):
        """Variant 02: Reflection/Wrap mode based on stagnation"""
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

    def _clip_v03(self, population):
        """Variant 03: Periodic reflection"""
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

    def _clip_v04(self, population):
        """Variant 04: Adaptive damping based on stagnation"""
        reflected = np.copy(population)
        stagnation_factor = min(1.0 + 0.5 * self.stagnation_count, 2.5)
        damping = 1.0 / stagnation_factor
        
        upper_bounce = self.upper - (population - self.upper) * damping
        reflected = np.where(population > self.upper, upper_bounce, reflected)
        
        lower_bounce = self.lower + (self.lower - population) * damping
        reflected = np.where(population < self.lower, lower_bounce, reflected)
        
        return np.clip(reflected, self.lower, self.upper)

    def _clip_v05(self, population):
        """Variant 05: 0.7x excess/deficit reflection"""
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

    def _clip_v06(self, population):
        """Variant 06: Simple reflection with overflow/underflow"""
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

    def _clip_v07(self, population):
        """Variant 07: Probabilistic reflection with adaptive softness"""
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

    def _clip_v08(self, population):
        """Variant 08: Clip with jitter on boundary hits"""
        clipped = np.clip(population, self.lower, self.upper)
        
        hit_lower = population < self.lower
        hit_upper = population > self.upper
        
        if not (np.any(hit_lower) or np.any(hit_upper)):
            return clipped
        
        jitter = np.random.randn(*population.shape) * 0.01 * (self.upper - self.lower)
        
        reflected = np.where(hit_lower, 
                            self.lower + (self.lower - population[hit_lower]) + jitter[hit_lower], 
                            clipped)
        reflected = np.where(hit_upper, 
                            self.upper - (population[hit_upper] - self.upper) + jitter[hit_upper], 
                            reflected)
        
        return np.clip(reflected, self.lower, self.upper)

    def _eval_wrapper_batch(self, population, func):
        result = func(population)
        if len(result) < len(population):
            result = np.pad(result, (0, len(population) - len(result)), constant_values=np.inf)
        return result

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
        """Dispatch to selected mutation strategy"""
        if self.current_operator_idx == 0:
            return self._mutate_original(population, fitness)
        elif self.current_operator_idx == 1:
            return self._mutate_variant01(population, fitness)
        elif self.current_operator_idx == 2:
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
            F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count - 10)), 2.0)

        base = population + F_adapted * (global_best - population) + F_adapted * (population[r1] - population[r2])

        if self.stagnation_count > 5:
            perturbation_scale = min(0.3 * np.log1p(self.stagnation_count), 2.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            base = base + random_perturbation

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

    def _mutate_variant09(self, population, fitness):
        """Variant 09: Multi-Donor Composite Mutation with Diversity-Weighted Selection"""
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
            self.cultural_weights[self.culture_idx] = 1.0 / (1.0 + self.stagnation_count)
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
```