```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection and UCB1 for parameter adaptation.
    Automatically learns which mutation strategy and parameter adaptation strategy works best.
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
        self._prev_fitness = None

        # === Adaptive Operator Selection (Thompson Sampling) ===
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

        # === Adaptive Parameter Adaptation Strategy Selection (UCB1) ===
        self.num_adapt_strategies = 3
        self.adapt_strategy_names = ['adapt_v05', 'adapt_v07', 'adapt_original']
        self.adapt_strategy_rewards = [[] for _ in range(self.num_adapt_strategies)]
        self.adapt_strategy_counts = [0] * self.num_adapt_strategies
        self.adapt_strategy_avg_rewards = [0.0] * self.num_adapt_strategies
        self.current_adapt_strategy = 0
        self.last_adapt_switch_gen = 0
        self.adapt_min_switch_gens = 8

        # Archive for variant_01 (JADE-style)
        self.archive = None
        self._pending_archive_additions = None

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _select_adapt_strategy_ucb1(self):
        """Select adaptation strategy using UCB1"""
        if self.generation < 5:
            return 0
        
        ucb_values = []
        for i in range(self.num_adapt_strategies):
            if self.adapt_strategy_counts[i] == 0:
                ucb_values.append(float('inf'))
            else:
                exploitation = self.adapt_strategy_avg_rewards[i]
                exploration = np.sqrt(2.0 * np.log(float(max(1, self.generation))) / float(max(1, self.adapt_strategy_counts[i])))
                ucb_values.append(float(exploitation + exploration))
        
        selected = int(np.argmax(ucb_values))
        return selected

    def _update_adapt_strategy_rewards(self, strategy_idx, reward):
        """Update rewards for adaptation strategy with sliding window"""
        rewards = self.adapt_strategy_rewards[strategy_idx]
        rewards.append(float(reward))
        if len(rewards) > 50:
            rewards.pop(0)
        self.adapt_strategy_counts[strategy_idx] += 1
        if len(rewards) > 0:
            self.adapt_strategy_avg_rewards[strategy_idx] = float(np.mean(rewards))

    def _compute_adapt_reward(self, fitness, prev_fitness):
        """Compute reward for adaptation strategy based on improvement"""
        if prev_fitness is None:
            return 0.0
        prev_fitness = float(np.clip(prev_fitness, -1e50, 1e50))
        current_best = float(np.min(fitness))
        if not np.isfinite(current_best) or not np.isfinite(prev_fitness):
            return 0.0
        improvement_ratio = (current_best - prev_fitness) / (abs(prev_fitness) + 1e-30)
        reward = float(np.clip(-improvement_ratio * 0.1, -1.0, 1.0))
        return reward

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)
        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()
        self._prev_fitness = self.best_fitness

        self.current_operator_idx = self._select_operator_thompson()
        self.current_adapt_strategy = self._select_adapt_strategy_ucb1()

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

            reward = self._compute_reward(fitness, trial_fitness)

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

            self._update_operator_rewards(self.current_operator_idx, reward)

            if self.current_operator_idx == 1 and self.archive is not None and len(self.archive) > 0:
                self._update_archive(population, fitness, trials, trial_fitness)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            
            adapt_reward = self._compute_adapt_reward(fitness, self._prev_fitness)
            self._adapt_parameters_batch(population, fitness, mutants, trials)
            self._prev_fitness = float(np.min(fitness))

            self._update_adapt_strategy_rewards(self.current_adapt_strategy, adapt_reward)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            if self.generation - self.last_operator_switch_gen >= self.operator_min_switch_gens:
                new_op = self._select_operator_thompson()
                if new_op != self.current_operator_idx:
                    self.current_operator_idx = new_op
                    self.last_operator_switch_gen = self.generation
                    self.operator_switches.append((self.generation, new_op))

            if self.generation - self.last_adapt_switch_gen >= self.adapt_min_switch_gens:
                new_adapt = self._select_adapt_strategy_ucb1()
                if new_adapt != self.current_adapt_strategy:
                    self.current_adapt_strategy = new_adapt
                    self.last_adapt_switch_gen = self.generation

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

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

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(float(reward))
        if len(rewards) > self.reward_window_size:
            rewards.pop(0)
        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if r > 0)
            success_rate = float(successes) / float(len(recent))
            avg_reward = float(np.mean(recent))
            self.operator_alpha[operator_idx] = float(np.clip(1.0 + success_rate * 5.0 + avg_reward * 2.0, 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0, 0.1, 100.0))

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
        """Variant 09: Adaptive F based on improvement ratio, direction weight"""
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

        if np.isfinite(self.best_fitness) and self.generation > 5:
            current_best = np.min(fitness)
            denom = abs(self.best_fitness) + 1e-30
            improvement_ratio = (current_best - self.best_fitness) / denom

            F_escape = np.clip(self.F * (1.0 + max(0, -improvement_ratio) * 0.5), 0.5, 2.0)
            w_hist = np.clip(0.7 + max(0, -improvement_ratio) * 0.2, 0.5, 0.9)
            w_curr = 1.0 - w_hist

            if np.any(self.cultural_weights > 0):
                weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
                cultural_centroid = np.dot(weights_norm, self.cultural_memory)
                target = w_hist * self.best_solution + w_curr * p_best + 0.1 * cultural_centroid
            else:
                target = w_hist * self.best_solution + w_curr * p_best

            donors = population + F_escape * (target - population) + self.F * (population[r1] - population[r2])
        else:
            donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

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
        """Dispatch to selected parameter adaptation strategy"""
        if self.current_adapt_strategy == 0:
            self._adapt_v05(population, fitness, mutants, trials)
        elif self.current_adapt_strategy == 1:
            self._adapt_v07(population, fitness, mutants, trials)
        else:
            self._adapt_original(population, fitness, mutants, trials)

    def _adapt_v05(self, population, fitness, mutants, trials):
        """Variant 05 style: EMA-based adaptation with stagnation handling"""
        if not hasattr(self, '_ema_F'):
            self._ema_F = 0.7
            self._ema_CR = 0.5
            self._ema_p = 0.1

        improvements = fitness - trials
        improved_mask = improvements > 0
        n_improved = int(np.sum(improved_mask))

        if n_improved > 0:
            alpha = 0.3
            self._ema_F = float(np.clip(alpha * self.F + (1 - alpha) * self._ema_F, 0.1, 2.0))
            self._ema_CR = float(np.clip(alpha * self.CR + (1 - alpha) * self._ema_CR, 0.05, 0.99))
            blend = min(0.4, 0.1 + 0.05 * n_improved)
            self.F = float(np.clip(blend * self._ema_F + (1 - blend) * self.F, 0.3, 1.5))
            self.CR = float(np.clip(blend * self._ema_CR + (1 - blend) * self.CR, 0.1, 0.95))
        else:
            stagnation_level = min(float(self.stagnation_count) / max(1, float(self.stagnation_limit)), 1.0)
            F_expansion = 1.0 + stagnation_level * 0.5
            CR_increase = stagnation_level * 0.15
            self.F = float(np.clip(self.F + np.random.uniform(-0.1, 0.2) * F_expansion, 0.3 * F_expansion, 2.0))
            self.CR = float(np.clip(self.CR + np.random.uniform(-0.1, 0.1) + CR_increase, 0.1, 0.99))

        if hasattr(self, '_prev_diversity'):
            diversity_change = self._compute_diversity_batch(population) - self._prev_diversity
            if diversity_change < -0.01 * self.NP:
                self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(0.01, 0.03), 0.05, 0.3))
            elif diversity_change > 0.01 * self.NP:
                self.p_best_rate = float(np.clip(self.p_best_rate - np.random.uniform(0.01, 0.03), 0.05, 0.3))
        self._prev_diversity = self._compute_diversity_batch(population)

    def _adapt_v07(self, population, fitness, mutants, trials):
        """Variant 07 style: Success-history based adaptation with diversity awareness"""
        if not hasattr(self, '_F_success_history'):
            self._F_success_history = []
            self._CR_success_history = []
            self._improvement_streak = 0

        if len(trials) == len(fitness):
            trial_improvements = fitness - trials
            improved = trial_improvements > 0
            n_improved = int(np.sum(improved))

            if n_improved > 0:
                self._improvement_streak += 1
                self._F_success_history.append(float(self.F))
                self._CR_success_history.append(float(self.CR))
            else:
                self._improvement_streak = max(0, self._improvement_streak - 1)

        max_history = 20
        if len(self._F_success_history) > max_history:
            self._F_success_history = self._F_success_history[-max_history:]
            self._CR_success_history = self._CR_success_history[-max_history:]

        centroids = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroids, axis=1)
        pop_range = np.ptp(population, axis=0).mean() + 1e-10
        diversity = float(np.mean(distances) / pop_range)

        low_diversity = diversity < 0.15
        stagnant = self._improvement_streak < 2 and self.generation > 10

        if len(self._F_success_history) >= 3:
            success_F_mean = float(np.mean(self._F_success_history))
            F_base = 0.5 * self.F + 0.5 * success_F_mean
        else:
            F_base = self.F

        if low_diversity or stagnant:
            F_target = float(np.clip(F_base * 1.8, 0.6, 2.0))
        else:
            F_target = float(np.clip(F_base * 0.9 + 0.1 * np.random.uniform(0.4, 1.0), 0.3, 1.5))

        self.F = float(np.clip(F_target + np.random.randn() * 0.05, 0.1, 2.0))

        if len(self._CR_success_history) >= 3:
            success_CR_mean = float(np.mean(self._CR_success_history))
            CR_base = 0.5 * self.CR + 0.5 * success_CR_mean
        else:
            CR_base = self.CR

        if low_diversity:
            CR_target = float(np.clip(CR_base * 0.6, 0.1, 0.5))
        elif stagnant:
            CR_target = float(np.clip(CR_base * 1.3, 0.5, 0.95))
        else:
            CR_target = float(np.clip(CR_base * 0.95 + 0.05 * np.random.uniform(0.3, 0.9), 0.1, 0.95))

        self.CR = float(np.clip(CR_target + np.random.randn() * 0.05, 0.05, 0.98))

        if stagnant:
            delta = np.random.uniform(-0.08, -0.02)
        elif self._improvement_streak > 5:
            delta = np.random.uniform(0.02, 0.08)
        else:
            delta = np.random.uniform(-0.03, 0.03)

        self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.03, 0.4))

    def _adapt_original(self, population, fitness, mutants, trials):
        """Original simple adaptation: random sampling from ranges"""
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:self.NP // 2]

        F_candidates = np.random.uniform(0.4, 1.0, size=5)
        self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))

        if np.random.rand() < 0.2:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3))

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
        return (diversity < self.diversity_threshold or self.stagnation_count > self.stagnation_limit)

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