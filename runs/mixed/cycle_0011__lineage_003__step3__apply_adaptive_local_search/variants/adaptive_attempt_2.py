import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation AND local search strategy works best during optimization.
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

        # === Adaptive Local Search Operator Selection ===
        self.num_ls_operators = 8
        self.ls_operator_names = [
            'original', 'variant_02', 'variant_03', 'variant_04',
            'variant_05', 'variant_06', 'variant_07', 'variant_10'
        ]
        # Beta distribution parameters for Thompson Sampling (local search)
        self.ls_alpha = np.ones(self.num_ls_operators)
        self.ls_beta = np.ones(self.num_ls_operators)
        # Sliding window reward tracking (local search)
        self.ls_reward_window_size = 20
        self.ls_operator_rewards = [[] for _ in range(self.num_ls_operators)]
        self.current_ls_operator_idx = 0
        self.last_ls_switch_gen = 0
        self.ls_min_switch_gens = 3
        self.ls_switches = []
        self.ls_use_counts = np.zeros(self.num_ls_operators, dtype=int)
        self.ls_success_counts = np.zeros(self.num_ls_operators, dtype=int)

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _select_ls_operator_thompson(self):
        """Select local search operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.ls_alpha, self.ls_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
            """Compute reward based on relative improvement quality (log-scaled)"""
            # Guard against invalid values
            fitness = np.clip(fitness, -1e50, 1e50)
            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

            improvements = fitness - trial_fitness
            n_improved = int(np.sum(improvements > 0))
            total_improved = float(n_improved)

            if n_improved == 0:
                return -0.1

            # Compute relative improvement (fractional gain)
            relative_imp = improvements[improvements > 0] / (np.abs(fitness[improvements > 0]) + 1e-10)

            # Log-scaled reward: small gains on hard problems valued similarly to large gains on easy ones
            log_rewards = np.log1p(relative_imp)
            avg_log_reward = float(np.mean(log_rewards))

            # Consistency bonus: reward operators that improve many individuals
            consistency_ratio = total_improved / max(float(len(fitness)), 1.0)
            consistency_bonus = 0.15 * consistency_ratio

            # Combined reward: log-scale captures difficulty-normalized progress
            reward = avg_log_reward + consistency_bonus

            return float(np.clip(reward, -0.5, 1.0))

    def _compute_ls_reward(self, initial_fitness, final_fitness):
        """Compute reward for local search operator based on improvement"""
        initial_fitness = float(np.clip(initial_fitness, -1e50, 1e50))
        final_fitness = float(np.clip(final_fitness, -1e50, 1e50))
        
        if not np.isfinite(initial_fitness) or not np.isfinite(final_fitness):
            return -0.5
        
        improvement = initial_fitness - final_fitness
        
        if improvement <= 0:
            return -0.2
        
        # Relative improvement (fractional gain)
        relative_imp = improvement / (np.abs(initial_fitness) + 1e-10)
        
        # Log-scaled reward
        log_reward = float(np.log1p(relative_imp))
        
        # Bonus for significant improvement
        if relative_imp > 0.5:
            log_reward += 0.3
        elif relative_imp > 0.1:
            log_reward += 0.15
        
        return float(np.clip(log_reward, -0.3, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(float(reward))

        # Maintain sliding window
        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        # Update Beta distribution parameters based on recent performance
        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = float(np.mean(recent))

            # Update alpha (successes + 1) and beta (failures + 1)
            self.operator_alpha[operator_idx] = 1.0 + success_rate * 5.0 + avg_reward * 2.0
            self.operator_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0

            # Ensure valid parameters
            self.operator_alpha[operator_idx] = float(np.clip(self.operator_alpha[operator_idx], 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(self.operator_beta[operator_idx], 0.1, 100.0))

    def _update_ls_operator_rewards(self, operator_idx, reward):
        """Update rewards for local search operator with sliding window"""
        rewards = self.ls_operator_rewards[operator_idx]
        rewards.append(float(reward))

        # Maintain sliding window
        if len(rewards) > self.ls_reward_window_size:
            rewards.pop(0)

        # Update Beta distribution parameters based on recent performance
        recent = rewards[-min(8, len(rewards)):]
        if len(recent) >= 2:
            successes = sum(1 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = float(np.mean(recent))

            # Update alpha and beta
            self.ls_alpha[operator_idx] = 1.0 + success_rate * 4.0 + avg_reward * 2.0
            self.ls_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 4.0 - avg_reward * 1.5

            # Ensure valid parameters
            self.ls_alpha[operator_idx] = float(np.clip(self.ls_alpha[operator_idx], 0.1, 100.0))
            self.ls_beta[operator_idx] = float(np.clip(self.ls_beta[operator_idx], 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        # Guard against invalid initial fitness
        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self._select_operator_thompson()
        self._select_ls_operator_thompson()

        while not stopping_condition():
            mutants = self._mutate_current_to_pbest_with_culture(population, fitness)
            mutants = np.clip(mutants, self.lower, self.upper)

            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            # Guard against invalid trial fitness
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

            # Update archive if using variant_01
            if self.current_operator_idx == 1 and self.archive is not None and len(self.archive) > 0:
                self._update_archive(population, fitness, trials, trial_fitness)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                initial_fitness_before_ls = self.best_fitness
                self._apply_adaptive_local_search(func)
                final_fitness_after_ls = self.best_fitness
                
                # Compute and update local search reward
                if final_fitness_after_ls < initial_fitness_before_ls:
                    ls_reward = self._compute_ls_reward(initial_fitness_before_ls, final_fitness_after_ls)
                    self.ls_success_counts[self.current_ls_operator_idx] += 1
                else:
                    ls_reward = self._compute_ls_reward(initial_fitness_before_ls, final_fitness_after_ls)
                
                self._update_ls_operator_rewards(self.current_ls_operator_idx, ls_reward)
                self.ls_use_counts[self.current_ls_operator_idx] += 1
                
                # Local search operator switching with minimum generations between switches
                if self.generation - self.last_ls_switch_gen >= self.ls_min_switch_gens:
                    new_ls_op = self._select_ls_operator_thompson()
                    if new_ls_op != self.current_ls_operator_idx:
                        self.current_ls_operator_idx = new_ls_op
                        self.last_ls_switch_gen = self.generation
                        self.ls_switches.append((self.generation, new_ls_op))

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
        # Initialize archive if needed
        if self.archive is None:
            self.archive = np.zeros((0, self.dim))

        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        p_best_indices = sorted_idx[:n_best]
        p_best_idx = p_best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        # Generate random indices r1, r2
        r1 = np.random.randint(0, self.NP, size=self.NP)
        r2 = np.zeros(self.NP, dtype=int)
        for i in range(self.NP):
            while True:
                r2[i] = np.random.randint(0, self.NP)
                if r2[i] != i and r2[i] != r1[i]:
                    break

        # JADE mutation with archive
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

        # Bounded random F per individual
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

        # Adaptive F when stagnant
        F_adapted = self.F
        if self.stagnation_count > 10:
            F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count - 10)), 2.0)

        base = population + F_adapted * (global_best - population) + F_adapted * (population[r1] - population[r2])

        # Escalating random perturbation when stagnant
        if self.stagnation_count > 5:
            perturbation_scale = min(0.3 * np.log1p(self.stagnation_count), 2.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            base = base + random_perturbation

        # Cultural memory blending
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

        # Compute centroid of top 20% as exploration target
        top_percentile = max(1, self.NP // 5)
        top_indices = sorted_idx[:top_percentile]
        centroid = np.mean(population[top_indices], axis=0)

        # Generate THREE candidate donors per individual
        # Candidate 1: Best-directed (exploitation)
        donor_best = population + self.F * (p_best - population)
        # Candidate 2: Centroid-directed (diversification)
        donor_centroid = population + self.F * (centroid - population)
        # Candidate 3: Standard rand/diff (random exploration)
        donor_rand = population + self.F * (population[r1] - population[r2])

        # Diversity-weighted composite (no extra fitness evals needed)
        # Higher diversity contribution = more different from parent = better explorer
        dist_best = np.linalg.norm(donor_best - population, axis=1) + 1e-10
        dist_centroid = np.linalg.norm(donor_centroid - population, axis=1) + 1e-10
        dist_rand = np.linalg.norm(donor_rand - population, axis=1) + 1e-10

        # Normalize diversity scores
        total_dist = dist_best + dist_centroid + dist_rand + 1e-10
        w_best = dist_best / total_dist
        w_centroid = dist_centroid / total_dist
        w_rand = dist_rand / total_dist

        # Adaptive weight adjustment based on stagnation
        if self.stagnation_count > 10:
            # When stuck: boost centroid exploration, reduce best-pull
            w_best = np.clip(w_best * 0.5, 0.1, 0.6)
            w_centroid = np.clip(w_centroid * 1.5, 0.2, 0.6)
            w_rand = np.clip(w_rand * 1.2, 0.1, 0.4)
            # Renormalize
            total_w = w_best + w_centroid + w_rand
            w_best /= total_w
            w_centroid /= total_w
            w_rand /= total_w

        # Composite donors
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
        # Track weighted improvement rate for parameter adaptation
        if not hasattr(self, '_adapt_weighted_improvement'):
            self._adapt_weighted_improvement = 0.0
            self._adapt_fitness_spread = 1.0
            self._adapt_history_F = []
            self._adapt_history_CR = []

        # Compute directional improvement: is best fitness improving?
        current_best = float(np.min(fitness))
        if hasattr(self, '_prev_best_fitness') and np.isfinite(self._prev_best_fitness):
            direction = self._prev_best_fitness - current_best
            # Weighted moving average with exponential decay
            decay = 0.7
            self._adapt_weighted_improvement = decay * self._adapt_weighted_improvement + (1 - decay) * direction
        self._prev_best_fitness = current_best

        # Track fitness spread (convergence indicator)
        finite_fitness = fitness[np.isfinite(fitness)]
        if len(finite_fitness) > 1:
            spread = float(np.std(finite_fitness)) + 1e-10
            self._adapt_fitness_spread = 0.9 * self._adapt_fitness_spread + 0.1 * spread

        # Normalize improvement by spread to get adaptive signal
        norm_signal = self._adapt_weighted_improvement / (self._adapt_fitness_spread + 1e-10)

        # Adaptive F: exploit (lower F) when improving, explore (higher F) when stagnant
        if norm_signal > 0.1:
            # Making progress: focus on exploitation with finer steps
            F_base = 0.4 + 0.3 * np.random.rand()
            F_scale = 0.05
        elif norm_signal < -0.1:
            # Stagnant: increase exploration
            F_base = 0.7 + 0.4 * np.random.rand()
            F_scale = 0.15
        else:
            # Neutral: balanced approach
            F_base = 0.5 + 0.3 * np.random.rand()
            F_scale = 0.1

        F_candidate = F_base + np.random.randn() * F_scale
        new_F = float(np.clip(F_candidate, 0.1, 2.0))

        # Adaptive CR: higher CR when improving (combine more from mutant), lower when stagnant
        if norm_signal > 0.1:
            CR_base = 0.7 + 0.2 * np.random.rand()
        elif norm_signal < -0.1:
            CR_base = 0.3 + 0.2 * np.random.rand()
        else:
            CR_base = 0.5 + 0.2 * np.random.rand()

        CR_candidate = CR_base + np.random.randn() * 0.1
        new_CR = float(np.clip(CR_candidate, 0.05, 0.98))

        # Momentum: blend with history to reduce oscillation
        if len(self._adapt_history_F) > 0:
            momentum = 0.3
            new_F = momentum * np.mean(self._adapt_history_F[-5:]) + (1 - momentum) * new_F
            new_CR = momentum * np.mean(self._adapt_history_CR[-5:]) + (1 - momentum) * new_CR

        self.F = float(np.clip(new_F, 0.1, 2.0))
        self.CR = float(np.clip(new_CR, 0.05, 0.98))

        # Track history
        self._adapt_history_F.append(self.F)
        self._adapt_history_CR.append(self.CR)
        if len(self._adapt_history_F) > 20:
            self._adapt_history_F.pop(0)
            self._adapt_history_CR.pop(0)

        # Adapt p_best_rate based on stagnation
        if self.stagnation_count > 15:
            delta = np.random.uniform(0.02, 0.08)
            self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.05, 0.4))
        elif self.stagnation_count > 5:
            if np.random.rand() < 0.3:
                self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.03, 0.03), 0.05, 0.3))

    def _apply_adaptive_local_search(self, func):
        """Dispatch to selected local search strategy using Thompson Sampling"""
        if self.best_solution is None:
            return

        # Dispatch based on selected operator index
        if self.current_ls_operator_idx == 0:
            self._ls_original(func)
        elif self.current_ls_operator_idx == 1:
            self._ls_variant02(func)
        elif self.current_ls_operator_idx == 2:
            self._ls_variant03(func)
        elif self.current_ls_operator_idx == 3:
            self._ls_variant04(func)
        elif self.current_ls_operator_idx == 4:
            self._ls_variant05(func)
        elif self.current_ls_operator_idx == 5:
            self._ls_variant06(func)
        elif self.current_ls_operator_idx == 6:
            self._ls_variant07(func)
        else:
            self._ls_variant10(func)

    # =========================================================================
    # LOCAL SEARCH VARIANT 0: Original (baseline)
    # =========================================================================
    def _ls_original(self, func):
        """Original local search strategy"""
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
            self.best_fitness = float(current_fitness)
            self.best_solution = center.copy()
            self.local_radius = float(min(radius * 1.2, 10.0))
        else:
            self.local_radius = float(max(radius * 0.8, 0.1))

    # =========================================================================
    # LOCAL SEARCH VARIANT 2: Direction history learning (1 win)
    # =========================================================================
    def _ls_variant02(self, func):
        """Variant 02: Per-dimension adaptive step sizes with direction history"""
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness

        # Per-dimension adaptive step sizes
        if not hasattr(self, '_ls_dim_scales'):
            self._ls_dim_scales = np.ones(self.dim) * self.local_radius

        # Initialize direction history for learning
        if not hasattr(self, '_ls_direction_history'):
            self._ls_direction_history = []

        max_iter = self.local_max_iter * 3

        for iteration in range(max_iter):
            improved = False

            # Exploit learned successful directions first
            if len(self._ls_direction_history) > 0:
                for direction, step_size in reversed(self._ls_direction_history[-5:]):
                    cand = np.clip(center + step_size * direction, self.lower, self.upper)
                    fit = float(self._eval_wrapper_batch(cand.reshape(1, -1), func)[0])

                    if np.isfinite(fit) and fit < current_fitness:
                        self._ls_direction_history.append((direction.copy(), step_size * 1.3))
                        if len(self._ls_direction_history) > 15:
                            self._ls_direction_history.pop(0)

                        center = cand.copy()
                        current_fitness = fit
                        improved = True
                        break

            # Systematic coordinate descent for unexplored improvements
            if not improved:
                dim = iteration % self.dim
                step = self._ls_dim_scales[dim]

                for sign in [1, -1]:
                    cand = center.copy()
                    cand[dim] = float(np.clip(cand[dim] + sign * step, self.lower[dim], self.upper[dim]))
                    fit = float(self._eval_wrapper_batch(cand.reshape(1, -1), func)[0])

                    if np.isfinite(fit) and fit < current_fitness:
                        direction = np.zeros(self.dim)
                        direction[dim] = sign
                        self._ls_direction_history.append((direction.copy(), step))
                        if len(self._ls_direction_history) > 15:
                            self._ls_direction_history.pop(0)

                        self._ls_dim_scales[dim] = min(step * 1.5, 10.0)
                        center = cand.copy()
                        current_fitness = fit
                        improved = True
                        break

                if not improved:
                    self._ls_dim_scales[dim] = max(step * 0.5, 1e-8)

            if not improved and iteration > self.dim * 2:
                break

        if current_fitness < self.best_fitness:
            self.best_fitness = float(current_fitness)
            self.best_solution = center.copy()
            self.local_radius = float(np.clip(np.mean(self._ls_dim_scales), 0.1, 10.0))
        else:
            self.local_radius = float(np.clip(np.mean(self._ls_dim_scales) * 0.8, 0.1, 10.0))

    # =========================================================================
    # LOCAL SEARCH VARIANT 3: Dense direction sampling with extrapolation (1 win)
    # =========================================================================
    def _ls_variant03(self, func):
        """Variant 03: Dense direction sampling with extrapolation"""
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        # Generate dense set of test directions (more candidates = better coverage)
        num_directions = max(16, 2 * self.dim)
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)

        for _ in range(self.local_max_iter):
            if radius < 1e-6:
                break

            # Build candidate set: multi-radius, multi-direction grid
            candidates = [center]
            for r in [radius, radius * 3.0]:  # Test fine AND coarse scale
                for angle in angles:
                    # Create direction in a random 2D subspace for this angle
                    direction = np.zeros(self.dim)
                    d1 = np.random.randint(0, self.dim)
                    d2 = np.random.randint(0, self.dim - 1)
                    if d2 >= d1:
                        d2 += 1
                    direction[d1] = np.cos(angle)
                    direction[d2] = np.sin(angle)
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    candidates.append(np.clip(center + r * direction, self.lower, self.upper))

            candidates = np.vstack(candidates)
            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = np.argmin(fit_candidates)
            best_local_fitness = float(fit_candidates[best_local_idx])

            if best_local_fitness < current_fitness:
                new_center = candidates[best_local_idx]
                # Compute vector from old center to new center
                step_vector = new_center - center
                step_norm = np.linalg.norm(step_vector) + 1e-10

                if step_norm > 1e-10:
                    # Try extrapolating further along the improvement direction
                    extrapolated = np.clip(center + step_vector * 2.0, self.lower, self.upper)
                    extra_fitness = float(self._eval_wrapper_batch(extrapolated[np.newaxis], func)[0])

                    if extra_fitness < best_local_fitness:
                        center = extrapolated
                        current_fitness = extra_fitness
                    else:
                        center = new_center
                        current_fitness = best_local_fitness
                else:
                    center = new_center
                    current_fitness = best_local_fitness

                # Expand radius on success
                radius = min(radius * 1.8, 10.0)
            else:
                # Contract more aggressively on failure
                radius *= 0.3

        if current_fitness < self.best_fitness:
            self.best_fitness = float(current_fitness)
            self.best_solution = center.copy()
            self.local_radius = float(min(radius * 1.2, 10.0))
        else:
            self.local_radius = float(max(radius * 0.8, 0.1))

    # =========================================================================
    # LOCAL SEARCH VARIANT 4: Multi-scale restart with memory (2 wins)
    # =========================================================================
    def _ls_variant04(self, func):
        """Variant 04: Multi-scale restart with historical best memory"""
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness

        # Adaptive radius starts from current setting, with bounds
        radius = self.local_radius
        max_radius = float(max(self.upper - self.lower) * 0.1)
        min_radius = float((self.upper - self.lower).min() * 1e-5)

        # Multi-scale restart with memory
        if not hasattr(self, '_local_search_restarts'):
            self._local_search_restarts = 0
        if not hasattr(self, '_local_best_history'):
            self._local_best_history = []

        max_restarts = 3
        max_iter_per_restart = self.local_max_iter

        for restart in range(max_restarts):
            for _ in range(max_iter_per_restart):
                if radius < min_radius:
                    break

                # Generate N+1 candidates: N coordinate perturbations + 1 center
                candidates = np.tile(center, (self.dim + 1, 1))
                for d in range(self.dim):
                    candidates[d, d] = float(np.clip(center[d] + radius, self.lower[d], self.upper[d]))

                fit_candidates = self._eval_wrapper_batch(candidates, func)
                best_local_idx = int(np.argmin(fit_candidates))
                best_local_fitness = float(fit_candidates[best_local_idx])

                if best_local_idx < self.dim:
                    # Improvement found: update and expand radius
                    center[best_local_idx] = candidates[best_local_idx, best_local_idx]
                    current_fitness = best_local_fitness
                    radius = min(radius * 1.5, max_radius)
                else:
                    # No improvement: contract radius
                    radius *= 0.5

            # Check if converged
            if current_fitness >= self.best_fitness and radius < min_radius * 2:
                break

            # Restart from historical best if stuck
            if restart < max_restarts - 1:
                if len(self._local_best_history) > 0:
                    # Pick random historical best (different from current center)
                    hist_best = self._local_best_history[np.random.randint(0, len(self._local_best_history))]
                    if np.linalg.norm(hist_best - center) > radius:
                        center = hist_best.copy()
                        self._local_search_restarts += 1
                        radius = max(radius * 2, min_radius * 10)
                        continue
                break

        # Update global best if improved
        if current_fitness < self.best_fitness:
            self.best_fitness = float(current_fitness)
            self.best_solution = center.copy()
            # Record in history for restarts
            self._local_best_history.append(center.copy())
            if len(self._local_best_history) > 10:
                self._local_best_history.pop(0)

        # Adapt local radius for next call
        self.local_radius = float(np.clip(radius, min_radius, max_radius))

    # =========================================================================
    # LOCAL SEARCH VARIANT 5: Pattern directions with cultural memory (2 wins)
    # =========================================================================
    def _ls_variant05(self, func):
        """Variant 05: Orthogonal pattern directions with cultural memory"""
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        # Generate orthogonal pattern directions for systematic exploration
        def _generate_pattern_directions(n_dims, n_dirs):
            dirs = []
            for _ in range(n_dirs):
                d = np.random.randn(n_dims)
                # Orthogonalize against existing directions via Gram-Schmidt
                for existing in dirs:
                    d = d - np.dot(existing, d) * existing / (np.dot(existing, existing) + 1e-10)
                norm = np.linalg.norm(d)
                if norm > 1e-8:
                    dirs.append(d / norm)
            return dirs

        n_pattern_dirs = min(8, self.dim)
        pattern_dirs = _generate_pattern_directions(self.dim, n_pattern_dirs)

        candidates = [center.copy()]

        # Add pattern direction perturbations
        for direction in pattern_dirs:
            candidates.append(np.clip(center + radius * direction, self.lower, self.upper))
            candidates.append(np.clip(center - radius * direction, self.lower, self.upper))

        # Add population-guided directions: toward/from best-known solutions
        if hasattr(self, 'cultural_memory') and len(self.cultural_memory) > 0:
            weights = self.cultural_weights[:len(self.cultural_memory)]
            if np.sum(weights) > 1e-10:
                cultural_dir = self.cultural_memory[:len(self.cultural_memory)] * weights[:, np.newaxis]
                cultural_dir = np.sum(cultural_dir, axis=0)
                norm_cd = np.linalg.norm(cultural_dir)
                if norm_cd > 1e-10:
                    cultural_dir = cultural_dir / norm_cd
                    candidates.append(np.clip(center + radius * cultural_dir, self.lower, self.upper))
                    candidates.append(np.clip(center - radius * cultural_dir, self.lower, self.upper))

        # Add intermediate radius candidates for fine-grained search
        for direction in pattern_dirs[:3]:
            candidates.append(np.clip(center + radius * 0.5 * direction, self.lower, self.upper))
            candidates.append(np.clip(center - radius * 0.5 * direction, self.lower, self.upper))

        candidates = np.vstack(candidates)
        fit_candidates = self._eval_wrapper_batch(candidates, func)
        best_local_idx = np.argmin(fit_candidates)
        best_local_fitness = float(fit_candidates[best_local_idx])

        # Update center and radius based on improvement
        if best_local_fitness < current_fitness:
            improvement = current_fitness - best_local_fitness
            center = candidates[best_local_idx].copy()
            current_fitness = best_local_fitness

            # If improvement is small, reduce radius for finer search
            if improvement < 1e-4 * abs(current_fitness + 1e-10):
                radius = max(radius * 0.5, 1e-6)
            else:
                radius = min(radius * 1.2, 10.0)
        else:
            radius *= 0.5

        # Update best if improved
        if current_fitness < self.best_fitness:
            self.best_fitness = float(current_fitness)
            self.best_solution = center.copy()
            self.local_radius = float(max(radius, 0.1))
        else:
            self.local_radius = float(max(radius * 0.9, 0.05))

    # =========================================================================
    # LOCAL SEARCH VARIANT 6: L-BFGS-B (4 wins - most successful)
    # =========================================================================
    def _ls_variant06(self, func):
        """Variant 06: Multi-scale L-BFGS-B with restart"""
        if self.best_solution is None:
            return

        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback to original if scipy not available
            self._ls_original(func)
            return

        def objective(x):
            return float(func(x.reshape(1, -1))[0])

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        bounds = [(float(self.lower[i]), float(self.upper[i])) for i in range(self.dim)]

        # Multi-scale restart: start large, refine progressively
        scales = [max(self.local_radius, 1.0), 0.5, 0.1]
        max_iter_per_scale = [30, 60, 100]

        best_result = None
        best_result_fitness = current_fitness

        for scale, max_iter in zip(scales, max_iter_per_scale):
            x0 = center.copy()

            for restart in range(2):
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': max_iter,
                            'ftol': 0.0,
                            'gtol': 1e-12,
                            'maxfun': max_iter * self.dim * 2,
                            'maxls': 20,
                            'disp': False
                        }
                    )

                    if result.fun < best_result_fitness:
                        best_result = result
                        best_result_fitness = float(result.fun)
                        if best_result_fitness < current_fitness:
                            center = result.x.copy()

                    # Perturb for next restart if no improvement yet
                    if restart == 0 and result.fun >= best_result_fitness - 1e-10:
                        perturbation = np.random.randn(self.dim) * scale * 0.5
                        x0 = np.clip(center + perturbation, self.lower, self.upper)
                    else:
                        break

                except Exception:
                    break

            # Early exit if good enough
            if best_result_fitness <= 1e-10:
                break

        if best_result_fitness < current_fitness:
            self.best_fitness = float(best_result_fitness)
            self.best_solution = center.copy()
            # Adjust radius based on how much improvement achieved
            improvement_ratio = (current_fitness - best_result_fitness) / (abs(current_fitness) + 1e-10)
            if improvement_ratio > 0.5:
                self.local_radius = float(min(self.local_radius * 1.5, 10.0))
            elif improvement_ratio > 0.1:
                self.local_radius = float(max(self.local_radius * 0.9, 0.05))
            else:
                self.local_radius = float(max(self.local_radius * 0.7, 0.01))

    # =========================================================================
    # LOCAL SEARCH VARIANT 7: Random direction sampling with bias (1 win)
    # =========================================================================
    def _ls_variant07(self, func):
        """Variant 07: Random direction sampling with successful direction bias"""
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        # Track successful directions for adaptive bias
        if not hasattr(self, '_ls_successful_dirs'):
            self._ls_successful_dirs = []
        successful_dirs = self._ls_successful_dirs

        for _ in range(self.local_max_iter):
            if radius < 1e-8:
                break

            # Sample many random directions (4*dim or at least 24)
            n_directions = max(24, self.dim * 4)
            directions = np.random.randn(n_directions, self.dim)
            norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10
            directions = directions / norms

            # Build large candidate set: forward, antipodal, center, biased
            candidates = np.clip(center + radius * directions, self.lower, self.upper)

            # Antipodal points for diversity
            n_antipodal = n_directions // 3
            antipodal = np.clip(center - radius * directions[:n_antipodal], self.lower, self.upper)
            candidates = np.vstack([candidates, antipodal])

            # Include center
            candidates = np.vstack([candidates, center.reshape(1, -1)])

            # Add biased direction toward best successful direction if available
            if successful_dirs:
                best_dir = successful_dirs[-1]
                biased = np.clip(center + radius * 1.5 * best_dir, self.lower, self.upper)
                candidates = np.vstack([candidates, biased.reshape(1, -1)])

            # Evaluate all candidates in ONE batch
            fit_candidates = self._eval_wrapper_batch(candidates, func)

            # Find best candidate
            best_local_idx = np.argmin(fit_candidates)
            best_local_fitness = float(fit_candidates[best_local_idx])
            best_dir = candidates[best_local_idx] - center

            # Normalize successful direction
            if best_local_fitness < current_fitness and np.linalg.norm(best_dir) > 1e-10:
                dir_norm = np.linalg.norm(best_dir)
                successful_dirs.append(best_dir / dir_norm)
                if len(successful_dirs) > 10:
                    successful_dirs.pop(0)

            # Update center if improvement found
            if best_local_fitness < current_fitness:
                center = candidates[best_local_idx].copy()
                current_fitness = best_local_fitness
                radius = min(radius * 1.8, 10.0)
            else:
                radius *= 0.4

            # Early termination if target reached
            if current_fitness <= 1e-8:
                break

        # Update best solution
        if current_fitness < self.best_fitness:
            self.best_fitness = float(current_fitness)
            self.best_solution = center.copy()
            self.local_radius = float(min(radius * 1.2, 10.0))
        else:
            self.local_radius = float(max(radius * 0.8, 0.1))

    # =========================================================================
    # LOCAL SEARCH VARIANT 10: Golden section search with PCA (2 wins)
    # =========================================================================
    def _ls_variant10(self, func):
        """Variant 10: Golden section search with PCA-based subspace optimization"""
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        if not hasattr(self, '_ls_direction_history'):
            self._ls_direction_history = np.zeros((20, self.dim))
            self._ls_history_idx = 0
            self._ls_history_count = 0

        def golden_section_search(func_obj, x0, direction, a, b, tol=1e-8, max_iter=30):
            phi = (1 + np.sqrt(5)) / 2
            res_phi = 2 - phi
            x1 = a + res_phi * (b - a)
            x2 = b - res_phi * (b - a)
            f1 = float('inf')
            f2 = float('inf')
            try:
                f1 = float(func_obj(x0 + x1 * direction))
            except:
                pass
            try:
                f2 = float(func_obj(x0 + x2 * direction))
            except:
                pass
            for _ in range(max_iter):
                if b - a < tol:
                    break
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = a + res_phi * (b - a)
                    try:
                        f1 = float(func_obj(x0 + x1 * direction))
                    except:
                        f1 = float('inf')
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = b - res_phi * (b - a)
                    try:
                        f2 = float(func_obj(x0 + x2 * direction))
                    except:
                        f2 = float('inf')
            if f1 < f2:
                return x1, f1
            return x2, f2

        improved = False
        for _ in range(self.local_max_iter):
            if radius < 1e-8:
                break
            best_pos = center.copy()
            best_local_fit = current_fitness
            step_reduced = True

            for d in range(self.dim):
                direction = np.zeros(self.dim)
                direction[d] = 1.0
                step_pos = radius
                step_neg = -radius

                try:
                    fit_pos = float(func(center + step_pos * direction))
                except:
                    fit_pos = float('inf')
                try:
                    fit_neg = float(func(center + step_neg * direction))
                except:
                    fit_neg = float('inf')

                if fit_pos < best_local_fit or fit_neg < best_local_fit:
                    if fit_pos < fit_neg:
                        direction[d] = 1.0
                        step_bound = step_pos
                        fit_bound = fit_pos
                    else:
                        direction[d] = -1.0
                        step_bound = -step_neg
                        fit_bound = fit_neg
                    step_reduced = False
                    opt_step, opt_fit = golden_section_search(func, center, direction, 0.0, step_bound)
                    if opt_fit < best_local_fit:
                        best_local_fit = opt_fit
                        best_pos = center + opt_step * direction
                        if abs(opt_step) > 1e-10:
                            dir_vec = opt_step * direction
                            self._ls_direction_history[self._ls_history_idx] = dir_vec
                            self._ls_history_idx = (self._ls_history_idx + 1) % 20
                            self._ls_history_count = min(self._ls_history_count + 1, 20)

            if best_local_fit < current_fitness:
                center = best_pos.copy()
                current_fitness = best_local_fit
                improved = True
                radius = min(radius * 1.5, 10.0)
            else:
                radius *= 0.4

        if improved or current_fitness < self.best_fitness:
            if current_fitness < self.best_fitness:
                self.best_fitness = float(current_fitness)
                self.best_solution = center.copy()
                self.local_radius = float(min(radius * 1.2, 10.0))
            else:
                self.local_radius = float(max(radius * 0.8, 0.1))

            if self._ls_history_count >= 5:
                history = self._ls_direction_history[:self._ls_history_count]
                hist_mean = np.mean(history, axis=0)
                centered = history - hist_mean
                cov = np.cov(centered.T)
                cov = (cov + cov.T) / 2.0
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]
                    if eigvals[0] > 1e-6 and eigvals[0] / (np.sum(eigvals) + 1e-10) > 0.3:
                        R = eigvecs[:, :min(3, self.dim)]
                        center_rot = np.dot(R.T, center)
                        improved_rot = False
                        for _ in range(2):
                            for d in range(min(3, self.dim)):
                                direction = np.zeros(min(3, self.dim))
                                direction[d] = 1.0
                                step_bounds = [radius * 0.5, -radius * 0.5]
                                for sb in step_bounds:
                                    if abs(sb) < 1e-10:
                                        continue
                                    opt_step, opt_fit = golden_section_search(
                                        lambda x: func(center + np.dot(R, x)),
                                        center_rot, direction, 0.0, sb)
                                    if opt_fit < current_fitness:
                                        center_rot = center_rot + opt_step * direction
                                        current_fitness = opt_fit
                                        improved_rot = True
                            if not improved_rot:
                                break
                        center = np.dot(R, center_rot)
                        center = np.clip(center, self.lower, self.upper)
                        if current_fitness < self.best_fitness:
                            self.best_fitness = float(current_fitness)
                            self.best_solution = center.copy()
                except:
                    pass

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
