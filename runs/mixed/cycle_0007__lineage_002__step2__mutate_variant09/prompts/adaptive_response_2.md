```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation strategy works best during optimization.
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

        # === Adaptive Operator Selection (Thompson Sampling) for _mutate_variant09 ===
        self.num_variant09_operators = 5
        self.operator_names = ['variant_02', 'variant_04', 'variant_08', 'variant_09', 'variant_10']
        # Beta distribution parameters for Thompson Sampling
        self.operator_alpha = np.ones(self.num_variant09_operators, dtype=float)
        self.operator_beta = np.ones(self.num_variant09_operators, dtype=float)
        # Sliding window reward tracking
        self.reward_window_size = 30
        self.operator_rewards = [[] for _ in range(self.num_variant09_operators)]
        self.current_variant09_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []

        # Archive for variant_01 (JADE-style)
        self.archive = None
        self._pending_archive_additions = None
        
        # Per-operator improvement tracking
        self.operator_successes = np.zeros(self.num_variant09_operators, dtype=float)
        self.operator_attempts = np.zeros(self.num_variant09_operators, dtype=float)
        self.operator_recent_improvements = [[] for _ in range(self.num_variant09_operators)]

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
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

        self.operator_successes[operator_idx] = float(self.operator_successes[operator_idx]) + max(float(reward > 0), 0.0)
        self.operator_attempts[operator_idx] = float(self.operator_attempts[operator_idx]) + 1.0

        recent_improvements = self.operator_recent_improvements[operator_idx]
        recent_improvements.append(float(reward))
        if len(recent_improvements) > 20:
            recent_improvements.pop(0)

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        self.current_variant09_idx = self._select_operator_thompson()

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

            self._update_operator_rewards(self.current_variant09_idx, reward)

            if self.current_variant09_idx == 1 and self.archive is not None and len(self.archive) > 0:
                self._update_archive(population, fitness, trials, trial_fitness)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            if self.generation - self.last_operator_switch_gen >= self.operator_min_switch_gens:
                new_op = self._select_operator_thompson()
                if new_op != self.current_variant09_idx:
                    self.current_variant09_idx = new_op
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

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
        """Dispatch to selected mutation strategy"""
        if self.current_variant09_idx == 0:
            return self._mutate_variant09_v02(population, fitness)
        elif self.current_variant09_idx == 1:
            return self._mutate_variant09_v04(population, fitness)
        elif self.current_variant09_idx == 2:
            return self._mutate_variant09_v08(population, fitness)
        elif self.current_variant09_idx == 3:
            return self._mutate_variant09_v09(population, fitness)
        else:
            return self._mutate_variant09_v10(population, fitness)

    def _mutate_variant09_v02(self, population, fitness):
        """Variant 09: Mean-Reversion Opposition DE — escape local optima via centroid pull and opposition"""
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

        centroid = np.mean(population, axis=0)
        search_center = 0.5 * (self.lower + self.upper)
        opposition_vector = 2.0 * search_center - population

        is_stagnant = self.stagnation_count > 10
        stagnation_factor = min(self.stagnation_count / 50.0, 1.0) if is_stagnant else 0.0

        w_mean = 0.1 + 0.4 * stagnation_factor
        w_opp = 0.3 * stagnation_factor
        w_best = 0.5
        w_diff = 0.5

        base_direction = w_best * (p_best - population) + w_diff * (population[r1] - population[r2])
        mean_direction = w_mean * (centroid - population)
        opp_direction = w_opp * (opposition_vector - population)

        F_base = float(self.F)
        F_escape = float(np.clip(F_base * (1.0 + stagnation_factor * 0.5), 0.5, 2.0))

        donors = population + F_escape * (base_direction + mean_direction + opp_direction)

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant09_v04(self, population, fitness):
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
            w_best = w_best / total_w
            w_centroid = w_centroid / total_w
            w_rand = w_rand / total_w

        donors = (w_best[:, np.newaxis] * donor_best +
                  w_centroid[:, np.newaxis] * donor_centroid +
                  w_rand[:, np.newaxis] * donor_rand)

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant09_v08(self, population, fitness):
        """Variant 09: Centroid Repulsion — escape local optima by mutating away from population center"""
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

        centroid = np.mean(population, axis=0)
        space_diameter = float(np.linalg.norm(self.upper - self.lower)) + 1e-10

        best_sol = self.best_solution if self.best_solution is not None else p_best[0]
        centroid_to_best = best_sol - centroid
        dist_centroid_best = float(np.linalg.norm(centroid_to_best)) + 1e-10

        anti_centroid = 2.0 * centroid - best_sol

        stagnation = float(self.stagnation_count)
        repulsion_scale = float(np.clip(1.5 + min(stagnation * 0.05, 2.0), 1.5, 3.5))

        repulsion_direction = anti_centroid - centroid
        repulsion_norm = float(np.linalg.norm(repulsion_direction)) + 1e-10
        unit_repulsion = repulsion_direction / repulsion_norm

        target = centroid + unit_repulsion * dist_centroid_best * repulsion_scale
        target = np.clip(target, self.lower, self.upper)

        F_large = float(np.clip(0.8 + min(stagnation * 0.03, 1.0), 0.8, 2.0))
        F_secondary = float(np.clip(self.F + 0.2, 0.5, 2.0))

        donors = population + F_large * (target - population) + F_secondary * (population[r1] - population[r2])

        if stagnation > 10:
            diversity_noise = np.random.randn(self.NP, self.dim)
            diversity_scale = float(min(0.2 * np.log1p(stagnation), 2.0))
            donors = donors + diversity_scale * diversity_noise

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant09_v09(self, population, fitness):
        """Variant 09: Opposition-Based DE with centroid reflection and restart injection"""
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

        base_mutants = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        centroid = np.mean(population, axis=0)
        lower_arr = np.array(self.lower)
        upper_arr = np.array(self.upper)
        range_arr = upper_arr - lower_arr

        opposition = lower_arr + upper_arr - base_mutants

        alpha = np.random.rand(self.NP, 1)
        donors = alpha * base_mutants + (1 - alpha) * opposition

        if self.stagnation_count > 20:
            n_inject = max(1, self.NP // 4)
            inject_idx = np.random.randint(0, self.NP, size=n_inject)
            inject = np.random.uniform(self.lower, self.upper, (n_inject, self.dim))
            donors[inject_idx] = inject

        if self.stagnation_count > 10:
            perturb_mask = np.random.rand(self.NP, self.dim) < 0.3
            perturb = np.random.randn(self.NP, self.dim) * 0.5 * (self.stagnation_count / 50.0)
            donors = donors + perturb_mask * perturb

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant09_v10(self, population, fitness):
        """Variant 09: Opposition-guided mutation with restart and niching"""
        pop_dim = population.shape

        needs_restart = (self.stagnation_count > 40 or 
                         float(np.min(fitness)) > 1e2)

        if needs_restart and self.best_solution is not None:
            opp_pop = self.lower + self.upper - population

            escape_dir = np.random.randn(self.NP, self.dim)
            escape_dir = escape_dir / (np.linalg.norm(escape_dir, axis=1, keepdims=True) + 1e-10)
            escape_scale = np.random.uniform(10, 50, size=(self.NP, 1))
            escape_pop = np.clip(self.best_solution + escape_dir * escape_scale, self.lower, self.upper)

            selector = np.random.randint(0, 3, size=self.NP)
            restart_pop = np.where(selector[:, None] == 0, population,
                                  np.where(selector[:, None] == 1, opp_pop, escape_pop))

            sorted_idx = np.argsort(fitness)
            n_keep = max(5, self.NP // 4)
            keep_indices = sorted_idx[:n_keep]

            fill_pop = np.vstack([
                restart_pop[keep_indices],
                self.lower + np.random.rand(self.NP - n_keep, self.dim) * (self.upper - self.lower)
            ])
            population = np.clip(fill_pop, self.lower, self.upper)
            fitness = np.full(self.NP, np.inf)
            self.stagnation_count = 0

        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        best_fitness_val = float(fitness[sorted_idx[0]])
        niche_mask = fitness < (best_fitness_val * 1.5 + 1.0)
        niche_indices = np.where(niche_mask)[0]
        if len(niche_indices) < 2:
            niche_indices = best_indices

        p_best_idx = niche_indices[np.random.randint(0, len(niche_indices), size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        opp_r1 = self.lower + self.upper - population[r1]
        opp_r2 = self.lower + self.upper - population[r2]

        if self.stagnation_count > 10:
            F_adapt = float(np.clip(self.F * (1.0 + 0.3 * np.log1p(self.stagnation_count)), 0.5, 2.5))
        else:
            F_adapt = float(np.clip(self.F, 0.5, 1.5))

        base_mutation = p_best - population
        diff_opp = opp_r1 - opp_r2
        diff_current = population[r1] - population[r2]

        donors = population + F_adapt * base_mutation + 0.5 * F_adapt * (0.6 * diff_current + 0.4 * diff_opp)

        opp_perturb = np.random.rand(self.NP, self.dim) < 0.2
        perturb_scale = np.random.uniform(1, 10, size=(self.NP, self.dim))
        opp_offset = (self.lower + self.upper - donors) * opp_perturb * perturb_scale * 0.1
        donors = donors + opp_offset

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
```