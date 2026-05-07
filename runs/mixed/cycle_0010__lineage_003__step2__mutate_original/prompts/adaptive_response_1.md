```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation strategy works best during optimization.
    Includes 7 mutation operators based on benchmark results.
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
        self.num_operators = 7
        self.operator_names = ['original', 'variant_02', 'variant_05', 'variant_06',
                               'variant_07', 'variant_08', 'variant_10']
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
        # Track per-operator improvement counts
        self.operator_improvements = [0] * self.num_operators
        self.operator_trials = [0] * self.num_operators

        # Archive for variant_06 (JADE-style)
        self.archive = None

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on relative improvement quality (log-scaled)"""
        # Guard against invalid values
        fitness = np.array(fitness, dtype=np.float64)
        trial_fitness = np.array(trial_fitness, dtype=np.float64)
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        n_improved = int(np.sum(improvements > 0))
        total_improved = float(n_improved)

        if n_improved == 0:
            return -0.1

        # Compute relative improvement (fractional gain)
        pos_mask = improvements > 0
        rel_imp = improvements[pos_mask] / (np.abs(fitness[pos_mask]) + 1e-10)

        # Log-scaled reward: small gains on hard problems valued similarly to large gains on easy ones
        log_rewards = np.log1p(np.clip(rel_imp, -1e10, 1e10))
        avg_log_reward = float(np.mean(log_rewards)) if len(log_rewards) > 0 else 0.0

        # Consistency bonus: reward operators that improve many individuals
        consistency_ratio = total_improved / max(float(len(fitness)), 1.0)
        consistency_bonus = 0.15 * consistency_ratio

        # Combined reward: log-scale captures difficulty-normalized progress
        reward = avg_log_reward + consistency_bonus

        return float(np.clip(reward, -0.5, 1.0))

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
            successes = sum(1.0 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = float(np.mean(recent))

            # Update alpha (successes + 1) and beta (failures + 1)
            self.operator_alpha[operator_idx] = 1.0 + success_rate * 5.0 + avg_reward * 2.0
            self.operator_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0

            # Ensure valid parameters
            self.operator_alpha[operator_idx] = float(np.clip(self.operator_alpha[operator_idx], 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(self.operator_beta[operator_idx], 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        # Guard against invalid initial fitness
        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self.current_operator_idx = self._select_operator_thompson()

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

            # Update archive if using variant_06
            if self.current_operator_idx == 3 and self.archive is not None and len(self.archive) > 0:
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
        result = np.array(result, dtype=np.float64)
        if len(result) < len(population):
            result = np.pad(result, (0, len(population) - len(result)), constant_values=np.inf)
        return result

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
        """Dispatch to selected mutation strategy"""
        if self.current_operator_idx == 0:
            return self._mutate_original(population, fitness)
        elif self.current_operator_idx == 1:
            return self._mutate_variant02(population, fitness)
        elif self.current_operator_idx == 2:
            return self._mutate_variant05(population, fitness)
        elif self.current_operator_idx == 3:
            return self._mutate_variant06(population, fitness)
        elif self.current_operator_idx == 4:
            return self._mutate_variant07(population, fitness)
        elif self.current_operator_idx == 5:
            return self._mutate_variant08(population, fitness)
        else:
            return self._mutate_variant10(population, fitness)

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

    def _mutate_variant02(self, population, fitness):
        """Historical Best-Guided Mutation with Convergence-Aware Exploration"""
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

        # Historical best guidance (NEW: uses global best from ALL generations)
        if self.best_solution is not None:
            hist_best = self.best_solution
            # Compute distance-based weights: closer individuals get stronger pull
            hist_dist = np.linalg.norm(population - hist_best, axis=1) + 1e-10
            hist_weights = 1.0 / hist_dist
            hist_weights = hist_weights / (np.sum(hist_weights) + 1e-10)
            # Direction toward historical best, weighted by convergence level
            pop_center = np.mean(population, axis=0)
            convergence = np.linalg.norm(population - pop_center, axis=1) / (np.linalg.norm(hist_best - pop_center) + 1e-10)
            convergence = np.clip(convergence, 0, 1)
            # Stronger pull when more converged (population clustered away from best)
            hist_strength = 0.3 * (1.0 - convergence)
            hist_direction = (hist_best - population)
            base = base + hist_strength[:, np.newaxis] * hist_direction

        # Convergence-aware perturbation (NEW: escapes local optima)
        pop_center = np.mean(population, axis=0)
        pop_dispersion = np.linalg.norm(population - pop_center, axis=1)
        avg_dispersion = np.mean(pop_dispersion) + 1e-10
        normalized_dispersion = pop_dispersion / avg_dispersion

        # When dispersion is low (population clustered), add stronger perturbation
        low_disp_mask = normalized_dispersion < 0.5
        if np.any(low_disp_mask):
            perturbation_scale = 0.5 * self.F * (1.0 - normalized_dispersion[low_disp_mask])
            perturbation = np.random.randn(np.sum(low_disp_mask), self.dim) * perturbation_scale[:, np.newaxis]
            base[low_disp_mask] = base[low_disp_mask] + perturbation

        return np.clip(base, self.lower, self.upper)

    def _mutate_variant05(self, population, fitness):
        """Opposition-Guided DE with Adaptive Reflection and Quasi-Opposition"""
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

        # Standard base mutation
        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Opposition-based learning: reflect solutions across search space center
        # Adaptive reflection probability (high at start, decays over generations)
        decay_rate = 0.01
        reflect_prob = max(0.05, 0.8 * np.exp(-decay_rate * self.generation))

        # Search space center
        center = (self.lower + self.upper) / 2.0

        # Generate opposite candidates (full reflection)
        opposite_full = 2.0 * center - base

        # Generate quasi-opposition candidates (between center and opposite)
        quasi_opposite = np.random.uniform(center, opposite_full)

        # Generate quasi-reflected (between center and base)
        quasi_reflected = np.random.uniform(center, base)

        # Dimension-wise random walk for additional exploration
        dim_perturb = np.random.randn(self.NP, self.dim) * (0.1 + 0.3 * np.exp(-0.005 * self.generation))

        # Decide which candidates to use per individual
        rand_mask = np.random.rand(self.NP, self.dim)

        # Apply different strategies to different dimensions for diversity
        donors = base.copy()
        reflect_mask = np.random.rand(self.NP) < reflect_prob
        n_reflect = int(np.sum(reflect_mask))

        if n_reflect > 0:
            reflect_indices = np.where(reflect_mask)[0]
            # Half use full opposition, half use quasi-opposition
            half = n_reflect // 2
            full_opp_indices = reflect_indices[:half]
            quasi_opp_indices = reflect_indices[half:]

            donors[full_opp_indices] = opposite_full[full_opp_indices]
            donors[quasi_opp_indices] = quasi_opposite[quasi_opp_indices]

        # Apply dimension-wise perturbation to non-reflected individuals
        non_reflect_indices = np.where(~reflect_mask)[0]
        if len(non_reflect_indices) > 0:
            donors[non_reflect_indices] = donors[non_reflect_indices] + dim_perturb[non_reflect_indices]
            # Mix with quasi-reflected for some individuals
            mix_mask = np.random.rand(len(non_reflect_indices)) < 0.3
            mix_indices = non_reflect_indices[mix_mask]
            if len(mix_indices) > 0:
                donors[mix_indices] = 0.7 * donors[mix_indices] + 0.3 * quasi_reflected[mix_indices]

        # Blend with cultural memory if available
        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * donors + 0.15 * (population + cultural_offset)

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant06(self, population, fitness):
        """Opposition-Guided Escape: basin-hopping via dynamic opposition vectors"""
        # Initialize archive if needed
        if self.archive is None:
            self.archive = np.zeros((0, self.dim))

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

        # Compute centroid for opposition reference
        centroid = np.mean(population, axis=0)

        # Adaptive F: larger when stagnant to escape basins
        if self.stagnation_count > 10:
            F_escape = min(self.F * (1.0 + 0.5 * np.log1p(self.stagnation_count)), 2.5)
        else:
            F_escape = self.F

        # Standard mutation direction (exploitation)
        mutation_dir = p_best - population + F_escape * (population[r1] - population[r2])

        # Opposition direction: reflect around centroid (exploration)
        # Opposition of x is x_opp = centroid + (centroid - x) = 2*centroid - x
        opposition_vector = 2.0 * centroid - population
        opposition_dir = opposition_vector - population + F_escape * (population[r2] - population[r1])

        # Adaptive selection: more opposition when stagnant
        # Start conservative, increase opposition probability with stagnation
        base_opp_prob = 0.1
        stagnation_boost = min(0.6, 0.05 * self.stagnation_count)
        opp_prob = np.clip(base_opp_prob + stagnation_boost, 0.0, 0.7)

        use_opposition = np.random.rand(self.NP) < opp_prob

        # Compute donors for both directions
        donor_exploit = population + mutation_dir
        donor_explore = population + opposition_dir

        # Select based on adaptive probability
        donors = np.where(use_opposition[:, np.newaxis], donor_explore, donor_exploit)

        # Small perturbation when very stagnant (adds randomization for escape)
        if self.stagnation_count > 20:
            perturbation_scale = min(0.5 * np.log1p(self.stagnation_count - 20), 5.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            donors = donors + random_perturbation

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
        """Restart-based mutation: aggressive escape from local optima via random restarts"""
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

    def _mutate_variant08(self, population, fitness):
        """Reflective Diversification: escape local optima by bouncing away from population centroid"""
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

        # Compute population centroid and direction vectors from centroid
        centroid = np.mean(population, axis=0)
        diff_from_centroid = population - centroid
        dist_from_centroid = np.linalg.norm(diff_from_centroid, axis=1, keepdims=True) + 1e-10
        direction_from_centroid = diff_from_centroid / dist_from_centroid

        # Reflected direction (opposite to cluster direction)
        reflected_direction = -direction_from_centroid

        # Compute reflection strength based on stagnation and convergence
        convergence_ratio = np.std(population, axis=0) / (self.upper - self.lower + 1e-10)
        avg_convergence = np.mean(convergence_ratio)

        if self.stagnation_count > 10 or avg_convergence < 0.05:
            # Strong reflection when stuck or highly converged
            reflection_strength = min(0.5, 0.25 + 0.025 * self.stagnation_count)
        elif self.stagnation_count > 3:
            # Moderate reflection when starting to stagnate
            reflection_strength = 0.15 + 0.02 * self.stagnation_count
        else:
            reflection_strength = 0.0

        if reflection_strength > 0:
            # Scale jump based on search space and distance from centroid
            jump_scale = (self.upper - self.lower) * 0.15
            reflected_points = centroid + reflected_direction * jump_scale
            donors = (1 - reflection_strength) * base + reflection_strength * reflected_points
        else:
            donors = base

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant10(self, population, fitness):
        """Restart-Driven Opposition-Based Mutation: escape local optima via reflection"""
        bounds_center = (self.lower + self.upper) / 2.0

        # Detect deep stagnation: stuck far from target
        is_stuck = self.stagnation_count > 15

        if is_stuck:
            # Opposition-based exploration: reflect population across bounds center
            # This jumps to the opposite side of the search space
            opposition_pop = 2.0 * bounds_center - population
            opposition_pop = np.clip(opposition_pop, self.lower, self.upper)

            # Mutate from opposition points instead of current population
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

            # Mutation from OPPOSITION population with amplified F
            F_oppose = min(self.F * 1.5, 2.0)
            opp_base = opposition_pop + F_oppose * (p_best - opposition_pop) + F_oppose * (population[r1] - population[r2])

            # Blend: 60% opposition-mutated, 40% standard mutation from current
            F_std = self.F
            std_base = population + F_std * (p_best - population) + F_std * (population[r1] - population[r2])

            donors = 0.6 * opp_base + 0.4 * std_base
            return np.clip(donors, self.lower, self.upper)

        # Standard mutation (same as original for non-stuck state)
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
        r1, r2