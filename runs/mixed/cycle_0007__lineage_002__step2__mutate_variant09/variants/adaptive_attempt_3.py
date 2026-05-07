import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for operator selection.
    Automatically learns which mutation strategy works best during optimization.
    Includes 6 top-performing variant strategies identified from benchmark results.
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
        # 6 operators: original, variant_02, variant_04, variant_08, variant_09, variant_10
        self.num_operators = 6
        self.operator_names = ['original', 'variant_02', 'variant_04', 'variant_08', 'variant_09', 'variant_10']
        
        # Beta distribution parameters for Thompson Sampling
        self.operator_alpha = np.ones(self.num_operators)
        self.operator_beta = np.ones(self.num_operators)
        
        # Sliding window reward tracking
        self.reward_window_size = 30
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        self.operator_successes = [0] * self.num_operators
        self.operator_attempts = [0] * self.num_operators
        
        self.current_operator_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []
        
        # Credit assignment parameters
        self.credit_decay = 0.95
        self.pending_credits = [0.0] * self.num_operators

        # Archive for variant_01 (JADE-style)
        self.archive = None
        self._pending_archive_additions = None

        # Performance tracking for adaptive weighting
        self.operator_generation_scores = [0.0] * self.num_operators
        self.generation_score_window = 50

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on improvement quality"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            reward = -0.01
        else:
            # Normalize by population size and number of improvements
            reward = total_improvement / max(float(n_improved), 1.0)

        return float(np.clip(reward, -1.0, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window and credit assignment"""
        # Cast reward to float
        reward = float(reward)
        
        rewards = self.operator_rewards[operator_idx]
        rewards.append(reward)
        
        # Track attempts and successes
        self.operator_attempts[operator_idx] += 1
        if reward > 0:
            self.operator_successes[operator_idx] += 1

        # Maintain sliding window
        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        # Apply pending credits with decay
        pending = self.pending_credits[operator_idx] * self.credit_decay
        self.pending_credits[operator_idx] = pending + reward

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

        # Update generation score
        self.operator_generation_scores[operator_idx] += reward

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

            # Periodic operator performance reset to prevent lock-in
            if self.generation % 100 == 0:
                self._decay_operator_scores()

        return self.best_fitness, self.best_solution

    def _decay_operator_scores(self):
        """Decay operator scores to allow adaptation to changing problem landscape"""
        for i in range(self.num_operators):
            self.operator_generation_scores[i] *= 0.8
            # Reset alpha/beta slightly to allow re-exploration
            self.operator_alpha[i] = 0.5 * self.operator_alpha[i] + 0.5
            self.operator_beta[i] = 0.5 * self.operator_beta[i] + 0.5

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
            return self._mutate_variant02(population, fitness)
        elif self.current_operator_idx == 2:
            return self._mutate_variant04(population, fitness)
        elif self.current_operator_idx == 3:
            return self._mutate_variant08(population, fitness)
        elif self.current_operator_idx == 4:
            return self._mutate_variant09(population, fitness)
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
        """Variant 02: Mean-Reversion Opposition DE — escape local optima via centroid pull and opposition"""
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

        # Compute population centroid (geometric center of search)
        centroid = np.mean(population, axis=0)

        # Compute search-space center and opposition vector
        search_center = 0.5 * (self.lower + self.upper)
        opposition_vector = 2.0 * search_center - population

        # Adaptive stagnation detection
        is_stagnant = self.stagnation_count > 10
        stagnation_factor = min(self.stagnation_count / 50.0, 1.0) if is_stagnant else 0.0

        # Mean-reversion weight: increases when stagnant to pull away from converged cluster
        w_mean = 0.1 + 0.4 * stagnation_factor

        # Opposition weight: only significant when stagnant (escape to opposite side)
        w_opp = 0.3 * stagnation_factor

        # Standard DE components (always present for exploitation)
        w_best = 0.5
        w_diff = 0.5

        # Build composite donor
        base_direction = w_best * (p_best - population) + w_diff * (population[r1] - population[r2])
        mean_direction = w_mean * (centroid - population)
        opp_direction = w_opp * (opposition_vector - population)

        # Combine with adaptive F
        F_base = self.F
        F_escape = np.clip(F_base * (1.0 + stagnation_factor * 0.5), 0.5, 2.0)

        donors = population + F_escape * (base_direction + mean_direction + opp_direction)

        # Clamp to bounds and return
        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant04(self, population, fitness):
        """Variant 04: Multi-Donor Composite Mutation with Diversity-Weighted Selection"""
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
            w_best = w_best / total_w
            w_centroid = w_centroid / total_w
            w_rand = w_rand / total_w

        # Composite donors
        donors = (w_best[:, np.newaxis] * donor_best +
                  w_centroid[:, np.newaxis] * donor_centroid +
                  w_rand[:, np.newaxis] * donor_rand)

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant08(self, population, fitness):
        """Variant 08: Centroid Repulsion — escape local optima by mutating away from population center"""
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

        # Compute centroid (center of mass of current population)
        centroid = np.mean(population, axis=0)

        # Compute search space diameter for scale normalization
        space_diameter = np.linalg.norm(self.upper - self.lower) + 1e-10

        # Distance from centroid to best individual (normalized)
        if self.best_solution is not None:
            centroid_to_best = self.best_solution - centroid
        else:
            centroid_to_best = p_best - centroid
        dist_centroid_best = np.linalg.norm(centroid_to_best) + 1e-10

        # Anti-centroid: reflect best through centroid to explore opposite side
        if self.best_solution is not None:
            anti_centroid = 2.0 * centroid - self.best_solution
        else:
            anti_centroid = 2.0 * centroid - p_best

        # Stagnation-driven scale: larger when stuck, to escape basin of attraction
        stagnation = float(self.stagnation_count)
        repulsion_scale = 1.5 + min(stagnation * 0.05, 2.0)  # range [1.5, 3.5]
        repulsion_scale = min(repulsion_scale, 3.5)

        # Compute repulsion vector: move FROM centroid toward anti-centroid
        repulsion_direction = anti_centroid - centroid
        repulsion_norm = np.linalg.norm(repulsion_direction) + 1e-10
        unit_repulsion = repulsion_direction / repulsion_norm

        # Target = centroid + scaled repulsion vector
        target = centroid + unit_repulsion * dist_centroid_best * repulsion_scale

        # Clip target to bounds (edge case: handle out-of-bounds reflection)
        target = np.clip(target, self.lower, self.upper)

        # Large perturbation scale for exploration
        F_large = np.clip(0.8 + min(stagnation * 0.03, 1.0), 0.8, 2.0)
        F_secondary = np.clip(self.F + 0.2, 0.5, 2.0)

        # Primary mutation: move toward anti-centroid (away from best)
        donors = population + F_large * (target - population) + F_secondary * (population[r1] - population[r2])

        # Additional diversity injection when stagnant
        if stagnation > 10:
            diversity_noise = np.random.randn(self.NP, self.dim)
            diversity_scale = min(0.2 * np.log1p(stagnation), 2.0)
            donors = donors + diversity_scale * diversity_noise

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant09(self, population, fitness):
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

        # Standard DE mutation
        base_mutants = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        # Opposition learning: reflect around centroid
        centroid = np.mean(population, axis=0)
        lower_arr = np.array(self.lower)
        upper_arr = np.array(self.upper)
        range_arr = upper_arr - lower_arr

        # Opposition candidates (mirrored across search space)
        opposition = lower_arr + upper_arr - base_mutants

        # Blend DE mutant with opposition candidate
        alpha = np.random.rand(self.NP, 1)
        donors = alpha * base_mutants + (1 - alpha) * opposition

        # Inject random restart when severely stagnant
        if self.stagnation_count > 20:
            n_inject = max(1, self.NP // 4)
            inject_idx = np.random.randint(0, self.NP, size=n_inject)
            inject = np.random.uniform(self.lower, self.upper, (n_inject, self.dim))
            donors[inject_idx] = inject

        # Dimension-wise perturbation for escaping local traps
        if self.stagnation_count > 10:
            perturb_mask = np.random.rand(self.NP, self.dim) < 0.3
            perturb = np.random.randn(self.NP, self.dim) * 0.5 * (self.stagnation_count / 50.0)
            donors = donors + perturb_mask * perturb

        return np.clip(donors, self.lower, self.upper)

    def _mutate_variant10(self, population, fitness):
        """Variant 10: Opposition-guided mutation with restart and niching"""
        pop_dim = population.shape

        # Check if we need a restart (severe stagnation or trapped)
        needs_restart = (self.stagnation_count > 40 or 
                         np.min(fitness) > 1e2)  # Stuck in bad region

        if needs_restart and self.best_solution is not None:
            # Generate opposition population
            opp_pop = self.lower + self.upper - population

            # Blend best solution with random direction to escape basin
            escape_dir = np.random.randn(self.NP, self.dim)
            escape_dir_norm = np.linalg.norm(escape_dir, axis=1, keepdims=True) + 1e-10
            escape_dir = escape_dir / escape_dir_norm
            escape_scale = np.random.uniform(10, 50, size=(self.NP, 1))
            escape_pop = np.clip(self.best_solution + escape_dir * escape_scale, self.lower, self.upper)

            # Mix: keep some current, some opposition, some escape
            selector = np.random.randint(0, 3, size=self.NP)
            restart_pop = np.where(selector[:, None] == 0, population,
                                  np.where(selector[:, None] == 1, opp_pop, escape_pop))

            # Fitness-based niche selection: keep diverse solutions
            sorted_idx_local = np.argsort(fitness)
            n_keep = max(5, self.NP // 4)
            keep_indices = sorted_idx_local[:n_keep]

            # Fill rest with opposition/escape and random
            fill_pop = np.vstack([
                restart_pop[keep_indices],
                self.lower + np.random.rand(self.NP - n_keep, self.dim) * (self.upper - self.lower)
            ])
            population = np.clip(fill_pop, self.lower, self.upper)
            fitness = np.full(self.NP, np.inf)
            self.stagnation_count = 0

        # Main mutation: current-to-niche-best with opposition learning
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        # Find fitness niche: select from similar-fitness region
        best_fitness_val = float(fitness[sorted_idx[0]])
        niche_mask = fitness < (best_fitness_val * 1.5 + 1.0)
        niche_indices = np.where(niche_mask)[0]
        if len(niche_indices) < 2:
            niche_indices = best_indices

        # p_best from niche
        p_best_idx = niche_indices[np.random.randint(0, len(niche_indices), size=self.NP)]
        p_best = population[p_best_idx]

        # Random indices with opposition bias
        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        # Opposition vectors: generate opposite of random individuals
        opp_r1 = self.lower + self.upper - population[r1]
        opp_r2 = self.lower + self.upper - population[r2]

        # Adaptive F with higher values for exploration
        if self.stagnation_count > 10:
            F_adapt = np.clip(self.F * (1.0 + 0.3 * np.log1p(self.stagnation_count)), 0.5, 2.5)
        else:
            F_adapt = np.clip(self.F, 0.5, 1.5)

        # Mutation with opposition and niche guidance
        base_mutation = p_best - population
        diff_opp = opp_r1 - opp_r2
        diff_current = population[r1] - population[r2]

        # Blend: 60% niche-guided, 40% opposition-based difference
        donors = population + F_adapt * base_mutation + 0.5 * F_adapt * (0.6 * diff_current + 0.4 * diff_opp)

        # Add small opposition perturbation to some individuals
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
            new_F = momentum * float(np.mean(self._adapt_history_F[-5:])) + (1 - momentum) * new_F
            new_CR = momentum * float(np.mean(self._adapt_history_CR[-5:])) + (1 - momentum) * new_CR

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
