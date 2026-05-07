import numpy as np


class ReflectiveDEWithStrategyPortfolio:
    """
    Differential Evolution using reflective boundary handling and an adaptive
    portfolio of mutation strategies. Each individual maintains its own
    strategy label and per-strategy success memories that drive strategy
    selection and parameter adaptation.
    
    This version uses adaptive operator selection (sliding window + Thompson Sampling)
    to choose among multiple crossover strategies during optimization.
    """

    def __init__(self, dim, np_factor=6, seed=None):
        self.dim = dim
        self.NP = np_factor * dim
        self.NP = max(20, min(self.NP, 300))
        self.lower = -100.0
        self.upper = 100.0
        self.rng = np.random.default_rng(seed)

        # Strategy definitions (name, n_differences, uses_best)
        self.strategy_names = [
            "rand_1",
            "best_1",
            "current_1",
            "rand_2",
            "best_2",
        ]
        self.n_strategies = len(self.strategy_names)

        # Per-individual state
        self.F = np.empty(self.NP)
        self.CR = np.empty(self.NP)
        self.strategy_idx = np.zeros(self.NP, dtype=int)

        # Per-strategy success memory for adaptation
        self.strategy_success_fitness = np.zeros(self.n_strategies)
        self.strategy_success_count = np.zeros(self.n_strategies)
        self.strategy_attempt_count = np.zeros(self.n_strategies)

        # Global memory for F/CR adaptation (jADE-style)
        self.archive = []
        self.archive_max_size = self.NP

        # Stagnation detection
        self.generation_without_improvement = 0
        self.last_best_fitness = np.inf

        # Diversity tracking
        self.diversity_history = []

        # --- Adaptive crossover operator selection ---
        # We have 6 crossover operators indexed 0..5:
        # 0: original (binomial)
        # 1: variant_03 (success-rate guided crossover with adaptive mixing)
        # 2: variant_04 (success-guided adaptive crossover)
        # 3: variant_06 (confidence-weighted parent preservation)
        # 4: variant_08 (ultra-conservative crossover)
        # 5: variant_10 (coordinate-wise elite-preserving conservative)
        self.n_crossover_ops = 6
        # Thompson Sampling parameters (Beta distribution)
        self.cx_alpha = np.ones(self.n_crossover_ops)
        self.cx_beta = np.ones(self.n_crossover_ops)
        # Sliding window for credit assignment
        self.cx_window_size = 60
        self.cx_history = []  # list of (op_idx, reward)
        # Current operator index
        self.current_cx_op = 0
        # Per-operator tracking
        self.dim_success_rate = None
        self.dim_attempt_count = None
        # Track improvement rate
        self.last_improvement_rate = 0.5
        
        # For variant_03
        self.dim_cx_success = None
        self.dim_cx_total = None
        
        # For variant_04
        self.dim_improve_count = None
        self.dim_attempt_count_v04 = None
        self.prev_trial_pop = None
        
        # For variant_06
        self._dim_success_count = None
        self._dim_attempt_count_v06 = None
        self._dim_history_fitness = None
        
        # For variant_10
        self.cx_elite_memory = None
        
        # Generation counter
        self.gen_count = 0

    def _initialize_population(self):
        lo = self.lower
        hi = self.upper
        pop = lo + (hi - lo) * self.rng.uniform(size=(self.NP, self.dim))
        self.F[:] = 0.5
        self.CR[:] = 0.5
        self.strategy_idx[:] = self.rng.integers(0, self.n_strategies, size=self.NP)
        self.strategy_success_fitness[:] = 0.0
        self.strategy_success_count[:] = 1
        self.strategy_attempt_count[:] = 1
        self.archive = []
        # Reset crossover adaptation
        self.cx_alpha = np.ones(self.n_crossover_ops)
        self.cx_beta = np.ones(self.n_crossover_ops)
        self.cx_history = []
        self.dim_success_rate = np.zeros(self.dim)
        self.dim_attempt_count = np.zeros(self.dim)
        
        # Reset variant-specific state
        self.dim_cx_success = np.zeros(self.dim)
        self.dim_cx_total = np.zeros(self.dim) + 1e-10
        
        self.dim_improve_count = np.zeros(self.dim)
        self.dim_attempt_count_v04 = np.zeros(self.dim)
        self.prev_trial_pop = None
        
        self._dim_success_count = np.zeros(self.dim)
        self._dim_attempt_count_v06 = np.zeros(self.dim)
        self._dim_history_fitness = np.full(self.dim, np.inf)
        
        self.cx_elite_memory = None
        self.gen_count = 0
        
        return pop

    def _evaluate_batch(self, population, func):
        fitness = func(population)
        if len(fitness) < len(population):
            return fitness[:len(fitness)]
        return fitness

    def _clip_reflective(self, population):
        pop = population.copy()
        mask_lo = pop < self.lower
        mask_hi = pop > self.upper
        pop[mask_lo] = 2.0 * self.lower - pop[mask_lo]
        pop[mask_hi] = 2.0 * self.upper - pop[mask_hi]
        pop = np.clip(pop, self.lower, self.upper)
        return pop

    def _select_three_distinct(self, exclude_idx):
        pool = np.arange(self.NP)
        pool = pool[pool != exclude_idx]
        selected = self.rng.choice(pool, size=3, replace=False)
        return selected[0], selected[1], selected[2]

    def _select_five_distinct(self, exclude_idx):
        pool = np.arange(self.NP)
        pool = pool[pool != exclude_idx]
        selected = self.rng.choice(pool, size=5, replace=False)
        return selected[0], selected[1], selected[2], selected[3], selected[4]

    def _mutate_single_reflective(self, idx, population, fitness, trial_pop, trial_fit):
        x_i = population[idx]
        s_idx = self.strategy_idx[idx]
        strategy = self.strategy_names[s_idx]

        a, b, c = self._select_three_distinct(idx)
        x_a, x_b, x_c = population[a], population[b], population[c]
        f_scale = self.F[idx]

        if strategy == "rand_1":
            mutant = x_a + f_scale * (x_b - x_c)
        elif strategy == "best_1":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            mutant = x_best + f_scale * (x_a - x_b)
        elif strategy == "current_1":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            mutant = x_i + f_scale * (x_best - x_i) + f_scale * (x_a - x_b)
        elif strategy == "rand_2":
            d, e = self._select_three_distinct(idx)[:2]
            x_d, x_e = population[d], population[e]
            mutant = x_a + f_scale * (x_b - x_c) + f_scale * (x_d - x_e)
        elif strategy == "best_2":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            d, e = self._select_three_distinct(idx)[:2]
            x_d, x_e = population[d], population[e]
            mutant = x_best + f_scale * (x_a - x_b) + f_scale * (x_d - x_e)
        else:
            mutant = x_a + f_scale * (x_b - x_c)

        return self._clip_reflective(mutant.reshape(1, -1))[0]

    def _mutate_batch(self, population, fitness):
        n_trials = self.NP
        trial_population = np.empty((n_trials, self.dim))

        for idx in range(n_trials):
            trial_population[idx] = self._mutate_single_reflective(
                idx, population, fitness, trial_population, None
            )
        return trial_population

    # --- Crossover operator implementations ---

    def _crossover_original(self, population, trial_population):
        """Op 0: Standard binomial crossover."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_variant03(self, population, trial_population):
        """Op 1: Success-rate guided crossover with adaptive mixing (variant_03)."""
        n_trials = self.NP
        dim = self.dim

        if self.dim_cx_success is None or len(self.dim_cx_success) != dim:
            self.dim_cx_success = np.zeros(dim)
            self.dim_cx_total = np.zeros(dim) + 1e-10

        success_rates = self.dim_cx_success / (self.dim_cx_total + 1e-10)

        pop_std = np.std(population[:n_trials], axis=0)
        max_spread = self.upper - self.lower
        diversity = np.mean(pop_std / (max_spread + 1e-10))

        mix_base = np.clip(0.3 + 0.2 * diversity, 0.15, 0.5)

        fitness = self.population_fitness[:n_trials] if hasattr(self, 'population_fitness') else np.zeros(n_trials)
        valid_fitness = fitness[np.isfinite(fitness)]
        best_fit = np.min(valid_fitness) if len(valid_fitness) > 0 else 0.0
        worst_fit = np.max(valid_fitness) if len(valid_fitness) > 0 else 1.0
        fit_range = worst_fit - best_fit + 1e-10

        median_fit = np.median(valid_fitness) if len(valid_fitness) > 0 else best_fit
        mix_factor = np.clip(
            mix_base + 0.2 * np.maximum(0, (median_fit - fitness) / fit_range),
            0.1, 0.6
        )

        mix_expanded = mix_factor[:, np.newaxis]
        blended = (
            mix_expanded * trial_population +
            (1.0 - mix_expanded) * population[:n_trials]
        )

        inherit_prob = np.clip(success_rates, 0.2, 0.8)

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < inherit_prob[np.newaxis, :]

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        offspring = np.where(cross_mask, blended, population[:n_trials])

        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_improved = np.mean(trial_better, axis=0)
        self.dim_cx_total += 1
        alpha = 0.1
        self.dim_cx_success = alpha * dim_improved * n_trials + (1 - alpha) * self.dim_cx_success

        return offspring

    def _crossover_variant04(self, population, trial_population):
        """Op 2: Success-guided adaptive crossover (variant_04)."""
        n_trials = self.NP
        dim = self.dim

        if self.dim_improve_count is None or len(self.dim_improve_count) != dim:
            self.dim_improve_count = np.zeros(dim)
            self.dim_attempt_count_v04 = np.zeros(dim)
            self.prev_trial_pop = np.zeros((n_trials, dim))

        if self.prev_trial_pop is not None and self.prev_trial_pop.shape[0] == n_trials and self.prev_trial_pop.shape[1] == dim:
            improved_mask = trial_population < self.prev_trial_pop
            self.dim_improve_count += np.sum(improved_mask, axis=0)
            self.dim_attempt_count_v04 += n_trials

        self.prev_trial_pop = trial_population.copy()

        dim_success_rate = np.zeros(dim)
        valid_dims = self.dim_attempt_count_v04 > 10
        dim_success_rate[valid_dims] = (
            self.dim_improve_count[valid_dims] / self.dim_attempt_count_v04[valid_dims]
        )

        pop_std = np.std(population, axis=0)
        max_spread = self.upper - self.lower
        diversity_factor = np.clip(np.mean(pop_std) / (max_spread + 1e-10), 0.05, 0.95)

        base_cr = 0.3 + 0.4 * diversity_factor

        dim_weights = np.clip(dim_success_rate, 0.1, 0.9)
        dim_weights_sum = np.sum(dim_weights) + 1e-10
        dim_weights_norm = dim_weights / dim_weights_sum

        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for i in range(n_trials):
            n_cross = max(1, int(np.clip(
                base_cr * dim * (0.5 + 0.5 * self.rng.random()),
                1, dim
            )))

            selected_dims = set()
            attempts = 0
            while len(selected_dims) < n_cross and attempts < n_cross * 3:
                d = self.rng.choice(dim, p=dim_weights_norm)
                selected_dims.add(d)
                attempts += 1
            selected_dims = list(selected_dims)

            if len(selected_dims) == 0:
                selected_dims = [self.rng.integers(0, dim)]

            cross_mask[i, selected_dims] = True

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        blend_weights = 0.7 + 0.3 * dim_weights
        blended_trial = (
            blend_weights[np.newaxis, :] * trial_population +
            (1 - blend_weights[np.newaxis, :]) * population[:n_trials]
        )

        offspring = np.where(cross_mask, blended_trial, population[:n_trials])
        return offspring

    def _crossover_variant06(self, population, trial_population):
        """Op 3: Confidence-weighted parent preservation crossover (variant_06)."""
        n_trials = self.NP
        dim = self.dim

        if self._dim_success_count is None or len(self._dim_success_count) != dim:
            self._dim_success_count = np.zeros(dim)
            self._dim_attempt_count_v06 = np.zeros(dim)
            self._dim_history_fitness = np.full(dim, np.inf)

        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_signal = np.mean(trial_better, axis=0)

        self._dim_attempt_count_v06 += 1
        alpha = 0.1
        self._dim_success_count = (
            alpha * dim_signal * float(n_trials) +
            (1 - alpha) * self._dim_success_count
        )

        success_rate = np.zeros(dim)
        valid = self._dim_attempt_count_v06 > 5
        success_rate[valid] = self._dim_success_count[valid] / self._dim_attempt_count_v06[valid]

        conf_preserve = np.clip(success_rate, 0.05, 0.95)

        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for i in range(n_trials):
            preserve_prob = conf_preserve + 0.2 * (self.CR[i] - 0.5)
            preserve_prob = np.clip(preserve_prob, 0.0, 1.0)

            rand_vals = self.rng.uniform(size=dim)
            cross_mask[i] = rand_vals >= preserve_prob

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_variant08(self, population, trial_population):
        """Op 4: Ultra-conservative crossover (variant_08)."""
        n_trials = self.NP
        dim = self.dim

        cr_batch = np.clip(self.CR.copy() * 0.15, 0.02, 0.08)

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        blend_weight = 0.1
        blended = (
            (1.0 - blend_weight) * population[:n_trials] +
            blend_weight * trial_population
        )

        offspring = np.where(cross_mask, blended, population[:n_trials])
        offspring = np.clip(offspring, self.lower, self.upper)
        return offspring

    def _crossover_variant10(self, population, trial_population):
        """Op 5: Coordinate-wise elite-preserving conservative crossover (variant_10)."""
        n_trials = self.NP
        dim = self.dim

        if self.cx_elite_memory is None or self.cx_elite_memory.shape != (n_trials, dim):
            self.cx_elite_memory = population[:n_trials].copy()

        pop_centroid = np.mean(population, axis=0)
        avg_distance = np.mean(np.linalg.norm(population - pop_centroid, axis=1))
        max_spread = self.upper - self.lower
        diversity_ratio = avg_distance / (max_spread + 1e-10)

        cr_base = np.clip(0.15 + 0.25 * diversity_ratio, 0.1, 0.4)
        cr_batch = np.clip(cr_base + self.rng.uniform(-0.05, 0.05, size=n_trials), 0.05, 0.5)

        parent_trial_dist = np.linalg.norm(
            trial_population - population[:n_trials], axis=1, keepdims=True
        )
        dist_threshold = 0.3 * dim * (self.upper - self.lower) / (n_trials ** 0.5 + 1e-10)

        jump_risk = np.clip(parent_trial_dist / (dist_threshold + 1e-10), 0.0, 1.0)

        effective_cr = cr_batch[:, np.newaxis] * (1.0 - 0.5 * jump_risk)

        better_mask = trial_population < self.cx_elite_memory
        self.cx_elite_memory = np.where(better_mask, trial_population, self.cx_elite_memory)

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < effective_cr

        coord_dist_from_elite = np.abs(population[:n_trials] - self.cx_elite_memory)
        max_coord_dist = (self.upper - self.lower) * 0.1
        stable_mask = coord_dist_from_elite < max_coord_dist
        cross_mask = cross_mask & ~stable_mask

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        alpha = 0.2
        blended_trial = (
            alpha * trial_population +
            (1.0 - alpha) * population[:n_trials]
        )

        offspring = np.where(cross_mask, blended_trial, population[:n_trials])
        offspring = np.clip(offspring, self.lower, self.upper)
        return offspring

    def _select_crossover_operator(self):
        """Thompson Sampling to select crossover operator."""
        samples = np.array([
            self.rng.beta(max(self.cx_alpha[i], 1e-6), max(self.cx_beta[i], 1e-6))
            for i in range(self.n_crossover_ops)
        ])
        return int(np.argmax(samples))

    def _update_crossover_reward(self, op_idx, n_improved, n_total, fitness_improvement):
        """Update Thompson Sampling parameters based on observed reward."""
        op_idx = int(op_idx)
        if n_total > 0:
            success_rate = float(n_improved) / float(n_total)
        else:
            success_rate = 0.0
        
        fit_reward = float(np.clip(fitness_improvement, 0.0, 1.0))
        
        reward = 0.6 * success_rate + 0.4 * fit_reward
        reward = float(np.clip(reward, 0.0, 1.0))
        
        self.cx_history.append((op_idx, reward))
        if len(self.cx_history) > self.cx_window_size:
            self.cx_history.pop(0)
        
        # Recompute alpha/beta from sliding window
        self.cx_alpha = np.ones(self.n_crossover_ops)
        self.cx_beta = np.ones(self.n_crossover_ops)
        for (oidx, rew) in self.cx_history:
            oidx = int(oidx)
            if 0 <= oidx < self.n_crossover_ops:
                self.cx_alpha[oidx] += float(rew)
                self.cx_beta[oidx] += float(1.0 - rew)

    def _crossover_batch(self, population, trial_population):
        """Adaptive crossover: selects among 6 operators using Thompson Sampling."""
        self.current_cx_op = self._select_crossover_operator()
        
        try:
            if self.current_cx_op == 0:
                result = self._crossover_original(population, trial_population)
            elif self.current_cx_op == 1:
                result = self._crossover_variant03(population, trial_population)
            elif self.current_cx_op == 2:
                result = self._crossover_variant04(population, trial_population)
            elif self.current_cx_op == 3:
                result = self._crossover_variant06(population, trial_population)
            elif self.current_cx_op == 4:
                result = self._crossover_variant08(population, trial_population)
            elif self.current_cx_op == 5:
                result = self._crossover_variant10(population, trial_population)
            else:
                result = self._crossover_original(population, trial_population)
        except Exception:
            self.current_cx_op = 0
            result = self._crossover_original(population, trial_population)
        
        result = np.nan_to_num(result, nan=0.0, posinf=self.upper, neginf=self.lower)
        result = np.clip(result, self.lower, self.upper)
        return result

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved = trial_fitness < fitness[:len(trial_fitness)]
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[:len(trials)][improved] = trials[improved]
        new_fitness[:len(trials)][improved] = trial_fitness[improved]
        return new_population, new_fitness, improved

    def _update_strategy_success(self, improved_mask):
        for s in range(self.n_strategies):
            count_s = np.sum((self.strategy_idx == s) & improved_mask)
            self.strategy_success_count[s] += count_s
            self.strategy_attempt_count[s] += np.sum(self.strategy_idx == s)
            if count_s > 0:
                self.strategy_success_fitness[s] = (
                    0.5 * self.strategy_success_fitness[s] + 0.5 * count_s
                )

    def _adapt_strategy_probabilities(self):
        success_rates = np.zeros(self.n_strategies)
        valid_strategies = self.strategy_attempt_count > 0
        success_rates[valid_strategies] = (
            self.strategy_success_count[valid_strategies] / 
            self.strategy_attempt_count[valid_strategies]
        )

        best_fitness = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.min(self.archive) if self.archive else 0.0

        ref_fitness = getattr(self, 'reference_fitness', 1e-3)
        threshold = ref_fitness * 1.5 if ref_fitness > 0 else 1e-4

        if best_fitness < threshold:
            orig_prob = 1.0 / self.n_strategies
            adaptive_prob = success_rates / (np.sum(success_rates) + 1e-10)
            alpha = 0.85
            probs = alpha * orig_prob + (1 - alpha) * adaptive_prob
        else:
            temperature = 0.3
            exp_rates = np.exp(success_rates / (temperature + 1e-10))
            probs = exp_rates / (np.sum(exp_rates) + 1e-10)

        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (np.sum(probs) + 1e-10)

        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_F_CR_jade_style(self, population, fitness, trials, trial_fitness, improved_mask):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved_trials = trials[improved_mask]
        improved_parents = population[:len(trials)][improved_mask]

        if len(improved_trials) > 0:
            fi_values = np.abs(improved_trials - improved_parents)
            fi_values = fi_values / (np.linalg.norm(fi_values, axis=1, keepdims=True) + 1e-10)
            self.archive.extend(improved_trials.tolist())
            if len(self.archive) > self.archive_max_size:
                remove_n = len(self.archive) - self.archive_max_size
                self.archive = self.archive[remove_n:]

        for i in range(self.NP):
            if improved_mask[i]:
                self.F[i] = self.rng.uniform(0.1, 0.9)
            else:
                self.F[i] = 0.9 * self.F[i] + 0.1 * 0.5

            if improved_mask[i]:
                self.CR[i] = self.rng.beta(1.0, 0.5)
            else:
                self.CR[i] = 0.9 * self.CR[i] + 0.1 * 0.5
            self.CR[i] = np.clip(self.CR[i], 0.0, 1.0)

    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return float(np.mean(distances))

    def _check_stagnation(self, fitness):
        current_best = float(np.min(fitness))
        if current_best < self.last_best_fitness - 1e-10:
            self.generation_without_improvement = 0
            self.last_best_fitness = current_best
            return False
        else:
            self.generation_without_improvement += 1
            return self.generation_without_improvement > 50

    def _reinitialize_stagnant_subpopulation(self, population, fitness):
        n_replace = max(5, self.NP // 10)
        replace_idx = self.rng.choice(self.NP, size=n_replace, replace=False)
        lo, hi = self.lower, self.upper
        new_individuals = lo + (hi - lo) * self.rng.uniform(size=(n_replace, self.dim))
        population[replace_idx] = new_individuals
        fitness[replace_idx] = np.inf
        self.generation_without_improvement = 0
        return population, fitness

    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        population = self._clip_reflective(population)
        fitness = self._evaluate_batch(population, func)

        if len(fitness) == 0:
            return float(np.inf), np.zeros(self.dim)

        self.population_fitness = fitness.copy()

        best_idx = int(np.argmin(fitness))
        f_opt = float(fitness[best_idx])
        x_opt = population[best_idx].copy()

        while not stopping_condition():
            self.gen_count += 1
            trials = self._mutate_batch(population, fitness)
            
            # Store pre-crossover state for reward computation
            pre_best = float(np.min(fitness))
            
            trials = self._crossover_batch(population, trials)
            trials = self._clip_reflective(trials)
            trial_fitness = self._evaluate_batch(trials, func)

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]

            if len(trial_fitness) == 0:
                break

            if stopping_condition():
                for i in range(len(trial_fitness)):
                    if trial_fitness[i] < f_opt:
                        f_opt = float(trial_fitness[i])
                        x_opt = trials[i].copy()
                break

            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            self.population_fitness = fitness.copy()

            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < f_opt:
                f_opt = float(fitness[current_best_idx])
                x_opt = population[current_best_idx].copy()

            # Update crossover operator reward
            n_improved = int(np.sum(improved_mask))
            n_total = len(improved_mask)
            post_best = float(np.min(fitness))
            
            if abs(pre_best) > 1e-30:
                fit_improvement = float(max(0.0, (pre_best - post_best) / (abs(pre_best) + 1e-30)))
            else:
                fit_improvement = float(max(0.0, pre_best - post_best))
            fit_improvement = float(np.clip(fit_improvement, 0.0, 1.0))
            
            self.last_improvement_rate = float(n_improved) / float(max(n_total, 1))
            
            self._update_crossover_reward(
                self.current_cx_op, n_improved, n_total, fit_improvement
            )

            self._update_strategy_success(improved_mask)
            self._adapt_strategy_probabilities()
            self._adapt_F_CR_jade_style(population, fitness, trials, trial_fitness, improved_mask)

            div = self._compute_diversity(population)
            self.diversity_history.append(div)

            if self._check_stagnation(fitness):
                population, fitness = self._reinitialize_stagnant_subpopulation(population, fitness)
                self.population_fitness = fitness.copy()
                self.archive = []
                self.F[:] = 0.5
                self.CR[:] = 0.5

        return f_opt, x_opt
