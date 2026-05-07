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
        # We have 5 crossover operators indexed 0..4:
        # 0: original (binomial)
        # 1: variant_02 (dim-success adaptive CR)
        # 2: variant_03 (diversity-adaptive blending)
        # 3: variant_05 (block crossover) -- dominant winner
        # 4: variant_09 (binomial/exponential hybrid)
        self.n_crossover_ops = 5
        # Thompson Sampling parameters (Beta distribution)
        self.cx_alpha = np.ones(self.n_crossover_ops)  # successes + 1
        self.cx_beta = np.ones(self.n_crossover_ops)   # failures + 1
        # Sliding window for credit assignment
        self.cx_window_size = 50
        self.cx_history = []  # list of (op_idx, reward)
        # Current operator index
        self.current_cx_op = 3  # start with the best overall (variant_05)
        # Per-operator tracking for dim-success (variant_02)
        self.dim_success_rate = None
        self.dim_attempt_count = None
        # Track improvement rate for variant_06-style temp
        self.last_improvement_rate = 0.5

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

    def _crossover_dim_adaptive(self, population, trial_population):
        """Op 1: variant_02 - Per-dimension success adaptive CR."""
        n_trials = self.NP
        dim = self.dim
        if self.dim_success_rate is None or len(self.dim_success_rate) != dim:
            self.dim_success_rate = np.zeros(dim)
            self.dim_attempt_count = np.zeros(dim)

        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_signal = np.mean(trial_better, axis=0)
        self.dim_attempt_count += 1
        alpha = 0.1
        self.dim_success_rate = alpha * dim_signal + (1 - alpha) * self.dim_success_rate

        dim_probs = np.clip(self.dim_success_rate, 0.01, 0.99)
        dim_probs_norm = dim_probs / (np.sum(dim_probs) + 1e-10)
        cr_base = 0.3 + 0.4 * np.mean(dim_probs_norm)
        cr_batch = np.full(n_trials, np.clip(cr_base, 0.0, 1.0))

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_diversity_blend(self, population, trial_population):
        """Op 2: variant_03 - Diversity-adaptive blending crossover."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        max_spread = self.upper - self.lower
        pop_spread = np.std(population, axis=0)
        normalized_diversity = np.mean(pop_spread / (max_spread + 1e-10))

        alpha_base = np.clip(0.5 - normalized_diversity, 0.1, 0.5)
        alpha_arr = np.clip(
            alpha_base + self.rng.uniform(-0.05, 0.05, size=n_trials),
            0.05, 0.6
        )

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True

        blended = (
            alpha_arr[:, np.newaxis] * trial_population +
            (1.0 - alpha_arr[:, np.newaxis]) * population[:n_trials]
        )

        offspring = np.where(cross_mask, blended, population[:n_trials])
        return offspring

    def _crossover_block(self, population, trial_population):
        """Op 3: variant_05 - Block crossover (dominant winner)."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        block_size = max(1, dim // 8)
        n_blocks = (dim + block_size - 1) // block_size

        cross_blocks = self.rng.uniform(size=(n_trials, n_blocks)) < cr_batch[:, np.newaxis]

        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, dim)
            cross_mask[:, start:end] = cross_blocks[:, b:b+1]

        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _crossover_hybrid_binexp(self, population, trial_population):
        """Op 4: variant_09 - Binomial/exponential hybrid crossover."""
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        cross_mask = np.zeros((n_trials, dim), dtype=bool)

        for i in range(n_trials):
            if self.rng.random() < 0.5:
                # Binomial crossover
                mask = self.rng.uniform(size=dim) < cr_batch[i]
                j_rand = self.rng.integers(0, dim)
                mask[j_rand] = True
            else:
                # Exponential crossover
                j_start = self.rng.integers(0, dim)
                mask = np.zeros(dim, dtype=bool)
                L = 0
                while L < dim and self.rng.random() < cr_batch[i]:
                    j = (j_start + L) % dim
                    mask[j] = True
                    L += 1
                if not np.any(mask):
                    mask[j_start] = True
            cross_mask[i] = mask

        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    def _select_crossover_operator(self):
        """Thompson Sampling to select crossover operator."""
        samples = np.array([
            self.rng.beta(self.cx_alpha[i], self.cx_beta[i])
            for i in range(self.n_crossover_ops)
        ])
        return int(np.argmax(samples))

    def _update_crossover_reward(self, op_idx, n_improved, n_total, fitness_improvement):
        """Update Thompson Sampling parameters based on observed reward."""
        op_idx = int(op_idx)
        # Reward is the fraction of improved individuals + bonus for fitness improvement
        if n_total > 0:
            success_rate = float(n_improved) / float(n_total)
        else:
            success_rate = 0.0
        
        # Normalize fitness improvement to [0, 1] range
        fit_reward = float(np.clip(fitness_improvement, 0.0, 1.0))
        
        # Combined reward
        reward = 0.7 * success_rate + 0.3 * fit_reward
        reward = float(np.clip(reward, 0.0, 1.0))
        
        # Store in sliding window
        self.cx_history.append((op_idx, reward))
        if len(self.cx_history) > self.cx_window_size:
            self.cx_history.pop(0)
        
        # Recompute alpha/beta from sliding window
        self.cx_alpha = np.ones(self.n_crossover_ops)
        self.cx_beta = np.ones(self.n_crossover_ops)
        for (oidx, rew) in self.cx_history:
            oidx = int(oidx)
            self.cx_alpha[oidx] += float(rew)
            self.cx_beta[oidx] += float(1.0 - rew)

    def _crossover_batch(self, population, trial_population):
        """Adaptive crossover: selects among 5 operators using Thompson Sampling."""
        self.current_cx_op = self._select_crossover_operator()
        
        try:
            if self.current_cx_op == 0:
                result = self._crossover_original(population, trial_population)
            elif self.current_cx_op == 1:
                result = self._crossover_dim_adaptive(population, trial_population)
            elif self.current_cx_op == 2:
                result = self._crossover_diversity_blend(population, trial_population)
            elif self.current_cx_op == 3:
                result = self._crossover_block(population, trial_population)
            elif self.current_cx_op == 4:
                result = self._crossover_hybrid_binexp(population, trial_population)
            else:
                result = self._crossover_block(population, trial_population)
        except Exception:
            # Fallback to block crossover (best overall)
            self.current_cx_op = 3
            result = self._crossover_block(population, trial_population)
        
        # Safety: clip and handle NaN
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

        # Track successful F values using Lehmer mean (SHADE-style)
        successful_F = []
        successful_CR = []

        for i in range(min(len(trials), self.NP)):
            if improved_mask[i]:
                # Compute F_i from difference vector (like JADE)
                diff = np.abs(trials[i] - population[i])
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10:
                    fi = np.abs(trials[i] - population[i]) / (norm_diff + 1e-10)
                    successful_F.append(np.mean(fi))
                else:
                    successful_F.append(self.rng.uniform(0.5, 0.9))
                successful_CR.append(self.CR[i])

        # Update archive with improved solutions
        improved_trials = trials[improved_mask]
        if len(improved_trials) > 0:
            self.archive.extend(improved_trials.tolist())
            if len(self.archive) > self.archive_max_size:
                remove_n = len(self.archive) - self.archive_max_size
                self.archive = self.archive[remove_n:]

        # SHADE-style weighted Lehmer mean for global F and CR
        if len(successful_F) > 0:
            f_weights = np.array(successful_F) ** 2
            f_weights = f_weights / (np.sum(f_weights) + 1e-10)
            mean_F = float(np.sum(f_weights * np.array(successful_F)))
            mean_F = np.clip(mean_F, 0.1, 1.0)

            cr_arr = np.array(successful_CR)
            cr_weights = cr_arr ** 2 + 1e-10
            cr_weights = cr_weights / (np.sum(cr_weights) + 1e-10)
            mean_CR = float(np.sum(cr_weights * cr_arr))
            mean_CR = np.clip(mean_CR, 0.0, 1.0)

            # Store in memory buffers (circular)
            if not hasattr(self, 'F_memory'):
                self.F_memory = np.full(100, 0.5)
                self.CR_memory = np.full(100, 0.5)
                self.memory_idx = 0

            self.F_memory[self.memory_idx] = mean_F
            self.CR_memory[self.memory_idx] = mean_CR
            self.memory_idx = (self.memory_idx + 1) % len(self.F_memory)

        # Assign F and CR to all individuals from memory
        if hasattr(self, 'F_memory'):
            valid_F = self.F_memory[self.F_memory > 0.1]
            valid_CR = self.CR_memory[(self.CR_memory > 0.0) & (self.CR_memory < 1.0)]

            if len(valid_F) > 0:
                for i in range(self.NP):
                    self.F[i] = float(np.clip(
                        valid_F[self.rng.integers(0, len(valid_F))] * self.rng.uniform(0.9, 1.1),
                        0.1, 1.0
                    ))

            if len(valid_CR) > 0:
                for i in range(self.NP):
                    self.CR[i] = float(np.clip(
                        valid_CR[self.rng.integers(0, len(valid_CR))] * self.rng.uniform(0.9, 1.1),
                        0.0, 1.0
                    ))
            else:
                self.CR[:] = np.clip(self.CR[:], 0.0, 1.0)
        else:
            self.CR[:] = np.clip(self.CR[:], 0.0, 1.0)

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
                # Still update best before breaking
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
            
            # Fitness improvement signal (relative)
            if abs(pre_best) > 1e-30:
                fit_improvement = float(max(0.0, (pre_best - post_best) / (abs(pre_best) + 1e-30)))
            else:
                fit_improvement = float(max(0.0, pre_best - post_best))
            fit_improvement = float(np.clip(fit_improvement, 0.0, 1.0))
            
            # Track improvement rate for variant_06 style
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
