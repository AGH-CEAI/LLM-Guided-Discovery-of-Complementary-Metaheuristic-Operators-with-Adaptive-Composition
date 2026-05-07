

```python
import numpy as np


class ReflectiveDEWithStrategyPortfolio:
    """
    Differential Evolution using reflective boundary handling and an adaptive
    portfolio of mutation strategies. Each individual maintains its own
    strategy label and per-strategy success memories that drive strategy
    selection and parameter adaptation.
    
    This version adaptively selects among multiple crossover strategies
    using a sliding-window credit assignment mechanism.
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
        self.n_crossover_ops = 5  # number of crossover operators
        self.crossover_window_size = 50  # sliding window size
        self.crossover_rewards = [[] for _ in range(self.n_crossover_ops)]
        self.crossover_attempts = np.zeros(self.n_crossover_ops)
        self.crossover_successes = np.zeros(self.n_crossover_ops)
        self.crossover_alpha = np.ones(self.n_crossover_ops)  # Thompson sampling params
        self.crossover_beta = np.ones(self.n_crossover_ops)
        self.current_crossover_op = 0  # will be selected each generation
        
        # For variant_06 tracking
        self.last_improvement_rate = 0.5
        
        # For variant_02 tracking
        self.dim_success_rate = np.zeros(dim)
        self.dim_attempt_count = np.zeros(dim)

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
        self.crossover_alpha = np.ones(self.n_crossover_ops)
        self.crossover_beta = np.ones(self.n_crossover_ops)
        self.crossover_rewards = [[] for _ in range(self.n_crossover_ops)]
        self.crossover_attempts = np.zeros(self.n_crossover_ops)
        self.crossover_successes = np.zeros(self.n_crossover_ops)
        self.dim_success_rate = np.zeros(self.dim)
        self.dim_attempt_count = np.zeros(self.dim)
        self.last_improvement_rate = 0.5
        
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

    # --- Crossover operator 0: Original binomial ---
    def _crossover_original(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    # --- Crossover operator 1: Block-based (variant_05, 11 wins) ---
    def _crossover_block(self, population, trial_population):
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

    # --- Crossover operator 2: Dimension-adaptive (variant_02, 4 wins) ---
    def _crossover_dim_adaptive(self, population, trial_population):
        n_trials = self.NP
        dim = self.dim

        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_signal = np.mean(trial_better, axis=0)
        self.dim_attempt_count += 1
        alpha = 0.1
        self.dim_success_rate = alpha * dim_signal + (1 - alpha) * self.dim_success_rate

        dim_probs = self.dim_success_rate.copy()
        dim_probs = np.clip(dim_probs, 0.01, 0.99)
        dim_probs = dim_probs / (np.sum(dim_probs) + 1e-10)
        cr_val = 0.3 + 0.4 * np.mean(dim_probs)
        cr_batch = np.full(n_trials, np.clip(cr_val, 0.0, 1.0))

        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring

    # --- Crossover operator 3: Diversity-adaptive blend (variant_03, 2 wins) ---
    def _crossover_blend(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        max_spread = self.upper - self.lower
        pop_spread = np.std(population[:n_trials], axis=0)
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

    # --- Crossover operator 4: Mixed binomial/exponential (variant_09, 2 wins) ---
    def _crossover_mixed(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        cross_mask = np.zeros((n_trials, dim), dtype=bool)

        for i in range(n_trials):
            if self.rng.random() < 0.5:
                mask = self.rng.uniform(size=dim) < cr_batch[i]
                j_rand = self.rng.integers(0, dim)
                mask[j_rand] = True
            else:
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
        """Use Thompson Sampling to select a crossover operator."""
        samples = np.array([
            self.rng.beta(max(self.crossover_alpha[i], 0.01), max(self.crossover_beta[i], 0.01))
            for i in range(self.n_crossover_ops)
        ])
        return int(np.argmax(samples))

    def _update_crossover_reward(self, op_idx, improved_mask, fitness, trial_fitness):
        """Update Thompson Sampling parameters based on improvement rate."""
        n_trials = min(len(improved_mask), len(trial_fitness))
        if n_trials == 0:
            return
        
        success_count = float(np.sum(improved_mask[:n_trials]))
        fail_count = float(n_trials - success_count)
        
        # Compute fitness improvement magnitude
        improvements = np.where(
            improved_mask[:n_trials],
            np.clip(fitness[:n_trials] - trial_fitness[:n_trials], 0, None),
            0.0
        )
        total_improvement = float(np.sum(improvements))
        
        # Normalize improvement as a quality signal
        max_fit = float(np.max(np.abs(fitness[:n_trials])) + 1e-10)
        normalized_improvement = total_improvement / (max_fit * n_trials + 1e-10)
        
        # Update Thompson Sampling parameters with decay
        decay = 0.95
        for i in range(self.n_crossover_ops):
            self.crossover_alpha[i] = max(1.0, self.crossover_alpha[i] * decay)
            self.crossover_beta[i] = max(1.0, self.crossover_beta[i] * decay)
        
        # Add reward for selected operator
        reward = success_count / n_trials + 0.5 * normalized_improvement
        self.crossover_alpha[op_idx] += float(success_count) + reward * 2.0
        self.crossover_beta[op_idx] += float(fail_count)
        
        # Track for improvement rate
        self.last_improvement_rate = float(success_count / n_trials) if n_trials > 0 else 0.5

    def _crossover_batch(self, population, trial_population):
        """Adaptive crossover that selects among multiple strategies."""
        op_idx = self.current_crossover_op
        
        try:
            if op_idx == 0:
                return self._crossover_original(population, trial_population)
            elif op_idx == 1:
                return self._crossover_block(population, trial_population)
            elif op_idx == 2:
                return self._crossover_dim_adaptive(population, trial_population)
            elif op_idx == 3:
                return self._crossover_blend(population, trial_population)
            elif op_idx == 4:
                return self._crossover_mixed(population, trial_population)
            else:
                return self._crossover_original(population, trial_population)
        except Exception:
            # Fallback to original on any error
            return self._crossover_original(population, trial_population)

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

        generation = 0

        while not stopping_condition():
            # Select crossover operator for this generation using Thompson Sampling
            self.current_crossover_op = self._select_crossover_operator()

            trials = self._mutate_batch(population, fitness)
            trials = self._crossover_batch(population, trials)
            trials = self._clip_reflective(trials)
            
            # Safety: ensure no NaN or inf in trials
            nan_mask = np.any(~np.isfinite(trials), axis=1)
            if np.any(nan_mask):
                trials[nan_mask] = population[nan_mask]
            
            trial_fitness = self._evaluate_batch(trials, func)

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]

            if len(trial_fitness) == 0:
                break

            if stopping_condition():
                # Check if any trial is better before breaking
                valid_idx = np.argmin(trial_fitness)
                if float(trial_fitness[valid_idx]) < f_opt:
                    f_opt = float(trial_fitness[valid_idx])
                    x_opt = trials[valid_idx].copy()
                break

            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            self.population_fitness = fitness.copy()

            current_best_idx = int(np.argmin(fitness))
            if float(fitness[current_best_idx]) < f_opt:
                f_opt = float(fitness[current_best_idx])
                x_opt = population[current_best_idx].copy()

            # Update crossover operator reward
            self._update_crossover_reward(
                self.current_crossover_op, improved_mask, fitness, trial_fitness
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

            generation += 1

        return f_opt, x_opt
```