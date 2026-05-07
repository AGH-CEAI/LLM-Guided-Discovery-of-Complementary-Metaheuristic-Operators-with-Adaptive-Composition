

```python
import numpy as np


class ReflectiveDEWithStrategyPortfolio:
    """
    Differential Evolution using reflective boundary handling and an adaptive
    portfolio of mutation strategies. Each individual maintains its own
    strategy label and per-strategy success memories that drive strategy
    selection and parameter adaptation.
    
    This version uses a meta-level Multi-Armed Bandit (Thompson Sampling) to
    select among multiple operator-selection strategies during the run.
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

        # Meta-level: adaptive selection among multiple _adapt methods
        # We have 7 candidate adapt strategies
        self.n_meta_strategies = 7
        self.meta_success_count = np.ones(self.n_meta_strategies)
        self.meta_failure_count = np.ones(self.n_meta_strategies)
        self.meta_current_idx = 0
        self.meta_window_size = 5  # evaluate meta-strategy every N generations
        self.meta_gen_counter = 0
        self.meta_prev_best = np.inf
        self.meta_reward_history = [[] for _ in range(self.n_meta_strategies)]

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

    def _crossover_batch(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        cross_mask = self.rng.uniform(size=(n_trials, self.dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, self.dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population)
        return offspring

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
            self.strategy_attempt_count[s] += np.sum(self.strategy_idx == s)
            self.strategy_success_count[s] += count_s
            if count_s > 0:
                self.strategy_success_fitness[s] = (
                    0.5 * self.strategy_success_fitness[s] + 0.5 * float(count_s)
                )

    # =====================================================================
    # Seven candidate adapt-strategy-probabilities methods
    # =====================================================================

    def _adapt_strategy_probabilities_0(self):
        """Original: simple proportional to success fitness."""
        total_success = np.sum(self.strategy_success_fitness) + 1e-10
        probs = self.strategy_success_fitness / total_success
        probs = probs / (np.sum(probs) + 1e-10)
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_strategy_probabilities_1(self):
        """Thompson sampling (variant_01)."""
        for i in range(self.NP):
            samples = np.empty(self.n_strategies)
            for s in range(self.n_strategies):
                successes = self.strategy_success_count[s]
                attempts = self.strategy_attempt_count[s]
                failures = max(attempts - successes, 0.0)
                alpha = float(successes) + 1.0
                beta = float(failures) + 1.0
                samples[s] = self.rng.beta(alpha, beta)
            self.strategy_idx[i] = int(np.argmax(samples))

    def _adapt_strategy_probabilities_2(self):
        """Entropy-balanced (variant_02)."""
        total_attempts = np.sum(self.strategy_attempt_count) + 1e-10
        success_rate = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
        log_fitness = np.log1p(np.maximum(self.strategy_success_fitness, 0))
        base_score = 0.7 * success_rate + 0.3 * (log_fitness / (np.max(log_fitness) + 1e-10))
        uniform_prob = 1.0 / self.n_strategies
        current_probs = base_score / (np.sum(base_score) + 1e-10)
        entropy_bonus = uniform_prob - current_probs
        entropy_bonus = np.clip(entropy_bonus, 0, 0.3)
        max_score = np.max(current_probs)
        convergence_factor = 1.0 - (max_score - uniform_prob) / (1.0 - uniform_prob + 1e-10)
        convergence_factor = np.clip(convergence_factor, 0.1, 1.0)
        probs = current_probs + entropy_bonus * convergence_factor
        probs = probs / (np.sum(probs) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        probs = probs / (np.sum(probs) + 1e-10)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_strategy_probabilities_3(self):
        """Generation-adaptive blending (variant_03 - 6 wins)."""
        rates = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
        gen = len(self.diversity_history) + 1
        alpha = min(0.9, 0.3 + 0.005 * gen)
        weighted_scores = alpha * self.strategy_success_count + (1 - alpha) * np.log1p(self.strategy_success_count)
        scores = alpha * rates + (1 - alpha) * weighted_scores
        scores = np.maximum(scores, 1e-15)
        total = np.sum(scores)
        probs = scores / (total + 1e-10)
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / np.sum(probs)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_strategy_probabilities_4(self):
        """Rank-based with adaptive temperature (variant_04)."""
        scores = np.zeros(self.n_strategies)
        for s in range(self.n_strategies):
            count = self.strategy_success_count[s]
            fitness_improvement = self.strategy_success_fitness[s]
            scores[s] = np.log1p(count) * (1.0 + np.log1p(fitness_improvement) + 1e-10)
        ranks = np.argsort(np.argsort(scores)) + 1
        weighted_scores = scores * ranks
        score_variance = np.var(weighted_scores) + 1e-10
        T = 0.5 / (1.0 + np.log1p(score_variance))
        scores_shifted = weighted_scores - np.max(weighted_scores)
        exp_scores = np.exp(scores_shifted / (T + 1e-10))
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        if np.any(np.isnan(probs)) or np.sum(probs) < 1e-10:
            probs = np.ones(self.n_strategies) / self.n_strategies
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / np.sum(probs)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_strategy_probabilities_5(self):
        """Rank-based with diversity-sensitive exploration (variant_05)."""
        if np.any(self.strategy_success_count > 0):
            success_rate = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
            ranks = np.argsort(np.argsort(-success_rate)) + 1
            rank_probs = ranks / (np.sum(ranks) + 1e-10)
        else:
            rank_probs = np.ones(self.n_strategies) / self.n_strategies

        stagnation_factor = min(1.0, self.generation_without_improvement / 20.0)
        temperature = 0.5 + 1.5 * stagnation_factor
        entropy_weight = 0.2 + 0.4 * stagnation_factor

        if len(self.diversity_history) >= 10:
            recent_div = np.mean(self.diversity_history[-10:])
            window = self.diversity_history[-50:]
            div_range = max(np.max(window) - np.min(window), 1e-10)
            diversity_signal = np.clip((recent_div - np.min(window)) / div_range, 0.0, 1.0)
        else:
            diversity_signal = 0.5

        entropy_weight = np.clip(entropy_weight * (1.0 - 0.5 * diversity_signal), 0.1, 0.6)
        logits = np.log(rank_probs + 1e-10) / temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        exploration_probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        final_probs = (1.0 - entropy_weight) * rank_probs + entropy_weight * exploration_probs
        final_probs = np.clip(final_probs, 1e-10, 1.0)
        final_probs = final_probs / (np.sum(final_probs) + 1e-10)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=final_probs)

    def _adapt_strategy_probabilities_6(self):
        """Entropy-based softmax on fitness (variant_07)."""
        attempts = self.strategy_attempt_count + 1
        total = np.sum(self.strategy_success_fitness) + 1e-10
        normalized = self.strategy_success_fitness / total
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        max_entropy = np.log(self.n_strategies + 1e-10)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        temperature = 0.1 + 1.9 * normalized_entropy
        temperature = np.clip(temperature, 0.05, 2.0)
        weights = self.strategy_success_fitness + 1e-10
        logits = np.log(weights) / temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / np.sum(probs)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    # =====================================================================
    # Meta-level: select which adapt method to use via Thompson Sampling
    # =====================================================================

    def _meta_select_strategy(self):
        """Use Thompson Sampling to pick the meta-strategy."""
        samples = np.empty(self.n_meta_strategies)
        for m in range(self.n_meta_strategies):
            alpha = float(self.meta_success_count[m])
            beta = float(self.meta_failure_count[m])
            samples[m] = self.rng.beta(max(alpha, 0.1), max(beta, 0.1))
        self.meta_current_idx = int(np.argmax(samples))

    def _meta_update(self, improved):
        """Update the meta-level bandit based on whether improvement occurred."""
        if improved:
            self.meta_success_count[self.meta_current_idx] += 1.0
        else:
            self.meta_failure_count[self.meta_current_idx] += 1.0

    def _adapt_strategy_probabilities(self):
        """Dispatch to the currently selected meta-strategy."""
        idx = self.meta_current_idx
        if idx == 0:
            self._adapt_strategy_probabilities_0()
        elif idx == 1:
            self._adapt_strategy_probabilities_1()
        elif idx == 2:
            self._adapt_strategy_probabilities_2()
        elif idx == 3:
            self._adapt_strategy_probabilities_3()
        elif idx == 4:
            self._adapt_strategy_probabilities_4()
        elif idx == 5:
            self._adapt_strategy_probabilities_5()
        elif idx == 6:
            self._adapt_strategy_probabilities_6()
        else:
            self._adapt_strategy_probabilities_3()  # fallback to best

    def _adapt_F_CR_jade_style(self, population, fitness, trials, trial_fitness, improved_mask):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved_trials = trials[improved_mask]
        improved_fit = trial_fitness[improved_mask]
        improved_parents = population[:len(trials)][improved_mask]

        if len(improved_trials) > 0:
            fi_values = np.abs(improved_trials - improved_parents)
            fi_values = fi_values / (np.linalg.norm(fi_values, axis=1, keepdims=True) + 1e-10)
            fi_means = np.mean(fi_values, axis=1)
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
            return np.inf, np.zeros(self.dim)

        best_idx = np.argmin(fitness)
        f_opt = float(fitness[best_idx])
        x_opt = population[best_idx].copy()

        # Initialize meta-level
        self.meta_prev_best = f_opt
        self.meta_gen_counter = 0
        self._meta_select_strategy()

        while not stopping_condition():
            trials = self._mutate_batch(population, fitness)
            trials = self._crossover_batch(population, trials)
            trials = self._clip_reflective(trials)
            trial_fitness = self._evaluate_batch(trials, func)

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]

            if len(trial_fitness) == 0:
                break

            if stopping_condition():
                break

            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )

            current_best_idx = np.argmin(fitness)
            current_best_f = float(fitness[current_best_idx])
            if current_best_f < f_opt:
                f_opt = current_best_f
                x_opt = population[current_best_idx].copy()

            self._update_strategy_success(improved_mask)
            self._adapt_strategy_probabilities()
            self._adapt_F_CR_jade_style(population, fitness, trials, trial_fitness, improved_mask)

            div = self._compute_diversity(population)
            self.diversity_history.append(div)

            # Meta-level update: every meta_window_size generations, assess and switch
            self.meta_gen_counter += 1
            if self.meta_gen_counter >= self.meta_window_size:
                # Check if improvement occurred during this window
                improved = (f_opt < self.meta_prev_best - 1e-12)
                # Also consider fraction of improved individuals as a softer signal
                improvement_rate = float(np.sum(improved_mask)) / max(len(improved_mask), 1)
                
                if improved or improvement_rate > 0.15:
                    self.meta_success_count[self.meta_current_idx] += 1.0 + 5.0 * float(improved)
                else:
                    self.meta_failure_count[self.meta_current_idx] += 1.0
                
                # Decay counts slowly to allow adaptation over time
                decay = 0.995
                self.meta_success_count *= decay
                self.meta_failure_count *= decay
                # Ensure minimums
                self.meta_success_count = np.maximum(self.meta_success_count, 0.5)
                self.meta_failure_count = np.maximum(self.meta_failure_count, 0.5)
                
                self.meta_prev_best = f_opt
                self.meta_gen_counter = 0
                self._meta_select_strategy()

            if self._check_stagnation(fitness):
                population, fitness = self._reinitialize_stagnant_subpopulation(population, fitness)
                self.archive = []
                self.F[:] = 0.5
                self.CR[:] = 0.5

        return f_opt, x_opt
```