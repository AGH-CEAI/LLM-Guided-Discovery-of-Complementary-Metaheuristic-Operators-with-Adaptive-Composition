This is iteration 5 of 10. Propose ONE replacement implementation for `_crossover_block`.

Requirements:
- Keep the EXACT function signature: `def _crossover_block(self, population, trial_population):`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that wins on EACH task.
- Tasks marked UNCOVERED below are critical gaps — your variant should
  specifically target those.

TASK COVERAGE SUMMARY: 11 covered, 8 UNCOVERED, 5 trivial (out of 24)
Goal: get at least one winning variant for every non-trivial task.

  Task  0: TRIVIAL (already at optimum, error=1.00e-08)
  Task  1: *** UNCOVERED *** — orig error=1.222e+00, no variant improves on this
  Task  2: TRIVIAL (already at optimum, error=1.00e-08)
  Task  3: *** UNCOVERED *** — orig error=1.501e+00, no variant improves on this
  Task  4: TRIVIAL (already at optimum, error=1.00e-08)
  Task  5: *** UNCOVERED *** — orig error=1.444e+01, no variant improves on this
  Task  6: COVERED — best=variant_04_idea_0.py       error=3.503e+01 (orig=3.663e+01, -4%)
  Task  7: *** UNCOVERED *** — orig error=3.852e+00, no variant improves on this
  Task  8: *** UNCOVERED *** — orig error=1.635e+00, no variant improves on this
  Task  9: COVERED — best=variant_01_idea_0.py       error=1.356e+01 (orig=2.150e+01, -37%)
  Task 10: TRIVIAL (already at optimum, error=1.00e-08)
  Task 11: COVERED — best=variant_04_idea_0.py       error=5.144e+01 (orig=8.567e+01, -40%)
  Task 12: TRIVIAL (already at optimum, error=1.82e-08)
  Task 13: COVERED — best=variant_04_idea_0.py       error=1.119e+01 (orig=1.278e+01, -12%)
  Task 14: COVERED — best=variant_03_idea_0.py       error=4.493e+00 (orig=6.927e+00, -35%)
  Task 15: COVERED — best=variant_04_idea_0.py       error=2.642e+00 (orig=2.768e+00, -5%)
  Task 16: COVERED — best=variant_04_idea_0.py       error=6.482e+02 (orig=8.035e+02, -19%)
  Task 17: COVERED — best=variant_04_idea_0.py       error=5.919e+03 (orig=1.046e+04, -43%)
  Task 18: *** UNCOVERED *** — orig error=2.647e+01, no variant improves on this
  Task 19: COVERED — best=variant_03_idea_0.py       error=6.960e+01 (orig=8.725e+01, -20%)
  Task 20: *** UNCOVERED *** — orig error=1.633e+01, no variant improves on this
  Task 21: COVERED — best=variant_03_idea_0.py       error=4.669e+00 (orig=4.884e+00, -4%)
  Task 22: *** UNCOVERED *** — orig error=1.002e+01, no variant improves on this
  Task 23: COVERED — best=variant_03_idea_0.py       error=4.008e+01 (orig=4.183e+01, -4%)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  
-------------------------------------------------------------------------------------------------------------------
0    1.0000e-08            1.0000e-08            -inf                  1.0000e-08            1.0000e-08            
1    1.2220e+00            3.4117e+00            -inf                  4.7127e+00            2.0643e+00            
2    1.0000e-08            1.0000e-08            -inf                  1.0000e-08            1.0000e-08            
3    1.5010e+00            2.8815e+00            -inf                  3.2279e+01            1.7801e+00            
4    1.0000e-08            5.1805e-02            -inf                  1.4478e-01            3.5774e-02            
5    1.4439e+01            1.7878e+02            -inf                  2.3892e+05            1.0726e+02            
6    3.6629e+01            4.0501e+01            -inf                  4.6615e+01            3.5026e+01            
7    3.8524e+00            5.9440e+00            -inf                  1.2132e+01            4.2626e+00            
8    1.6350e+00            1.6812e+00            -inf                  2.0629e+00            1.6900e+00            
9    2.1498e+01            1.3563e+01            -inf                  3.5429e+01            1.7431e+01            
10   1.0000e-08            1.0000e-08            -inf                  1.0000e-08            1.0000e-08            
11   8.5667e+01            8.2849e+01            -inf                  8.7068e+01            5.1440e+01            
12   1.8164e-08            5.0815e-06            -inf                  7.6731e-06            1.0000e-08            
13   1.2779e+01            2.3181e+01            -inf                  2.2028e+01            1.1194e+01            
14   6.9274e+00            6.8553e+00            -inf                  4.4925e+00            7.5770e+00            
15   2.7678e+00            3.8032e+00            -inf                  2.8548e+00            2.6421e+00            
16   8.0353e+02            1.0717e+03            -inf                  9.0653e+02            6.4821e+02            
17   1.0458e+04            6.5205e+03            -inf                  2.0874e+04            5.9192e+03            
18   2.6469e+01            3.1235e+01            -inf                  3.9291e+01            3.0262e+01            
19   8.7248e+01            8.0404e+01            -inf                  6.9603e+01            1.0221e+02            
20   1.6331e+01            1.9223e+01            -inf                  2.0653e+01            1.7557e+01            
21   4.8840e+00            4.8955e+00            -inf                  4.6694e+00            4.9673e+00            
22   1.0019e+01            1.1237e+01            -inf                  1.0915e+01            1.0490e+01            
23   4.1830e+01            4.1422e+01            -inf                  4.0076e+01            4.1652e+01            

TASKS WHERE NO VARIANT CURRENTLY BEATS THE ORIGINAL (priority targets):
1, 3, 5, 6, 7, 8, 15, 18, 20, 21, 22, 23

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to perform WELL on the UNCOVERED tasks listed above, even at the cost of being worse on tasks where other variants already succeed. Reason explicitly:
1. What property of those uncovered tasks (multimodality, ill-conditioning, separability, ruggedness, noise, deceptive local optima, etc.) are the existing operators failing to handle?
2. What specific mechanism in your proposed operator addresses that property?
3. Why is this approach fundamentally different from the prior variants?

Current implementation:
```python
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
```

Full algorithm for context:
```python
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

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _crossover_block(self, ...):
    ...
```