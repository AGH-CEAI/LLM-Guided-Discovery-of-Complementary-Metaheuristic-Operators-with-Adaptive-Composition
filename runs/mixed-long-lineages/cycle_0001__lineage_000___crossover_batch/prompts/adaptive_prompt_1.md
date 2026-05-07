Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_crossover_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            2.456693e+01            -inf                    1.000000e-08            1.000000e-08            -inf                    -inf                    1.000000e-08            1.000000e-08            
1      1.107109e+00            8.617951e-01            1.672137e+00            8.340452e+01            -inf                    8.644825e-01            8.868031e+00            -inf                    -inf                    8.399589e-01            8.952110e+00            
2      1.000000e-08            1.000000e-08            1.000000e-08            2.122989e+02            -inf                    1.000000e-08            1.000000e-08            -inf                    -inf                    1.000000e-08            1.000000e-08            
3      2.306179e+00            2.380622e+00            2.091964e+00            1.772002e+02            -inf                    1.904544e+00            3.875247e+01            -inf                    -inf                    2.296667e+00            3.957047e+01            
4      1.904959e-01            7.605467e-02            9.439631e-01            4.219794e+00            -inf                    1.000000e-08            1.420065e+00            -inf                    -inf                    1.000000e-08            1.458202e+00            
5      1.164725e+01            6.575563e+00            3.389886e+01            3.804352e+08            -inf                    2.157402e+00            6.677075e+05            -inf                    -inf                    7.382069e+01            5.511174e+05            
6      2.584171e+02            2.107599e+02            8.808194e+02            1.582994e+03            -inf                    1.587387e+02            8.729172e+02            -inf                    -inf                    9.773424e+01            9.525875e+02            
7      3.287525e+00            4.380899e+00            2.902531e+00            2.199827e+02            -inf                    3.422993e+00            3.757209e+01            -inf                    -inf                    5.984135e+00            3.869998e+01            
8      1.596721e+00            1.632447e+00            2.059234e+00            3.393614e+01            -inf                    1.990568e+00            9.299270e+00            -inf                    -inf                    1.787776e+00            9.316349e+00            
9      4.244114e+01            6.049907e+01            1.212335e+02            1.413024e+02            -inf                    3.366872e+01            1.325130e+02            -inf                    -inf                    3.459975e+01            1.381510e+02            
10     1.000000e-08            2.744290e-03            5.129506e-06            1.983165e+02            -inf                    6.844011e-08            4.834209e-07            -inf                    -inf                    1.365659e-07            5.482675e-07            
11     2.936540e+01            8.941666e+02            1.746446e+02            5.331798e+02            -inf                    2.328746e+01            1.320935e+03            -inf                    -inf                    1.979889e+02            1.383869e+03            
12     7.683095e-04            7.035958e-02            4.014326e-03            1.481883e+02            -inf                    7.148930e-04            5.389933e-05            -inf                    -inf                    1.226448e-03            5.648731e-05            
13     3.807275e+00            2.776181e+01            2.768362e+00            3.852246e+01            -inf                    3.148278e+00            4.603498e+01            -inf                    -inf                    3.297544e+00            4.516245e+01            
14     9.937181e+00            1.372949e+01            1.482372e+01            3.727456e+00            -inf                    8.053232e+00            1.604286e+01            -inf                    -inf                    1.045243e+01            1.569935e+01            
15     2.120513e+00            4.006230e+00            3.080188e+00            3.816993e+00            -inf                    2.008028e+00            4.217692e+00            -inf                    -inf                    3.260374e+00            4.161876e+00            
16     9.661467e+02            3.557476e+03            4.823283e+03            8.985991e+03            -inf                    6.294379e+02            9.268548e+03            -inf                    -inf                    1.214476e+03            9.681499e+03            
17     9.687419e+02            8.621739e+04            6.428166e+02            1.595684e+05            -inf                    7.727308e+02            2.129455e+05            -inf                    -inf                    1.987220e+03            2.116394e+05            
18     2.302433e+01            5.925686e+01            2.188055e+01            8.383928e+01            -inf                    2.386831e+01            8.205841e+01            -inf                    -inf                    2.840312e+01            8.123596e+01            
19     1.368401e+02            2.956338e+02            2.899073e+02            1.187377e+02            -inf                    1.098133e+02            4.017550e+02            -inf                    -inf                    1.700364e+02            4.074741e+02            
20     1.724633e+01            2.104448e+01            2.591496e+01            2.918176e+01            -inf                    1.644603e+01            2.206206e+01            -inf                    -inf                    1.727600e+01            2.246692e+01            
21     5.167630e+00            5.726037e+00            5.582188e+00            5.859504e+00            -inf                    4.952353e+00            5.917367e+00            -inf                    -inf                    5.331075e+00            5.912694e+00            
22     1.070434e+01            1.684740e+01            1.650445e+01            1.348109e+01            -inf                    1.054439e+01            2.111840e+01            -inf                    -inf                    1.446212e+01            2.085669e+01            
23     6.296859e+01            8.697846e+01            9.123660e+01            4.709363e+01            -inf                    5.124959e+01            9.650698e+01            -inf                    -inf                    6.643196e+01            9.797223e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_09_idea_0.py  (error=8.399589e-01)
Task  2: SKIPPED (trivial)
Task  3: variant_05_idea_0.py  (error=1.904544e+00)
Task  4: variant_05_idea_0.py  (error=1.000000e-08)
Task  5: variant_05_idea_0.py  (error=2.157402e+00)
Task  6: variant_09_idea_0.py  (error=9.773424e+01)
Task  7: variant_02_idea_0.py  (error=2.902531e+00)
Task  8: original.py  (error=1.596721e+00)
Task  9: variant_05_idea_0.py  (error=3.366872e+01)
Task 10: original.py  (error=1.000000e-08)
Task 11: variant_05_idea_0.py  (error=2.328746e+01)
Task 12: variant_06_idea_0.py  (error=5.389933e-05)
Task 13: variant_02_idea_0.py  (error=2.768362e+00)
Task 14: variant_03_idea_0.py  (error=3.727456e+00)
Task 15: variant_05_idea_0.py  (error=2.008028e+00)
Task 16: variant_05_idea_0.py  (error=6.294379e+02)
Task 17: variant_02_idea_0.py  (error=6.428166e+02)
Task 18: variant_02_idea_0.py  (error=2.188055e+01)
Task 19: variant_05_idea_0.py  (error=1.098133e+02)
Task 20: variant_05_idea_0.py  (error=1.644603e+01)
Task 21: variant_05_idea_0.py  (error=4.952353e+00)
Task 22: variant_05_idea_0.py  (error=1.054439e+01)
Task 23: variant_03_idea_0.py  (error=4.709363e+01)

WIN COUNTS:
  variant_05_idea_0.py: 11 wins
  variant_02_idea_0.py: 4 wins
  variant_09_idea_0.py: 2 wins
  original.py: 2 wins
  variant_03_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (4 wins) ---
```python
def _crossover_batch(self, population, trial_population):
        n_trials, dim = trial_population.shape
        # Track per-dimension success: positive when trial improves fitness at that dim
        if not hasattr(self, 'dim_success_rate'):
            self.dim_success_rate = np.zeros(dim)
            self.dim_attempt_count = np.zeros(dim)
        # Compute per-dimension improvement signals
        trial_vs_parent = trial_population - population[:n_trials]
        trial_better = (trial_population < population[:n_trials]).astype(float)
        dim_signal = np.mean(trial_better, axis=0)
        self.dim_attempt_count += 1
        # Running average with momentum
        alpha = 0.1
        self.dim_success_rate = alpha * dim_signal + (1 - alpha) * self.dim_success_rate
        # Build adaptive CR per individual based on dimension history
        cr_batch = np.ones(n_trials) * 0.3  # Start conservative
        for i in range(n_trials):
            # Use global dim success rate to weight dimension selection
            dim_probs = self.dim_success_rate.copy()
            dim_probs = np.clip(dim_probs, 0.01, 0.99)
            dim_probs = dim_probs / (np.sum(dim_probs) + 1e-10)
            cr_batch[i] = 0.3 + 0.4 * np.mean(dim_probs)
        cr_batch = np.clip(cr_batch, 0.0, 1.0)
        # Binomial crossover with adaptive per-dim probabilities
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population[:n_trials])
        return offspring
```

# --- From variant_03_idea_0.py (2 wins) ---
```python
def _crossover_batch(self, population, trial_population):
            cr_batch = self.CR.copy()
            n_trials = self.NP
            dim = self.dim

            # Compute population diversity (normalized spread)
            centroid = np.mean(population, axis=0)
            max_spread = self.upper - self.lower
            pop_spread = np.std(population, axis=0)
            normalized_diversity = np.mean(pop_spread / (max_spread + 1e-10))

            # Diversity-adaptive blending weight: low diversity -> more trial exploration
            # Clamp to [0.1, 0.5] to avoid extreme behavior
            alpha_base = np.clip(0.5 - normalized_diversity, 0.1, 0.5)

            # Per-individual diversity weight with noise for exploration
            alpha = np.clip(
                alpha_base + self.rng.uniform(-0.05, 0.05, size=n_trials),
                0.05, 0.6
            )

            # Generate binomial crossover mask (standard DE binomial)
            cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
            j_rand = self.rng.integers(0, dim, size=n_trials)
            for i in range(n_trials):
                cross_mask[i, j_rand[i]] = True

            # Arithmetic blend of parent and trial (continuous exploration)
            blended = (
                alpha[:, np.newaxis] * trial_population +
                (1.0 - alpha[:, np.newaxis]) * population
            )

            # Binomial mask selects discrete dimensions, blended fills continuous exploration
            # Where mask=True: use trial (with blend weight already applied)
            # Where mask=False: use parent
            offspring = np.where(cross_mask, blended, population)

            return offspring
```

# --- From variant_05_idea_0.py (11 wins) ---
```python
def _crossover_batch(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        # Group dimensions into blocks for correlated decisions
        block_size = max(1, dim // 8)
        n_blocks = (dim + block_size - 1) // block_size

        # Create block-level crossover decisions
        cross_blocks = self.rng.uniform(size=(n_trials, n_blocks)) < cr_batch[:, np.newaxis]

        # Expand block decisions to dimension-level mask
        cross_mask = np.zeros((n_trials, dim), dtype=bool)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, dim)
            cross_mask[:, start:end] = cross_blocks[:, b:b+1]

        # Ensure at least one dimension per individual
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            if not cross_mask[i, :].any():
                cross_mask[i, j_rand[i]] = True

        offspring = np.where(cross_mask, trial_population, population)
        return offspring
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, trial_population):
        n_trials = self.NP
        dim = self.dim
        cr_batch = self.CR.copy()

        # Compute mutation vector differences (magnitude per individual)
        diff_mag = np.linalg.norm(trial_population - population, axis=1, keepdims=True)
        diff_mag = np.maximum(diff_mag, 1e-10)

        # Normalize per-dimension differences -> exploration probability
        diff_norm = np.abs(trial_population - population) / diff_mag

        # Temperature inversely correlated with recent improvement rate
        if hasattr(self, 'last_improvement_rate'):
            temp = max(0.05, 2.0 * (1.0 - self.last_improvement_rate))
        else:
            temp = 1.0

        # Softmax over dimensions for each trial vector
        exp_diff = np.exp(diff_norm / (temp + 1e-10))
        denom = np.sum(exp_diff, axis=1, keepdims=True) + 1e-10
        explore_prob = exp_diff / denom

        # Combine base CR with exploration probability
        cr_expanded = cr_batch[:, np.newaxis]
        cross_prob = cr_expanded * explore_prob + (1 - cr_expanded) * (1.0 / dim)
        cross_prob = np.clip(cross_prob, 1e-6, 1.0)

        # Generate crossover mask
        cross_mask = self.rng.uniform(size=(n_trials, dim)) < cross_prob

        # Ensure at least one dimension from trial
        j_rand = self.rng.integers(0, dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True

        offspring = np.where(cross_mask, trial_population, population)
        return offspring
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _crossover_batch(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        dim = self.dim

        cross_mask = np.zeros((n_trials, dim), dtype=bool)

        for i in range(n_trials):
            if self.rng.random() < 0.5:
                # Binomial crossover (standard)
                mask = self.rng.uniform(size=dim) < cr_batch[i]
                j_rand = self.rng.integers(0, dim)
                mask[j_rand] = True
            else:
                # Exponential crossover: copy a contiguous segment from trial
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

        offspring = np.where(cross_mask, trial_population, population)
        return offspring
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
import numpy as np


class ReflectiveDEWithStrategyPortfolio:
    """
    Differential Evolution using reflective boundary handling and an adaptive
    portfolio of mutation strategies. Each individual maintains its own
    strategy label and per-strategy success memories that drive strategy
    selection and parameter adaptation.
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
            "rand_1",      # classical rand/1
            "best_1",      # classical best/1
            "current_1",   # current-to-best/1 variant
            "rand_2",      # rand/2 for diversity
            "best_2",      # best/2 for convergence
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
            self.strategy_success_count[s] += count_s
            self.strategy_attempt_count[s] += np.sum(self.strategy_idx == s)
            if count_s > 0:
                self.strategy_success_fitness[s] = (
                    0.5 * self.strategy_success_fitness[s] + 0.5 * count_s
                )

    def _adapt_strategy_probabilities(self):
        # Compute success rates per strategy (avoid division by zero)
        success_rates = np.zeros(self.n_strategies)
        valid_strategies = self.strategy_attempt_count > 0
        success_rates[valid_strategies] = (
            self.strategy_success_count[valid_strategies] / 
            self.strategy_attempt_count[valid_strategies]
        )

        # Get current best fitness
        best_fitness = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.min(self.archive) if self.archive else 0.0

        # Threshold for "already near-optimal" - conservative mode
        # Use a relative threshold: 1.1x the best fitness seen so far
        ref_fitness = getattr(self, 'reference_fitness', 1e-3)
        threshold = ref_fitness * 1.5 if ref_fitness > 0 else 1e-4

        # Dual-mode: conservative if already good, adaptive otherwise
        if best_fitness < threshold:
            # Conservative mode: bias toward original uniform distribution
            # This prevents harmful drift on already-converged tasks like #12
            orig_prob = 1.0 / self.n_strategies
            adaptive_prob = success_rates / (np.sum(success_rates) + 1e-10)
            alpha = 0.85  # Strong weight on original distribution
            probs = alpha * orig_prob + (1 - alpha) * adaptive_prob
        else:
            # Adaptive mode: use softmax with temperature on success rates
            # Low temperature = more exploitative, high = more exploratory
            temperature = 0.3
            exp_rates = np.exp(success_rates / (temperature + 1e-10))
            probs = exp_rates / (np.sum(exp_rates) + 1e-10)

        # Numerical safety
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (np.sum(probs) + 1e-10)

        # Assign strategies
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

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
        return np.mean(distances)

    def _check_stagnation(self, fitness):
        current_best = np.min(fitness)
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
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()

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
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()

            self._update_strategy_success(improved_mask)
            self._adapt_strategy_probabilities()
            self._adapt_F_CR_jade_style(population, fitness, trials, trial_fitness, improved_mask)

            div = self._compute_diversity(population)
            self.diversity_history.append(div)

            if self._check_stagnation(fitness):
                population, fitness = self._reinitialize_stagnant_subpopulation(population, fitness)
                self.archive = []
                self.F[:] = 0.5
                self.CR[:] = 0.5

        return f_opt, x_opt

```