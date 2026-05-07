This is iteration 4 of 10. Propose ONE replacement implementation for `_crossover_batch`.

Requirements:
- Keep the EXACT function signature: `def _crossover_batch(self, population, trial_population):`
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

TASK COVERAGE SUMMARY: 12 covered, 9 UNCOVERED, 3 trivial (out of 24)
Goal: get at least one winning variant for every non-trivial task.

  Task  0: TRIVIAL (already at optimum, error=1.00e-08)
  Task  1: COVERED — best=variant_01_idea_0.py       error=8.618e-01 (orig=1.107e+00, -22%)
  Task  2: TRIVIAL (already at optimum, error=1.00e-08)
  Task  3: COVERED — best=variant_02_idea_0.py       error=2.092e+00 (orig=2.306e+00, -9%)
  Task  4: COVERED — best=variant_01_idea_0.py       error=7.605e-02 (orig=1.905e-01, -60%)
  Task  5: COVERED — best=variant_01_idea_0.py       error=6.576e+00 (orig=1.165e+01, -44%)
  Task  6: COVERED — best=variant_01_idea_0.py       error=2.108e+02 (orig=2.584e+02, -18%)
  Task  7: COVERED — best=variant_02_idea_0.py       error=2.903e+00 (orig=3.288e+00, -12%)
  Task  8: *** UNCOVERED *** — orig error=1.597e+00, no variant improves on this
  Task  9: *** UNCOVERED *** — orig error=4.244e+01, no variant improves on this
  Task 10: TRIVIAL (already at optimum, error=1.00e-08)
  Task 11: *** UNCOVERED *** — orig error=2.937e+01, no variant improves on this
  Task 12: *** UNCOVERED *** — orig error=7.683e-04, no variant improves on this
  Task 13: COVERED — best=variant_02_idea_0.py       error=2.768e+00 (orig=3.807e+00, -27%)
  Task 14: COVERED — best=variant_03_idea_0.py       error=3.727e+00 (orig=9.937e+00, -62%)
  Task 15: *** UNCOVERED *** — orig error=2.121e+00, no variant improves on this
  Task 16: *** UNCOVERED *** — orig error=9.661e+02, no variant improves on this
  Task 17: COVERED — best=variant_02_idea_0.py       error=6.428e+02 (orig=9.687e+02, -34%)
  Task 18: COVERED — best=variant_02_idea_0.py       error=2.188e+01 (orig=2.302e+01, -5%)
  Task 19: COVERED — best=variant_03_idea_0.py       error=1.187e+02 (orig=1.368e+02, -13%)
  Task 20: *** UNCOVERED *** — orig error=1.725e+01, no variant improves on this
  Task 21: *** UNCOVERED *** — orig error=5.168e+00, no variant improves on this
  Task 22: *** UNCOVERED *** — orig error=1.070e+01, no variant improves on this
  Task 23: COVERED — best=variant_03_idea_0.py       error=4.709e+01 (orig=6.297e+01, -25%)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  
---------------------------------------------------------------------------------------------
0    1.0000e-08            1.0000e-08            1.0000e-08            2.4567e+01            
1    1.1071e+00            8.6180e-01            1.6721e+00            8.3405e+01            
2    1.0000e-08            1.0000e-08            1.0000e-08            2.1230e+02            
3    2.3062e+00            2.3806e+00            2.0920e+00            1.7720e+02            
4    1.9050e-01            7.6055e-02            9.4396e-01            4.2198e+00            
5    1.1647e+01            6.5756e+00            3.3899e+01            3.8044e+08            
6    2.5842e+02            2.1076e+02            8.8082e+02            1.5830e+03            
7    3.2875e+00            4.3809e+00            2.9025e+00            2.1998e+02            
8    1.5967e+00            1.6324e+00            2.0592e+00            3.3936e+01            
9    4.2441e+01            6.0499e+01            1.2123e+02            1.4130e+02            
10   1.0000e-08            2.7443e-03            5.1295e-06            1.9832e+02            
11   2.9365e+01            8.9417e+02            1.7464e+02            5.3318e+02            
12   7.6831e-04            7.0360e-02            4.0143e-03            1.4819e+02            
13   3.8073e+00            2.7762e+01            2.7684e+00            3.8522e+01            
14   9.9372e+00            1.3729e+01            1.4824e+01            3.7275e+00            
15   2.1205e+00            4.0062e+00            3.0802e+00            3.8170e+00            
16   9.6615e+02            3.5575e+03            4.8233e+03            8.9860e+03            
17   9.6874e+02            8.6217e+04            6.4282e+02            1.5957e+05            
18   2.3024e+01            5.9257e+01            2.1881e+01            8.3839e+01            
19   1.3684e+02            2.9563e+02            2.8991e+02            1.1874e+02            
20   1.7246e+01            2.1044e+01            2.5915e+01            2.9182e+01            
21   5.1676e+00            5.7260e+00            5.5822e+00            5.8595e+00            
22   1.0704e+01            1.6847e+01            1.6504e+01            1.3481e+01            
23   6.2969e+01            8.6978e+01            9.1237e+01            4.7094e+01            

TASKS WHERE NO VARIANT CURRENTLY BEATS THE ORIGINAL (priority targets):
3, 8, 9, 11, 12, 15, 16, 18, 20, 21, 22

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to perform WELL on the UNCOVERED tasks listed above, even at the cost of being worse on tasks where other variants already succeed. Reason explicitly:
1. What property of those uncovered tasks (multimodality, ill-conditioning, separability, ruggedness, noise, deceptive local optima, etc.) are the existing operators failing to handle?
2. What specific mechanism in your proposed operator addresses that property?
3. Why is this approach fundamentally different from the prior variants?

Current implementation:
```python
def _crossover_batch(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        cross_mask = self.rng.uniform(size=(n_trials, self.dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, self.dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population)
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

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _crossover_batch(self, ...):
    ...
```