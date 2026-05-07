This is iteration 5 of 10. Propose ONE replacement implementation for `_adapt_strategy_probabilities`.

Requirements:
- Keep the EXACT function signature: `def _adapt_strategy_probabilities(self):`
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

TASK COVERAGE SUMMARY: 21 covered, 1 UNCOVERED, 2 trivial (out of 24)
Goal: get at least one winning variant for every non-trivial task.

  Task  0: TRIVIAL (already at optimum, error=1.00e-08)
  Task  1: COVERED — best=variant_04_idea_0.py       error=1.438e+00 (orig=3.318e+00, -57%)
  Task  2: TRIVIAL (already at optimum, error=1.00e-08)
  Task  3: COVERED — best=variant_04_idea_0.py       error=1.785e+00 (orig=1.829e+00, -2%)
  Task  4: COVERED — best=variant_03_idea_0.py       error=3.672e-02 (orig=8.728e-02, -58%)
  Task  5: COVERED — best=variant_02_idea_0.py       error=6.338e+00 (orig=3.740e+01, -83%)
  Task  6: COVERED — best=variant_03_idea_0.py       error=7.431e+01 (orig=1.655e+02, -55%)
  Task  7: COVERED — best=variant_01_idea_0.py       error=2.865e+00 (orig=3.674e+00, -22%)
  Task  8: COVERED — best=variant_04_idea_0.py       error=1.542e+00 (orig=2.049e+00, -25%)
  Task  9: COVERED — best=variant_03_idea_0.py       error=3.030e+01 (orig=4.063e+01, -25%)
  Task 10: COVERED — best=variant_04_idea_0.py       error=1.198e-08 (orig=3.917e-04, -100%)
  Task 11: COVERED — best=variant_02_idea_0.py       error=3.451e+01 (orig=1.030e+02, -66%)
  Task 12: *** UNCOVERED *** — orig error=7.319e-05, no variant improves on this
  Task 13: COVERED — best=variant_04_idea_0.py       error=3.214e+00 (orig=8.830e+00, -64%)
  Task 14: COVERED — best=variant_04_idea_0.py       error=5.602e+00 (orig=8.034e+00, -30%)
  Task 15: COVERED — best=variant_02_idea_0.py       error=2.074e+00 (orig=2.550e+00, -19%)
  Task 16: COVERED — best=variant_03_idea_0.py       error=5.930e+02 (orig=8.035e+02, -26%)
  Task 17: COVERED — best=variant_04_idea_0.py       error=5.491e+02 (orig=1.258e+04, -96%)
  Task 18: COVERED — best=variant_02_idea_0.py       error=2.135e+01 (orig=2.799e+01, -24%)
  Task 19: COVERED — best=variant_01_idea_0.py       error=5.893e+01 (orig=1.215e+02, -51%)
  Task 20: COVERED — best=variant_03_idea_0.py       error=1.459e+01 (orig=1.622e+01, -10%)
  Task 21: COVERED — best=variant_01_idea_0.py       error=4.738e+00 (orig=4.963e+00, -5%)
  Task 22: COVERED — best=variant_03_idea_0.py       error=7.251e+00 (orig=9.457e+00, -23%)
  Task 23: COVERED — best=variant_04_idea_0.py       error=4.864e+01 (orig=5.676e+01, -14%)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  
-------------------------------------------------------------------------------------------------------------------
0    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
1    3.3177e+00            2.4898e+00            2.1097e+00            1.6187e+00            1.4378e+00            
2    1.0000e-08            1.9957e-08            1.0000e-08            1.0000e-08            1.0000e-08            
3    1.8295e+00            2.0945e+00            1.9373e+00            2.4884e+00            1.7854e+00            
4    8.7280e-02            4.6511e-02            1.6182e-01            3.6716e-02            4.6263e-02            
5    3.7404e+01            8.2055e+01            6.3380e+00            6.9479e+01            2.1829e+01            
6    1.6552e+02            1.5491e+02            1.6233e+02            7.4311e+01            1.4683e+02            
7    3.6743e+00            2.8654e+00            4.1145e+00            3.5026e+00            4.0346e+00            
8    2.0487e+00            1.6211e+00            2.1645e+00            2.2783e+00            1.5419e+00            
9    4.0626e+01            3.0446e+01            4.8213e+01            3.0297e+01            5.6203e+01            
10   3.9173e-04            1.0882e+00            1.4204e-08            1.9362e-07            1.1981e-08            
11   1.0300e+02            1.0059e+02            3.4514e+01            5.7941e+01            5.5757e+01            
12   7.3192e-05            3.2372e+00            3.2985e-04            1.1373e-04            4.0432e-03            
13   8.8299e+00            2.0797e+01            7.7180e+00            1.6507e+01            3.2140e+00            
14   8.0335e+00            6.2898e+00            9.3682e+00            6.9202e+00            5.6023e+00            
15   2.5496e+00            2.7994e+00            2.0739e+00            2.5918e+00            2.7228e+00            
16   8.0353e+02            1.0584e+03            8.5189e+02            5.9301e+02            1.2940e+03            
17   1.2580e+04            1.8762e+04            5.8878e+02            4.0968e+03            5.4913e+02            
18   2.7990e+01            3.5975e+01            2.1352e+01            3.0939e+01            2.5545e+01            
19   1.2145e+02            5.8934e+01            1.3241e+02            7.6860e+01            1.0777e+02            
20   1.6216e+01            1.8952e+01            1.6118e+01            1.4594e+01            1.9072e+01            
21   4.9630e+00            4.7378e+00            5.1521e+00            4.8408e+00            5.2090e+00            
22   9.4574e+00            7.3623e+00            1.1218e+01            7.2506e+00            1.1602e+01            
23   5.6761e+01            4.9511e+01            6.2799e+01            5.2037e+01            4.8643e+01            

TASKS WHERE NO VARIANT CURRENTLY BEATS THE ORIGINAL (priority targets):
3, 12, 21

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to perform WELL on the UNCOVERED tasks listed above, even at the cost of being worse on tasks where other variants already succeed. Reason explicitly:
1. What property of those uncovered tasks (multimodality, ill-conditioning, separability, ruggedness, noise, deceptive local optima, etc.) are the existing operators failing to handle?
2. What specific mechanism in your proposed operator addresses that property?
3. Why is this approach fundamentally different from the prior variants?

Current implementation:
```python
def _adapt_strategy_probabilities(self):
        total_success = np.sum(self.strategy_success_fitness) + 1e-10
        probs = self.strategy_success_fitness / total_success
        probs = probs / (np.sum(probs) + 1e-10)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
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
        total_success = np.sum(self.strategy_success_fitness) + 1e-10
        probs = self.strategy_success_fitness / total_success
        probs = probs / (np.sum(probs) + 1e-10)
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
def _adapt_strategy_probabilities(self, ...):
    ...
```