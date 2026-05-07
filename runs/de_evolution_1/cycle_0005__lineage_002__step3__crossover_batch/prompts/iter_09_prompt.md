This is iteration 9 of 10. Propose ONE replacement implementation for `_crossover_batch`.

Requirements:
- Keep the EXACT function signature: `def _crossover_batch(self, population, mutants, cr_values):`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that reaches error <= 1e-08 on EACH task.
- Tasks marked UNSOLVED below are critical gaps. The bigger the remaining
  error, the higher the priority — your variant should specifically target
  the WORST unsolved tasks at the top of the priority list.

TASK COVERAGE SUMMARY: 7 SOLVED (<= 1e-08), 17 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 23: *** UNSOLVED *** — best=8.756e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.9 decades above target)
  Task 19: *** UNSOLVED *** — best=5.395e+00 by variant_07_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 21: *** UNSOLVED *** — best=4.164e+00 by variant_08_idea_0.py      (target=1e-08, ~+8.6 decades above target)
  Task 20: *** UNSOLVED *** — best=3.724e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.6 decades above target)
  Task 17: *** UNSOLVED *** — best=3.120e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.671e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task 22: *** UNSOLVED *** — best=2.202e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task 14: *** UNSOLVED *** — best=2.120e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task 15: *** UNSOLVED *** — best=1.124e+00 by variant_08_idea_0.py      (target=1e-08, ~+8.1 decades above target)
  Task 13: *** UNSOLVED *** — best=9.310e-03 by variant_06_idea_0.py      (target=1e-08, ~+6.0 decades above target)
  Task 11: *** UNSOLVED *** — best=7.947e-04 by variant_06_idea_0.py      (target=1e-08, ~+4.9 decades above target)
  Task  9: *** UNSOLVED *** — best=1.856e-04 by variant_08_idea_0.py      (target=1e-08, ~+4.3 decades above target)
  Task  3: *** UNSOLVED *** — best=1.103e-04 by variant_04_idea_0.py      (target=1e-08, ~+4.0 decades above target)
  Task 12: *** UNSOLVED *** — best=4.286e-06 by original.py               (target=1e-08, ~+2.6 decades above target)
  Task  7: *** UNSOLVED *** — best=3.259e-06 by variant_04_idea_0.py      (target=1e-08, ~+2.5 decades above target)
  Task  6: *** UNSOLVED *** — best=5.309e-08 by variant_04_idea_0.py      (target=1e-08, ~+0.7 decades above target)
  Task 16: *** UNSOLVED *** — best=2.000e-08 by variant_06_idea_0.py      (target=1e-08, ~+0.3 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  0: SOLVED  — best=1.000e-08 by original.py                   
  Task  1: SOLVED  — best=1.000e-08 by original.py                   
  Task  2: SOLVED  — best=1.000e-08 by original.py                   
  Task  4: SOLVED  — best=1.000e-08 by original.py                   
  Task  5: SOLVED  — best=1.000e-08 by original.py                   
  Task  8: SOLVED  — best=1.000e-08 by original.py                   
  Task 10: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  variant_07_idea_0.py  variant_08_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.3069e-07            1.0000e-08            1.0000e-08            
1    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            2.7690e-07            1.0000e-08            1.0000e-08            
2    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            2.8555e-05            1.0000e-08            1.0000e-08            
3    1.3448e-04            1.3579e-04            1.4155e-04            1.3559e-04            1.1026e-04            1.3651e-04            3.9889e-04            1.3547e-04            1.3625e-04            
4    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            2.5761e-02            1.0000e-08            1.0000e-08            
5    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
6    5.6860e-08            5.9742e-08            6.5172e-08            5.6463e-08            5.3090e-08            5.8881e-08            4.8194e-07            5.7877e-08            5.9439e-08            
7    4.0358e-06            4.0382e-06            3.9540e-06            3.9042e-06            3.2586e-06            4.0556e-06            1.6613e-05            3.8393e-06            4.0656e-06            
8    1.0000e-08            7.7594e-04            7.7680e-04            1.0000e-08            4.9541e-04            1.0000e-08            1.9869e-03            1.0000e-08            1.0000e-08            
9    1.9561e-04            2.0132e-04            7.6398e-01            1.8802e-04            1.8794e-04            1.9042e-04            5.7113e-04            1.8791e-04            1.8560e-04            
10   1.0000e-08            7.0042e-02            2.1145e-01            5.2974e+00            8.6548e-01            5.1127e+00            8.5747e-05            1.7154e+00            4.0667e+00            
11   2.6340e+00            5.0557e+00            2.4421e+00            3.9634e+00            6.2540e+00            2.1849e+00            7.9470e-04            1.7090e+00            2.1656e+00            
12   4.2858e-06            8.6708e-01            4.8539e+00            2.4321e+00            2.0447e+01            1.7593e+00            8.0111e-01            1.1421e+00            1.9136e+00            
13   1.1971e+00            1.4689e+00            1.8666e+00            9.7751e-01            1.0610e+00            1.8226e+00            9.3101e-03            1.4171e+00            8.8694e-01            
14   2.4715e+00            2.4584e+00            2.4043e+00            2.5030e+00            3.7814e+00            2.5242e+00            2.1198e+00            2.4423e+00            2.5316e+00            
15   1.1487e+00            4.0582e+00            1.1884e+00            1.2258e+00            2.0488e+00            1.1513e+00            1.1574e+00            1.2156e+00            1.1236e+00            
16   9.6858e+01            9.3417e+01            1.5903e+02            7.1900e+01            1.0288e-02            6.8869e+01            2.0000e-08            6.1449e+01            3.6022e+01            
17   3.1201e+00            3.3332e+00            3.1201e+00            3.1201e+00            3.1212e+00            3.1201e+00            3.1201e+00            3.1201e+00            3.1201e+00            
18   2.6712e+00            3.2147e+00            2.6745e+00            2.6714e+00            2.8617e+00            2.6714e+00            2.6745e+00            2.6712e+00            2.6712e+00            
19   6.7420e+00            7.0990e+00            6.2254e+00            5.6284e+00            4.4038e+01            5.6218e+00            5.4678e+00            5.3948e+00            5.5025e+00            
20   1.2135e+01            3.5295e+01            5.9512e+00            1.0509e+01            9.6815e+00            1.2087e+01            3.7241e+00            1.1642e+01            7.3803e+00            
21   4.4032e+00            4.1880e+00            4.2188e+00            4.1692e+00            4.7397e+00            4.1736e+00            4.1693e+00            4.1721e+00            4.1644e+00            
22   2.3057e+00            2.4546e+00            2.2387e+00            2.2858e+00            8.4109e+00            2.2020e+00            2.5163e+00            2.2536e+00            2.2076e+00            
23   1.5652e+01            2.5964e+01            2.1375e+01            9.3742e+00            4.7197e+01            8.7561e+00            3.1937e+01            8.8593e+00            1.2353e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 23: best error so far = 8.756e+00  (target = 1e-08)
  Task 19: best error so far = 5.395e+00  (target = 1e-08)
  Task 21: best error so far = 4.164e+00  (target = 1e-08)
  Task 20: best error so far = 3.724e+00  (target = 1e-08)
  Task 17: best error so far = 3.120e+00  (target = 1e-08)
  Task 18: best error so far = 2.671e+00  (target = 1e-08)
  Task 22: best error so far = 2.202e+00  (target = 1e-08)
  Task 14: best error so far = 2.120e+00  (target = 1e-08)
  ... and 9 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to crush the error on the WORST unsolved
tasks above. It is acceptable — even expected — for the new variant to be
worse than existing variants on already-SOLVED tasks; the adaptive selector
will handle that. Reason explicitly about the priority targets:
1. What property of those WORST unsolved tasks (multimodality, ill-conditioning,
   separability, ruggedness, noise, deceptive local optima, narrow basins,
   non-separable rotation, etc.) is preventing existing operators from reaching
   1e-08? Use the per-task error magnitudes as evidence — errors
   stuck at ~1e+1 vs ~1e-3 vs ~1e-6 imply different failure modes.
2. What specific mechanism in your proposed operator is designed to break
   through that exact obstacle and push the error several orders of magnitude
   lower?
3. Why is this approach fundamentally different from the prior variants —
   especially from whichever variant currently holds the best (but still
   insufficient) error on the priority tasks?

Current implementation:
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to whole population at once."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        # Ensure at least one dimension is from mutant
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials
```

Full algorithm for context:
```python
import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, and intelligent restart mechanism.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(8 * dim, 300)
        self.archive_max = self.np_size
        self.memory_size = 5
        self.p_best_rate = 0.15
        self.restart_threshold = 50
        self.min_pop_size = max(4, dim // 2)

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        population, fitness = self._initialize_population(func, stopping_condition)
        if stopping_condition():
            return self.f_opt, self.x_opt

        self._update_best(population, fitness)
        archive = np.empty((0, self.dim))
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        prev_best_fitness = self.f_opt

        while not stopping_condition():
            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch()

            # Generate mutant vectors
            mutants = self._mutate_batch(population, fitness, archive, f_values, strategy_mask)

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values)

            # Clip to bounds
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                self._safe_update_best(trials, trial_fitness)
                break

            # Handle truncated results
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                f_values = f_values[:len(trial_fitness)]
                cr_values = cr_values[:len(trial_fitness)]

            valid = ~np.isnan(trial_fitness)
            
            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Update parameter memory
            memory_f, memory_cr, memory_idx = self._update_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta
            )

            self._update_best(population, fitness)

            # Detect stagnation
            stagnant = self._detect_stagnation(prev_best_fitness)
            if stagnant:
                population, fitness, archive, memory_f, memory_cr, memory_idx = (
                    self._restart(func, stopping_condition, population, fitness)
                )
                if stopping_condition():
                    break

            prev_best_fitness = self.f_opt
            self.generation += 1

            # Periodically try opposition-based learning
            if self.generation % 25 == 0 and not stopping_condition():
                population, fitness = self._opposition_based_jump(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
                self._update_best(population, fitness)

        return self.f_opt, self.x_opt

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values with generation-adaptive scaling for better convergence."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Adaptive scale factor: starts at 0.1, decreases over generations for finer tuning
        # but never goes below 0.02 to maintain some exploration
        base_scale = 0.1
        generation = getattr(self, 'generation', 0)
        scale_f = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))
        scale_cr = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))

        # Vectorized Cauchy distribution for F with rejection sampling
        mu_f = memory_f[indices]
        f_values = np.zeros(n)
        remaining = np.ones(n, dtype=bool)
        max_attempts = 100
        attempt = 0

        while np.any(remaining) and attempt < max_attempts:
            count = np.sum(remaining)
            # Cauchy samples: tan(pi * (U - 0.5))
            u = np.random.uniform(0, 1, size=count)
            cauchy_samples = np.tan(np.pi * (u - 0.5))
            candidates = mu_f[remaining] + scale_f * cauchy_samples

            # Accept those > 0
            valid = candidates > 0
            idx_remaining = np.where(remaining)[0]
            accepted = idx_remaining[valid]
            f_values[accepted] = np.minimum(candidates[valid], 1.0)
            remaining[accepted] = False
            attempt += 1

        # Fallback for any remaining (shouldn't happen but be safe)
        if np.any(remaining):
            f_values[remaining] = np.clip(mu_f[remaining], 0.01, 1.0)

        # Normal distribution for CR with adaptive scale
        cr_values = np.random.normal(memory_cr[indices], scale_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        # For some fraction of population, use more explorative parameters
        # This helps escape local optima on multimodal functions
        explore_mask = np.random.random(n) < 0.1
        if np.any(explore_mask):
            n_explore = np.sum(explore_mask)
            # Use larger F for exploration
            f_values[explore_mask] = np.clip(
                np.random.uniform(0.5, 1.0, size=n_explore), 0.1, 1.0
            )
            # Use higher CR for exploration (more dimensions from donor)
            cr_values[explore_mask] = np.clip(
                np.random.uniform(0.8, 1.0, size=n_explore), 0.0, 1.0
            )

        return f_values, cr_values

    def _select_strategies_batch(self):
        """Select mutation strategy for each individual. Returns boolean mask."""
        # True = current-to-pbest/1, False = rand-to-pbest/1
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using mixed strategies for the whole population."""
        n = len(population)
        dim = self.dim

        # Sort by fitness for p-best selection
        sorted_idx = np.argsort(fitness)
        p = max(2, int(np.ceil(self.p_best_rate * n)))

        # Select p-best indices for each individual
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Select r1 != i
        r1 = self._random_indices_not_equal(n, np.arange(n))

        # Union of population and archive for r2
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Select r2 from union, r2 != i and r2 != r1
        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        # Strategy 1: current-to-pbest/1 (for strategy_mask == True)
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            mutants[idx1] = (
                population[idx1]
                + f_col[idx1] * (population[pbest_idx[idx1]] - population[idx1])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            )

        # Strategy 2: rand-to-pbest/1 (for strategy_mask == False)
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            mutants[idx2] = (
                population[r_base]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            )

        return mutants

    def _random_indices_not_equal(self, n, exclude_indices):
        """Generate random indices in [0, n) that differ from exclude_indices."""
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to whole population at once."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        # Ensure at least one dimension is from mutant
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds with midpoint reflection."""
        # For out-of-bounds, reflect towards the center
        too_low = trials < self.lb
        too_high = trials > self.ub
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """Greedy selection: replace if trial is better. Track successful params."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        improved = np.zeros(n, dtype=bool)
        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    improved[i] = True
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))
                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    def _update_memory(self, memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta):
        """Update parameter memory using weighted Lehmer mean."""
        if len(success_f) == 0:
            return memory_f, memory_cr, memory_idx

        weights = success_delta / (np.sum(success_delta) + 1e-30)

        # Weighted Lehmer mean for F
        lehmer_f = np.sum(weights * success_f ** 2) / (np.sum(weights * success_f) + 1e-30)
        memory_f[memory_idx] = lehmer_f

        # Weighted arithmetic mean for CR
        mean_cr = np.sum(weights * success_cr)
        memory_cr[memory_idx] = mean_cr

        memory_idx = (memory_idx + 1) % self.memory_size
        return memory_f, memory_cr, memory_idx

    def _detect_stagnation(self, prev_best_fitness):
        """Detect if the algorithm is stagnating."""
        improvement = prev_best_fitness - self.f_opt
        if improvement < 1e-12 * (abs(prev_best_fitness) + 1e-30):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        return self.stagnation_counter >= self.restart_threshold

    def _restart(self, func, stopping_condition, population, fitness):
        """Partial restart: keep best individuals, reinitialize rest with opposition."""
        self.stagnation_counter = 0
        n = len(population)
        keep_count = max(2, n // 5)

        sorted_idx = np.argsort(fitness)
        elite = population[sorted_idx[:keep_count]].copy()
        elite_fit = fitness[sorted_idx[:keep_count]].copy()

        reinit_count = n - keep_count
        # Generate new individuals using quasi-opposition
        new_pop = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

        # Quasi-opposition: reflect around population center
        center = np.mean(elite, axis=0)
        opp_pop = 2.0 * center - new_pop
        opp_pop = np.clip(opp_pop, self.lb, self.ub)

        # Choose random mix of new and opposition
        mix_mask = np.random.random(reinit_count) < 0.5
        combined_new = np.where(mix_mask[:, np.newaxis], opp_pop, new_pop)

        if stopping_condition():
            population = np.vstack([elite, combined_new])
            fitness_new = np.full(reinit_count, np.inf)
            return population, np.concatenate([elite_fit, fitness_new]), np.empty((0, self.dim)), np.full(self.memory_size, 0.5), np.full(self.memory_size, 0.5), 0

        combined_new = self._clip_to_bounds(combined_new)
        new_fitness = func(combined_new)
        if len(new_fitness) < len(combined_new):
            combined_new = combined_new[:len(new_fitness)]

        self._safe_update_best(combined_new, new_fitness)

        population = np.vstack([elite, combined_new[:len(new_fitness)]])
        fitness_all = np.concatenate([elite_fit, new_fitness])

        # Reset memory
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        archive = np.empty((0, self.dim))

        self.np_size = len(population)

        return population, fitness_all, archive, memory_f, memory_cr, memory_idx

    def _opposition_based_jump(self, population, fitness, func, stopping_condition):
        """Apply opposition-based learning to diversify population."""
        center = (self.lb + self.ub) / 2.0
        opp_population = 2.0 * center - population
        opp_population = self._clip_to_bounds(opp_population)

        # Add random perturbation
        noise = np.random.normal(0, 1.0, opp_population.shape)
        opp_population = self._clip_to_bounds(opp_population + noise)

        if stopping_condition():
            return population, fitness

        opp_fitness = func(opp_population)
        if len(opp_fitness) < len(opp_population):
            opp_population = opp_population[:len(opp_fitness)]

        self._safe_update_best(opp_population, opp_fitness)

        # Combine and select best
        n = min(len(population), len(opp_population))
        combined_pop = np.vstack([population[:n], opp_population[:n]])
        combined_fit = np.concatenate([fitness[:n], opp_fitness[:n]])

        sorted_idx = np.argsort(combined_fit)
        best_idx = sorted_idx[:len(population)]
        return combined_pop[best_idx], combined_fit[best_idx]

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance."""
        center = np.mean(population, axis=0)
        diffs = population - center
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _update_best(self, population, fitness):
        """Update global best from population."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fitness = fitness[valid]
        valid_pop = population[valid]
        best_idx = np.argmin(valid_fitness)
        if valid_fitness[best_idx] < self.f_opt:
            self.f_opt = valid_fitness[best_idx]
            self.x_opt = valid_pop[best_idx].copy()

    def _safe_update_best(self, candidates, fitnesses):
        """Safely update best considering possible NaN values."""
        if len(fitnesses) == 0:
            return
        valid = ~np.isnan(fitnesses)
        if not np.any(valid):
            return
        best_idx = np.nanargmin(fitnesses)
        if fitnesses[best_idx] < self.f_opt:
            self.f_opt = fitnesses[best_idx]
            self.x_opt = candidates[best_idx].copy()

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _crossover_batch(self, ...):
    ...
```