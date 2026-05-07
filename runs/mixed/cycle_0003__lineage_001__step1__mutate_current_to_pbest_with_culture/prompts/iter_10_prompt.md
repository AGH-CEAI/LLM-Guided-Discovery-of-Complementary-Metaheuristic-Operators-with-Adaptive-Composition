This is iteration 10 of 10. Propose ONE replacement implementation for `_mutate_current_to_pbest_with_culture`.

Requirements:
- Keep the EXACT function signature: `def _mutate_current_to_pbest_with_culture(self, population, fitness):`
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

TASK COVERAGE SUMMARY: 3 SOLVED (<= 1e-08), 21 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 12: *** UNSOLVED *** — best=1.359e+03 by variant_01_idea_0.py      (target=1e-08, ~+11.1 decades above target)
  Task  8: *** UNSOLVED *** — best=7.832e+02 by original.py               (target=1e-08, ~+10.9 decades above target)
  Task 21: *** UNSOLVED *** — best=5.000e+01 by original.py               (target=1e-08, ~+9.7 decades above target)
  Task 13: *** UNSOLVED *** — best=1.467e+01 by original.py               (target=1e-08, ~+9.2 decades above target)
  Task 14: *** UNSOLVED *** — best=5.874e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 20: *** UNSOLVED *** — best=5.000e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task 23: *** UNSOLVED *** — best=1.884e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task  5: *** UNSOLVED *** — best=1.286e+00 by original.py               (target=1e-08, ~+8.1 decades above target)
  Task  4: *** UNSOLVED *** — best=1.239e+00 by original.py               (target=1e-08, ~+8.1 decades above target)
  Task  1: *** UNSOLVED *** — best=3.356e-01 by variant_07_idea_0.py      (target=1e-08, ~+7.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.293e-02 by variant_09_idea_0.py      (target=1e-08, ~+6.4 decades above target)
  Task 19: *** UNSOLVED *** — best=5.913e-03 by original.py               (target=1e-08, ~+5.8 decades above target)
  Task 22: *** UNSOLVED *** — best=2.039e-04 by original.py               (target=1e-08, ~+4.3 decades above target)
  Task 17: *** UNSOLVED *** — best=6.915e-06 by variant_01_idea_0.py      (target=1e-08, ~+2.8 decades above target)
  Task 10: *** UNSOLVED *** — best=6.090e-06 by original.py               (target=1e-08, ~+2.8 decades above target)
  Task 11: *** UNSOLVED *** — best=4.929e-06 by original.py               (target=1e-08, ~+2.7 decades above target)
  Task 16: *** UNSOLVED *** — best=6.282e-07 by variant_07_idea_0.py      (target=1e-08, ~+1.8 decades above target)
  Task  9: *** UNSOLVED *** — best=3.829e-07 by original.py               (target=1e-08, ~+1.6 decades above target)
  Task  6: *** UNSOLVED *** — best=2.151e-08 by original.py               (target=1e-08, ~+0.3 decades above target)
  Task  7: *** UNSOLVED *** — best=2.108e-08 by original.py               (target=1e-08, ~+0.3 decades above target)
  Task  3: *** UNSOLVED *** — best=1.673e-08 by original.py               (target=1e-08, ~+0.2 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  0: SOLVED  — best=1.000e-08 by original.py                   
  Task  2: SOLVED  — best=1.000e-08 by original.py                   
  Task 15: SOLVED  — best=1.000e-08 by variant_06_idea_0.py          

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  variant_07_idea_0.py  variant_08_idea_0.py  variant_09_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
1    3.4487e-01            3.5625e-01            3.7925e-01            3.4759e-01            3.5728e-01            3.5570e-01            3.3562e-01            3.8319e-01            3.4432e-01            
2    1.0000e-08            1.1505e+05            6.6503e+05            1.1572e+04            1.7190e-01            1.8416e+02            1.1537e-03            5.0914e-06            1.1561e-07            
3    1.6732e-08            4.9325e-04            2.5004e-03            9.2853e-05            2.0468e-04            8.1085e-05            4.5308e-08            2.4827e-03            5.3086e-08            
4    1.2388e+00            1.8091e+00            1.8127e+00            1.6857e+00            1.4538e+00            1.4693e+00            1.2968e+00            2.0134e+00            1.2949e+00            
5    1.2861e+00            1.7421e+00            1.9537e+00            1.6558e+00            1.6902e+00            1.5342e+00            1.4043e+00            2.0751e+00            1.3627e+00            
6    2.1511e-08            1.6791e+03            1.5041e+04            1.1311e+02            5.0962e-04            2.6458e-02            1.1965e-01            3.8554e+03            2.1202e-06            
7    2.1077e-08            1.4947e+03            1.7298e+04            8.8446e+01            4.9104e-04            2.4511e-02            2.7983e-02            3.7499e+03            2.0341e-06            
8    7.8319e+02            1.1096e+03            1.4142e+03            1.3431e+03            1.3702e+03            1.3989e+03            1.6628e+03            1.7013e+03            1.2484e+03            
9    3.8287e-07            1.7385e+03            1.7979e+04            6.4173e+01            1.3276e-03            6.9269e-02            1.6031e+00            6.3620e+03            1.7989e-05            
10   6.0900e-06            2.2434e+03            2.3682e+04            1.5965e+02            1.3808e-01            2.7513e+00            5.4602e+00            8.4589e+03            3.8867e-04            
11   4.9295e-06            2.2930e+03            1.9948e+04            1.2520e+02            5.9882e-02            2.0963e+00            1.8216e+01            1.0895e+04            1.1670e-03            
12   1.5419e+03            1.7982e+03            1.9122e+03            1.6775e+03            2.1007e+03            1.8408e+03            1.7181e+03            2.0870e+03            1.8830e+03            
13   1.4672e+01            4.9787e+03            1.5108e+04            1.2548e+03            1.4691e+03            2.3300e+03            3.7016e+03            1.0809e+04            1.0169e+02            
14   5.8743e+00            6.1331e+00            6.2070e+00            6.0486e+00            6.1664e+00            6.1716e+00            6.0567e+00            6.3008e+00            6.1010e+00            
15   5.5370e+02            1.3333e-08            5.5370e+02            2.0000e-08            4.0000e-08            1.0000e-08            2.0000e-08            4.0000e-08            1.0000e-08            
16   6.0010e+02            6.6771e+02            1.0963e-02            6.4698e+02            6.0109e+02            6.8574e-05            6.2824e-07            3.6538e-01            5.7342e+02            
17   1.0444e-03            2.4503e+03            1.6871e+04            1.9603e+02            5.2960e+02            1.5763e-01            5.6563e+02            7.5768e+03            2.3322e-05            
18   8.4192e+01            2.8160e+03            1.4862e+04            3.5848e+02            1.7097e+01            6.9676e+02            1.5881e+03            9.5864e+03            2.2928e-02            
19   5.9133e-03            6.5833e+00            1.1281e+01            3.3390e+00            1.7518e-01            5.7546e-01            1.9041e+00            8.6946e+00            4.7865e-02            
20   5.0000e+00            4.1837e+01            8.4640e+01            5.0000e+00            5.0004e+00            5.2178e+00            5.0779e+00            5.6194e+01            5.0000e+00            
21   5.0000e+01            1.8826e+03            2.9833e+03            6.3172e+02            5.8411e+01            2.0966e+02            4.9215e+02            1.7336e+03            5.0490e+01            
22   2.0390e-04            1.8254e+01            4.2987e+01            5.6963e+00            1.0309e-01            1.2344e+00            3.3052e+00            2.9910e+01            1.2303e-02            
23   1.8836e+00            8.8502e+01            1.6148e+02            4.9450e+01            2.3866e+01            3.5016e+01            4.5922e+01            1.1855e+02            1.3234e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 12: best error so far = 1.359e+03  (target = 1e-08)
  Task  8: best error so far = 7.832e+02  (target = 1e-08)
  Task 21: best error so far = 5.000e+01  (target = 1e-08)
  Task 13: best error so far = 1.467e+01  (target = 1e-08)
  Task 14: best error so far = 5.874e+00  (target = 1e-08)
  Task 20: best error so far = 5.000e+00  (target = 1e-08)
  Task 23: best error so far = 1.884e+00  (target = 1e-08)
  Task  5: best error so far = 1.286e+00  (target = 1e-08)
  ... and 13 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

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
def _mutate_current_to_pbest_with_culture(self, population, fitness):
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
            weights_culture = np.exp(-similarity / (np.std(similarity) + 1e-10))
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return donors
```

Full algorithm for context:
```python
import numpy as np


class CulturalMemeticDE:
    """
    Hybrid Differential Evolution with Cultural Memory and Adaptive Local Search.
    
    Key innovations vs previous algorithms:
    - Cultural memory stores successful mutation patterns, guiding search in similar regions
    - Adaptive local search (pattern search) refines best individuals periodically
    - Per-dimension adaptation using fitness landscape statistics
    - Diversity-triggered reinitialization with elitist preservation
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

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = population[best_idx].copy()

        while not stopping_condition():
            mutants = self._mutate_current_to_pbest_with_culture(population, fitness)
            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)

            best_idx = np.argmin(fitness)
            current_best = fitness[best_idx]
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

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
            weights_culture = np.exp(-similarity / (np.std(similarity) + 1e-10))
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return donors

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
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:self.NP // 2]
        improvements = fitness[top_half] - fitness[top_half]

        F_candidates = np.random.uniform(0.4, 1.0, size=5)
        self.F = np.mean(F_candidates)
        self.F = np.clip(self.F, 0.3, 1.5)

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = np.mean(CR_candidates)
        self.CR = np.clip(self.CR, 0.1, 0.95)

        dim_adapt = np.random.rand() < 0.2
        if dim_adapt:
            self.p_best_rate = np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3)

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
            best_local_fitness = fit_candidates[best_local_idx]

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
        return np.mean(distances)

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
        population[1:1+n_random] = new_part
        fitness[1:1+n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _mutate_current_to_pbest_with_culture(self, ...):
    ...
```