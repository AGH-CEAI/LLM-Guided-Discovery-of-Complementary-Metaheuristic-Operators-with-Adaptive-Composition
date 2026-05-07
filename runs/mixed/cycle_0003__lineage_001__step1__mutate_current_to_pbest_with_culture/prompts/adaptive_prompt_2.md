Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_mutate_current_to_pbest_with_culture` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    
1      3.448701e-01            3.489321e-01            3.562464e-01            3.792455e-01            3.475927e-01            3.572753e-01            3.556975e-01            3.356187e-01            3.831861e-01            3.443186e-01            -inf                    
2      1.000000e-08            1.474331e-05            1.150487e+05            6.650264e+05            1.157203e+04            1.719032e-01            1.841632e+02            1.153679e-03            5.091437e-06            1.156083e-07            -inf                    
3      1.673241e-08            1.932074e-07            4.932537e-04            2.500423e-03            9.285332e-05            2.046827e-04            8.108476e-05            4.530833e-08            2.482747e-03            5.308629e-08            -inf                    
4      1.238806e+00            1.316508e+00            1.809098e+00            1.812696e+00            1.685660e+00            1.453813e+00            1.469318e+00            1.296778e+00            2.013365e+00            1.294940e+00            -inf                    
5      1.286097e+00            1.470099e+00            1.742091e+00            1.953664e+00            1.655784e+00            1.690166e+00            1.534153e+00            1.404284e+00            2.075100e+00            1.362667e+00            -inf                    
6      2.151104e-08            6.689861e-06            1.679051e+03            1.504102e+04            1.131083e+02            5.096212e-04            2.645841e-02            1.196522e-01            3.855430e+03            2.120199e-06            -inf                    
7      2.107749e-08            8.831552e-06            1.494692e+03            1.729831e+04            8.844645e+01            4.910368e-04            2.451057e-02            2.798291e-02            3.749862e+03            2.034101e-06            -inf                    
8      7.831891e+02            1.074040e+03            1.109554e+03            1.414238e+03            1.343125e+03            1.370215e+03            1.398892e+03            1.662778e+03            1.701317e+03            1.248396e+03            -inf                    
9      3.828678e-07            6.261579e-05            1.738523e+03            1.797912e+04            6.417315e+01            1.327561e-03            6.926920e-02            1.603079e+00            6.361977e+03            1.798904e-05            -inf                    
10     6.090034e-06            2.641191e-04            2.243401e+03            2.368239e+04            1.596475e+02            1.380756e-01            2.751293e+00            5.460234e+00            8.458855e+03            3.886697e-04            -inf                    
11     4.929471e-06            2.500122e-04            2.292984e+03            1.994775e+04            1.251962e+02            5.988222e-02            2.096292e+00            1.821626e+01            1.089511e+04            1.166956e-03            -inf                    
12     1.541886e+03            1.359051e+03            1.798219e+03            1.912161e+03            1.677522e+03            2.100725e+03            1.840789e+03            1.718107e+03            2.087007e+03            1.883014e+03            -inf                    
13     1.467194e+01            2.092834e+02            4.978726e+03            1.510769e+04            1.254850e+03            1.469123e+03            2.330022e+03            3.701612e+03            1.080927e+04            1.016916e+02            -inf                    
14     5.874266e+00            6.095677e+00            6.133060e+00            6.207029e+00            6.048633e+00            6.166407e+00            6.171560e+00            6.056674e+00            6.300793e+00            6.100966e+00            -inf                    
15     5.537009e+02            5.296000e+02            1.333333e-08            5.537009e+02            2.000000e-08            4.000000e-08            1.000000e-08            2.000000e-08            4.000000e-08            1.000000e-08            -inf                    
16     6.001001e+02            6.001008e+02            6.677095e+02            1.096277e-02            6.469808e+02            6.010893e+02            6.857394e-05            6.282371e-07            3.653805e-01            5.734164e+02            -inf                    
17     1.044370e-03            6.914896e-06            2.450274e+03            1.687099e+04            1.960259e+02            5.296001e+02            1.576344e-01            5.656325e+02            7.576772e+03            2.332236e-05            -inf                    
18     8.419155e+01            3.411745e-02            2.815999e+03            1.486217e+04            3.584755e+02            1.709702e+01            6.967636e+02            1.588104e+03            9.586440e+03            2.292827e-02            -inf                    
19     5.913258e-03            3.032289e-02            6.583286e+00            1.128088e+01            3.338967e+00            1.751839e-01            5.754620e-01            1.904056e+00            8.694610e+00            4.786504e-02            -inf                    
20     5.000000e+00            5.000007e+00            4.183719e+01            8.464015e+01            5.000012e+00            5.000362e+00            5.217847e+00            5.077880e+00            5.619420e+01            5.000000e+00            -inf                    
21     5.000023e+01            5.001098e+01            1.882566e+03            2.983273e+03            6.317173e+02            5.841133e+01            2.096627e+02            4.921541e+02            1.733555e+03            5.049010e+01            -inf                    
22     2.038996e-04            2.524789e-03            1.825396e+01            4.298703e+01            5.696309e+00            1.030948e-01            1.234351e+00            3.305183e+00            2.990979e+01            1.230309e-02            -inf                    
23     1.883612e+00            7.254552e+00            8.850186e+01            1.614774e+02            4.945012e+01            2.386619e+01            3.501622e+01            4.592224e+01            1.185470e+02            1.323389e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_07_idea_0.py  (error=3.356187e-01)
Task  2: original.py  (error=1.000000e-08)
Task  3: original.py  (error=1.673241e-08)
Task  4: original.py  (error=1.238806e+00)
Task  5: original.py  (error=1.286097e+00)
Task  6: original.py  (error=2.151104e-08)
Task  7: original.py  (error=2.107749e-08)
Task  8: original.py  (error=7.831891e+02)
Task  9: original.py  (error=3.828678e-07)
Task 10: original.py  (error=6.090034e-06)
Task 11: original.py  (error=4.929471e-06)
Task 12: variant_01_idea_0.py  (error=1.359051e+03)
Task 13: original.py  (error=1.467194e+01)
Task 14: original.py  (error=5.874266e+00)
Task 15: SKIPPED (trivial)
Task 16: variant_07_idea_0.py  (error=6.282371e-07)
Task 17: variant_01_idea_0.py  (error=6.914896e-06)
Task 18: variant_09_idea_0.py  (error=2.292827e-02)
Task 19: original.py  (error=5.913258e-03)
Task 20: original.py  (error=5.000000e+00)
Task 21: original.py  (error=5.000023e+01)
Task 22: original.py  (error=2.038996e-04)
Task 23: original.py  (error=1.883612e+00)

WIN COUNTS:
  original.py: 17 wins
  variant_07_idea_0.py: 2 wins
  variant_01_idea_0.py: 2 wins
  variant_09_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
        # Initialize archive if needed
        if not hasattr(self, 'archive') or self.archive is None:
            self.archive = np.zeros((0, self.dim))

        # Select p-best individuals
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        p_best_indices = sorted_idx[:n_best]
        p_best_idx = p_best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        # Generate random indices r1, r2 (different from current index i)
        r1 = np.random.randint(0, self.NP, size=self.NP)
        r2 = np.zeros(self.NP, dtype=int)

        for i in range(self.NP):
            while True:
                r2[i] = np.random.randint(0, self.NP)
                if r2[i] != i and r2[i] != r1[i]:
                    break

        # JADE mutation: current-to-pbest/1 with archive
        # Use archive members if available, otherwise only current population
        archive_size = len(self.archive)
        total_size = self.NP + archive_size

        if archive_size > 0:
            # Sample from combined population + archive
            all_indices = np.concatenate([np.arange(self.NP), self.archive])
            r3_indices = np.random.randint(0, total_size, size=self.NP)
            r3 = np.where(r3_indices < self.NP, r3_indices, -(r3_indices - self.NP + 1))

            # Handle cases where r3 == current index
            for i in range(self.NP):
                if r3[i] == i:
                    if r3[i] < self.NP - 1:
                        r3[i] += 1
                    else:
                        r3[i] = 0
            r3_pop = np.where(r3 >= 0, population[r3], self.archive[-(r3 + 1)])
        else:
            # No archive: use population only with careful index handling
            r3 = np.random.randint(0, self.NP, size=self.NP)
            for i in range(self.NP):
                while r3[i] == i or r3[i] == r1[i]:
                    r3[i] = (r3[i] + 1) % self.NP
            r3_pop = population[r3]

        # Apply JADE mutation: x_i + F_i * (pbest - x_i) + F_i * (r3 - r1)
        # Use per-individual F with bounded random component
        F_i = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)

        donors = population + F_i[:, np.newaxis] * (p_best - population) + F_i[:, np.newaxis] * (r3_pop - population[r1])

        # Clip to bounds for numerical stability
        donors = np.clip(donors, self.lower, self.upper)

        # Update archive with rejected solutions (called from select after we know rejections)
        # Store for later use - actual archive update happens in survivor selection
        self._pending_archive_additions = None

        return donors
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]

        # Use GLOBAL best instead of random pbest from top 10%
        global_best_idx = best_indices[0]
        global_best = population[global_best_idx]

        # Generate unique random indices for each target individual
        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None], 
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1], 
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        # Adaptive scaling: increase F when stagnant to escape local optima
        F_adapted = self.F
        if self.stagnation_count > 10:
            F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count - 10)), 2.0)

        # Core mutation: current-to-global-best/1 (stronger than current-to-pbest)
        base = population + F_adapted * (global_best - population) + F_adapted * (population[r1] - population[r2])

        # Escalating random perturbation when stagnant (NEW: adds diversity on demand)
        if self.stagnation_count > 5:
            perturbation_scale = min(0.3 * np.log1p(self.stagnation_count), 2.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            base = base + random_perturbation

        # Cultural memory blending (same as before)
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

# --- From variant_09_idea_0.py (1 wins) ---
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

        if np.isfinite(self.best_fitness) and self.generation > 5:
            current_best = np.min(fitness)
            improvement_ratio = (current_best - self.best_fitness) / (abs(self.best_fitness) + 1e-30)

            # Adaptive F: increase when stuck far from optimum
            F_escape = np.clip(self.F * (1.0 + max(0, -improvement_ratio) * 0.5), 0.5, 2.0)

            # Direction weight: favor historical best when population is stuck
            w_hist = np.clip(0.7 + max(0, -improvement_ratio) * 0.2, 0.5, 0.9)
            w_curr = 1.0 - w_hist

            # Compute cultural centroid for direction guidance
            if np.any(self.cultural_weights > 0):
                weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
                cultural_centroid = np.dot(weights_norm, self.cultural_memory)
                # Blend historical best with cultural knowledge
                target = w_hist * self.best_solution + w_curr * p_best + 0.1 * cultural_centroid
            else:
                target = w_hist * self.best_solution + w_curr * p_best

            donors = population + F_escape * (target - population) + self.F * (population[r1] - population[r2])
        else:
            donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        return donors
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