Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_mutate_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            2.565424e+00            1.000000e-08            1.000000e-08            2.264524e-03            -inf                    1.000000e-08            1.000000e-08            1.262242e-04            6.619121e+00            
1      1.000000e-08            1.000000e-08            1.941636e+01            3.104110e-05            5.591641e-08            2.356218e+01            -inf                    2.552892e-08            7.679642e-07            2.857350e-03            8.254177e+00            
2      1.000000e-08            1.000000e-08            4.242665e+01            1.000000e-08            1.000000e-08            4.036598e-03            -inf                    1.000000e-08            1.913687e-06            1.359709e-02            2.753441e+01            
3      1.370373e-04            1.354823e-04            7.040359e+01            1.377810e+00            1.164935e-01            6.554802e+01            -inf                    1.231021e-01            1.532367e-01            1.362509e+00            4.022435e+01            
4      1.000000e-08            1.000000e-08            3.151568e+00            6.526633e-02            1.230918e-02            2.397189e-01            -inf                    1.000000e-08            1.841431e-02            2.624184e-01            2.141552e+00            
5      1.000000e-08            1.000000e-08            5.492096e+06            2.202065e-05            1.000000e-08            1.956177e+06            -inf                    1.000000e-08            1.000000e-08            4.561940e-02            1.669751e+05            
6      5.635165e-08            5.980027e-08            9.660575e+02            2.336997e+01            1.510581e+01            3.668965e+01            -inf                    1.223016e+01            2.605419e+01            1.685890e+01            7.246056e+02            
7      4.074027e-06            4.447608e-06            6.547267e+01            5.955765e-02            5.022400e-05            6.291369e+01            -inf                    6.310055e-05            8.152594e-05            3.838095e-01            3.229582e+01            
8      1.000000e-08            7.761487e-04            1.772504e+01            1.641542e-01            8.669074e-04            1.994500e+00            -inf                    9.168550e-04            8.314627e-03            6.623449e-01            1.016917e+01            
9      1.985128e-04            1.816689e+00            1.104091e+02            6.291461e+00            6.328980e+00            7.696183e+00            -inf                    6.949467e+00            7.044399e+00            6.672719e+00            9.211602e+01            
10     1.000000e-08            1.000000e-08            9.039532e+00            3.165832e-06            6.165420e+00            4.981426e+00            -inf                    6.638996e+00            6.778417e+00            5.131014e-01            1.092654e+02            
11     2.022464e+00            1.838029e+00            3.380842e+02            1.118424e+01            1.877084e+01            1.171302e+02            -inf                    6.271114e+01            2.178684e+01            4.023087e+00            8.025829e+02            
12     4.443249e-06            5.537112e-06            9.312972e+00            1.036107e-03            8.890710e-01            1.544339e+01            -inf                    8.113236e-01            1.269161e+00            1.532766e+00            8.517763e+01            
13     1.302050e+00            1.451455e+00            1.908188e+01            2.124200e+00            1.025997e+01            1.418703e+01            -inf                    1.800447e+01            1.228636e+01            2.216258e+00            4.495673e+01            
14     2.409909e+00            2.439057e+00            1.230247e+01            2.698386e+00            3.728503e+00            3.355518e+00            -inf                    3.250297e+00            3.900460e+00            2.742312e+00            5.775222e+00            
15     1.094612e+00            1.133590e+00            3.334182e+00            2.442675e+00            2.503457e+00            2.980918e+00            -inf                    2.821522e+00            2.304063e+00            1.491451e+00            3.944157e+00            
16     1.093736e+02            1.554542e+02            2.925499e+03            3.978556e+02            5.305474e+02            6.260896e+02            -inf                    6.095788e+02            5.401954e+02            1.783972e+02            1.381029e+04            
17     3.120103e+00            3.120103e+00            1.540307e+04            5.223763e+02            8.185004e+02            2.371813e+04            -inf                    7.060577e+03            4.692486e+03            2.416123e+01            2.190578e+05            
18     2.671167e+00            2.671942e+00            3.344698e+01            2.298260e+01            2.297272e+01            3.627655e+01            -inf                    2.822492e+01            3.024292e+01            6.528163e+00            7.779713e+01            
19     6.372955e+00            6.623622e+00            2.199873e+02            3.013250e+01            3.169828e+01            3.657474e+01            -inf                    1.051845e+02            3.193137e+01            1.395512e+01            2.437189e+02            
20     1.393510e+01            1.588885e+01            3.406233e+01            1.917205e+01            1.953287e+01            2.054781e+01            -inf                    2.103136e+01            2.171055e+01            1.193396e+01            3.402283e+01            
21     4.364589e+00            4.257936e+00            5.527011e+00            4.478331e+00            4.460829e+00            4.588849e+00            -inf                    4.496487e+00            4.589804e+00            4.538361e+00            5.857267e+00            
22     2.293471e+00            2.263163e+00            1.359868e+01            9.754950e+00            1.109815e+01            1.194776e+01            -inf                    1.109913e+01            1.062623e+01            3.104022e+00            1.689091e+01            
23     1.705691e+01            1.553724e+01            7.286633e+01            2.341998e+01            2.564906e+01            3.475591e+01            -inf                    3.125505e+01            2.555024e+01            2.480814e+01            6.522045e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: original.py  (error=1.000000e-08)
Task  2: original.py  (error=1.000000e-08)
Task  3: variant_01_idea_0.py  (error=1.354823e-04)
Task  4: original.py  (error=1.000000e-08)
Task  5: original.py  (error=1.000000e-08)
Task  6: original.py  (error=5.635165e-08)
Task  7: original.py  (error=4.074027e-06)
Task  8: original.py  (error=1.000000e-08)
Task  9: original.py  (error=1.985128e-04)
Task 10: original.py  (error=1.000000e-08)
Task 11: variant_01_idea_0.py  (error=1.838029e+00)
Task 12: original.py  (error=4.443249e-06)
Task 13: original.py  (error=1.302050e+00)
Task 14: original.py  (error=2.409909e+00)
Task 15: original.py  (error=1.094612e+00)
Task 16: original.py  (error=1.093736e+02)
Task 17: original.py  (error=3.120103e+00)
Task 18: original.py  (error=2.671167e+00)
Task 19: original.py  (error=6.372955e+00)
Task 20: variant_09_idea_0.py  (error=1.193396e+01)
Task 21: variant_01_idea_0.py  (error=4.257936e+00)
Task 22: variant_01_idea_0.py  (error=2.263163e+00)
Task 23: variant_01_idea_0.py  (error=1.553724e+01)

WIN COUNTS:
  original.py: 17 wins
  variant_01_idea_0.py: 5 wins
  variant_09_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (5 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using rank-weighted current-to-pbest/1 with adaptive p."""
        n = len(population)
        dim = self.dim

        # Sort by fitness for p-best selection and ranking
        sorted_idx = np.argsort(fitness)

        # Rank-based F scaling: worse individuals use larger F values
        ranks = np.empty(n, dtype=int)
        ranks[sorted_idx] = np.arange(n)  # rank 0 = best, rank n-1 = worst
        rank_ratio = ranks / max(n - 1, 1)  # 0 to 1

        # Scale F: best individuals use smaller F (exploitation), worst use larger (exploration)
        f_scaled = f_values * (0.5 + 0.5 * rank_ratio)
        f_scaled = np.clip(f_scaled, 0.01, 1.0)

        # Adaptive p-best rate per individual: top individuals use smaller p (more greedy)
        # bottom individuals use larger p (more diverse targets)
        generation = getattr(self, 'generation', 0)
        progress = min(1.0, generation / 500.0)
        p_min = max(2, int(np.ceil(0.05 * n)))
        p_max = max(2, int(np.ceil(0.25 * n)))

        # Per-individual p values
        p_values = np.clip(
            (p_min + (p_max - p_min) * rank_ratio * (1.0 - 0.5 * progress)).astype(int),
            2, n
        )

        # Select p-best indices for each individual with their own p
        pbest_idx = np.array([sorted_idx[np.random.randint(0, p_values[i])] for i in range(n)])

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
            attempts = 0
            while attempts < 100:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break
                attempts += 1
            else:
                r2[i] = np.random.randint(0, union_size)

        f_col = f_scaled[:, np.newaxis]

        # Weighted current-to-pbest/1 for all individuals
        # v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2_union)
        mutants = (
            population
            + f_col * (population[pbest_idx] - population)
            + f_col * (population[r1] - union[r2])
        )

        # For strategy_mask == False, add a second difference vector for more exploration
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r3 = self._random_indices_not_equal(n, np.arange(n))[idx2]
            r4 = self._random_indices_not_equal(n, r1)[idx2]
            # Add weighted second difference vector (smaller weight)
            weight2 = 0.5 * f_col[idx2]
            mutants[idx2] += weight2 * (population[r3] - population[r4])

        return mutants
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        n = len(population)
        dim = self.dim
        f_col = f_values[:, np.newaxis]

        sorted_idx = np.argsort(fitness)

        # Compute simplified covariance direction from top individuals
        n_top = max(3, n // 4)
        top_pop = population[sorted_idx[:n_top]]
        center = np.mean(top_pop, axis=0)

        # Compute covariance eigenvectors for rotation
        try:
            if dim <= 100 and n_top >= 3:
                diffs = top_pop - center
                cov = np.dot(diffs.T, diffs) / max(1, n_top - 1) + 1e-10 * np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-20)
                # Normalize eigenvalues for scaling
                sqrt_eigvals = np.sqrt(eigvals / np.max(eigvals))
            else:
                eigvecs = np.eye(dim)
                sqrt_eigvals = np.ones(dim)
        except:
            eigvecs = np.eye(dim)
            sqrt_eigvals = np.ones(dim)

        # Multiple donor strategy: pick 3 different ranked donors per individual
        p1 = max(2, int(0.1 * n))
        p2 = max(3, int(0.3 * n))
        p3 = max(4, int(0.6 * n))

        donor1_idx = sorted_idx[np.random.randint(0, p1, size=n)]
        donor2_idx = sorted_idx[np.random.randint(0, p2, size=n)]

        r1 = self._random_indices_not_equal(n, np.arange(n))

        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            for _ in range(100):
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        mutants = np.empty((n, dim))

        # Strategy 1: Eigenvector-guided current-to-multi-pbest
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            # Weighted combination of two donors
            w1, w2 = 0.7, 0.3
            direction = (w1 * (population[donor1_idx[idx1]] - population[idx1]) +
                         w2 * (population[donor2_idx[idx1]] - population[idx1]))
            diff_vec = population[r1[idx1]] - union[r2[idx1]]
            # Eigenvector-guided noise
            noise = np.random.normal(0, 1, (len(idx1), dim))
            rotated_noise = noise @ eigvecs.T * sqrt_eigvals[np.newaxis, :]
            rotated_noise = rotated_noise @ eigvecs
            noise_scale = 0.1 * f_col[idx1]

            mutants[idx1] = (population[idx1] + f_col[idx1] * direction +
                             f_col[idx1] * diff_vec + noise_scale * rotated_noise)

        # Strategy 2: rand/2 with eigenvector perturbation for exploration
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            r3 = self._random_indices_not_equal(n, np.arange(n))[idx2]

            mutants[idx2] = (population[r_base] +
                             f_col[idx2] * (population[donor1_idx[idx2]] - population[r3]) +
                             f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]]))

        return mutants
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