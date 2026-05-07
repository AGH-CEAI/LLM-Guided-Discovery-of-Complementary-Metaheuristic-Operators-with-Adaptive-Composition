Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_crossover_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.244193e-07            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
1      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.562342e-07            1.000000e-08            1.000000e-08            1.000000e-08            1.161979e-08            
2      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.817484e-05            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
3      1.340134e-04            1.376762e-04            1.319581e-04            1.363831e-04            1.385594e-04            1.335778e-04            3.861578e-04            1.327079e-04            1.348898e-04            1.356799e-04            1.299483e-04            
4      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            2.517218e-02            1.000000e-08            1.000000e-08            1.000000e-08            1.180525e-02            
5      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
6      5.854684e-08            6.006110e-08            5.682616e-08            5.195492e-08            5.037837e-06            5.559477e-08            4.776228e-07            5.568350e-08            5.507663e-08            5.619032e-08            9.841081e-08            
7      4.256942e-06            3.909615e-06            4.201698e-06            3.967529e-06            4.154081e-06            4.095313e-06            1.586914e-05            4.003840e-06            3.567551e-06            4.091328e-06            4.290091e-06            
8      1.000000e-08            7.441045e-04            7.306981e-04            7.146618e-04            7.831677e-04            1.000000e-08            1.919654e-03            1.000000e-08            1.000000e-08            1.000000e-08            6.445840e-04            
9      2.171307e-04            2.010495e-04            1.878296e-04            1.885233e-04            1.160358e+00            1.834995e-04            5.449253e-04            1.879804e-04            1.806591e-04            1.856552e-04            2.182449e-04            
10     1.000000e-08            6.758410e-02            1.572812e-01            2.686146e-01            7.150591e-01            5.202504e+00            5.455992e+00            2.029816e+01            5.984126e+00            5.701386e+00            2.170636e-04            
11     2.550406e+00            2.764008e+00            2.410104e+00            2.354512e+00            3.839632e+00            1.906208e+00            4.733763e+00            9.801322e+01            3.589884e+00            2.503481e+00            4.147955e-04            
12     4.564517e-06            8.532962e-01            1.702271e+00            3.315819e+00            4.280449e+00            2.421584e+00            3.538593e+01            3.885767e+01            6.815613e+00            1.920278e+00            1.276800e+00            
13     1.586451e+00            1.604887e+00            1.458463e+00            1.032064e+00            1.417842e+00            1.472183e+00            1.989022e-02            3.077055e-01            2.086138e+00            1.549885e+00            1.356999e+00            
14     2.448769e+00            2.429881e+00            2.450191e+00            2.456242e+00            2.414028e+00            2.583503e+00            3.197322e+00            9.351894e+00            2.464152e+00            2.553793e+00            2.485155e+00            
15     1.191335e+00            4.046917e+00            3.791463e+00            1.807249e+00            4.069814e+00            1.224484e+00            2.713269e+00            2.957183e+00            1.836999e+00            1.318179e+00            1.033785e+00            
16     1.487050e+02            2.335368e+02            9.440298e+01            9.082386e+01            1.439698e+02            5.402785e+01            1.801166e-08            1.000000e-08            8.460986e+01            5.273476e+01            4.000000e-08            
17     3.120103e+00            3.203022e+00            3.121871e+00            3.120106e+00            8.020889e+00            3.120103e+00            3.124070e+00            3.120103e+00            4.185815e+00            3.120103e+00            3.120103e+00            
18     2.671167e+00            3.631497e+00            2.924716e+00            2.683348e+00            7.863668e+00            2.676015e+00            2.960944e+00            2.986330e+00            9.378362e+00            2.671167e+00            2.674276e+00            
19     7.967546e+00            7.093984e+00            6.442309e+00            6.099596e+00            9.834976e+00            5.449173e+00            3.687792e+01            1.984346e+02            5.640789e+00            5.812459e+00            5.886196e+00            
20     1.239144e+01            3.483241e+01            3.191452e+01            4.401113e+00            3.699798e+01            8.656778e+00            1.067804e+01            2.803001e+01            1.216693e+01            1.002429e+01            2.924830e+00            
21     4.440833e+00            4.297317e+00            4.205638e+00            4.190935e+00            4.256557e+00            4.178323e+00            4.589265e+00            5.358910e+00            4.184596e+00            4.176692e+00            4.146178e+00            
22     2.408162e+00            2.277980e+00            2.282633e+00            2.253671e+00            2.452598e+00            2.204227e+00            8.823039e+00            1.611064e+01            2.403685e+00            2.246037e+00            2.493685e+00            
23     1.541896e+01            2.905342e+01            2.578715e+01            2.073028e+01            3.080814e+01            9.212771e+00            4.795879e+01            5.043416e+01            7.976582e+00            8.921983e+00            3.439245e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: SKIPPED (trivial)
Task  2: SKIPPED (trivial)
Task  3: variant_10_idea_0.py  (error=1.299483e-04)
Task  4: SKIPPED (trivial)
Task  5: SKIPPED (trivial)
Task  6: SKIPPED (trivial)
Task  7: variant_08_idea_0.py  (error=3.567551e-06)
Task  8: original.py  (error=1.000000e-08)
Task  9: variant_08_idea_0.py  (error=1.806591e-04)
Task 10: original.py  (error=1.000000e-08)
Task 11: variant_10_idea_0.py  (error=4.147955e-04)
Task 12: original.py  (error=4.564517e-06)
Task 13: variant_06_idea_0.py  (error=1.989022e-02)
Task 14: variant_04_idea_0.py  (error=2.414028e+00)
Task 15: variant_10_idea_0.py  (error=1.033785e+00)
Task 16: variant_07_idea_0.py  (error=1.000000e-08)
Task 17: original.py  (error=3.120103e+00)
Task 18: original.py  (error=2.671167e+00)
Task 19: variant_05_idea_0.py  (error=5.449173e+00)
Task 20: variant_10_idea_0.py  (error=2.924830e+00)
Task 21: variant_10_idea_0.py  (error=4.146178e+00)
Task 22: variant_05_idea_0.py  (error=2.204227e+00)
Task 23: variant_08_idea_0.py  (error=7.976582e+00)

WIN COUNTS:
  variant_10_idea_0.py: 5 wins
  original.py: 5 wins
  variant_08_idea_0.py: 3 wins
  variant_05_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins
  variant_04_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_04_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Exponential crossover with directional perturbation for non-separable functions."""
        n, dim = population.shape
        trials = population.copy()

        for i in range(n):
            cr = cr_values[i]
            # Exponential crossover: transfer a contiguous block from mutant
            start = np.random.randint(0, dim)
            L = 0  # length of crossover segment
            while True:
                L += 1
                if L >= dim or np.random.random() >= cr:
                    break

            # Apply contiguous block
            indices = [(start + j) % dim for j in range(L)]
            trials[i, indices] = mutants[i, indices]

        # Directional perturbation: add small step along parent->mutant direction
        # This helps with non-separable rotated functions
        direction = mutants - population
        dir_norms = np.sqrt(np.sum(direction ** 2, axis=1, keepdims=True)) + 1e-30
        direction_normalized = direction / dir_norms

        # Adaptive perturbation magnitude based on generation
        generation = getattr(self, 'generation', 0)
        # Start with larger perturbation, decay over time
        base_scale = 0.05 * (self.ub - self.lb)
        scale = base_scale / (1.0 + generation / 100.0)

        # Apply perturbation to a fraction of the population (30%)
        perturb_mask = np.random.random(n) < 0.3
        if np.any(perturb_mask):
            n_perturb = np.sum(perturb_mask)
            # Random magnitude per individual
            magnitudes = np.random.exponential(scale, size=(n_perturb, 1))
            # Also add a small random orthogonal component for diversity
            random_component = np.random.normal(0, scale * 0.1, size=(n_perturb, dim))

            trials[perturb_mask] = (
                trials[perturb_mask] 
                + magnitudes * direction_normalized[perturb_mask]
                + random_component
            )

        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)

        return trials
```

# --- From variant_05_idea_0.py (2 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-based crossover in rotated coordinate system for rotational invariance."""
        n, dim = population.shape
        cr_matrix = cr_values[:, np.newaxis]

        try:
            # Compute covariance matrix of population
            if n > dim + 1:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Add small regularization for numerical stability
                cov += 1e-12 * np.eye(dim)
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # eigenvectors columns are the principal axes

                # Transform population and mutants into eigenvector space
                pop_rotated = np.dot(population - center, eigenvectors)
                mut_rotated = np.dot(mutants - center, eigenvectors)

                # Binomial crossover in rotated space
                rand_matrix = np.random.random((n, dim))
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(n), j_rand] = True

                # Mix in rotated space
                trial_rotated = np.where(cross_mask, mut_rotated, pop_rotated)

                # Transform back to original space
                trials = np.dot(trial_rotated, eigenvectors.T) + center

                # Fallback: if any trial has NaN/Inf, use standard crossover for those
                bad = np.any(~np.isfinite(trials), axis=1)
                if np.any(bad):
                    rand_matrix2 = np.random.random((np.sum(bad), dim))
                    j_rand2 = np.random.randint(0, dim, size=np.sum(bad))
                    cross_mask2 = rand_matrix2 < cr_matrix[bad]
                    cross_mask2[np.arange(np.sum(bad)), j_rand2] = True
                    trials[bad] = np.where(cross_mask2, mutants[bad], population[bad])

                return trials
            else:
                raise ValueError("Not enough population for covariance")
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to standard binomial crossover
            rand_matrix = np.random.random((n, dim))
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)
            return trials
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Exponential crossover with eigenvector rotation for non-separable problems."""
        n, dim = population.shape

        # Compute covariance-based rotation from population for non-separable alignment
        try:
            if n > dim and dim <= 100:
                centered = population - np.mean(population, axis=0)
                cov = np.cov(centered, rowvar=False)
                # Add small regularization
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Rotate population and mutants into eigenvector space
                pop_rot = population @ eigenvectors
                mut_rot = mutants @ eigenvectors
                use_rotation = True
            else:
                pop_rot = population
                mut_rot = mutants
                use_rotation = False
        except:
            pop_rot = population
            mut_rot = mutants
            use_rotation = False

        trials_rot = pop_rot.copy()

        for i in range(n):
            cr = cr_values[i]

            # 15% chance: full mutant copy for maximum exploration
            if np.random.random() < 0.15:
                trials_rot[i] = mut_rot[i]
                continue

            # Exponential crossover: copies contiguous segments from mutant
            # This respects variable linkage in non-separable problems
            start = np.random.randint(0, dim)
            L = 0
            while L < dim:
                trials_rot[i, (start + L) % dim] = mut_rot[i, (start + L) % dim]
                L += 1
                if np.random.random() >= cr:
                    break

        # Rotate back to original space
        if use_rotation:
            trials = trials_rot @ eigenvectors.T
        else:
            trials = trials_rot

        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)

        return trials
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-based crossover that operates in the rotated coordinate system
        defined by the population's covariance structure, enabling correlated moves
        along natural landscape directions for non-separable functions."""
        n, dim = population.shape
        cr_matrix = cr_values[:, np.newaxis]

        try:
            if dim >= 2 and n >= dim + 1:
                # Compute covariance of population
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Regularize
                cov += 1e-10 * np.eye(dim)
                # Eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Ensure numerical stability
                eigvals = np.maximum(eigvals, 1e-20)

                # Rotate population and mutants into eigenvector space
                pop_rotated = np.dot(population - center, eigvecs)
                mut_rotated = np.dot(mutants - center, eigvecs)

                # Perform crossover in rotated space with dimension-adaptive CR
                # Higher CR for eigenvectors with larger variance (more important directions)
                importance = np.sqrt(eigvals / np.max(eigvals))  # 0 to 1
                # Boost CR for important dimensions, reduce for less important
                adjusted_cr = cr_matrix * (0.3 + 0.7 * importance[np.newaxis, :])
                adjusted_cr = np.clip(adjusted_cr, 0.0, 1.0)

                rand_matrix = np.random.random((n, dim))
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < adjusted_cr
                cross_mask[np.arange(n), j_rand] = True

                # Crossover in rotated space
                trials_rotated = np.where(cross_mask, mut_rotated, pop_rotated)

                # Rotate back to original space
                trials = np.dot(trials_rotated, eigvecs.T) + center
            else:
                raise ValueError("Fall back to standard")

        except (np.linalg.LinAlgError, ValueError):
            # Fallback: standard binomial crossover
            rand_matrix = np.random.random((n, dim))
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)

        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)

        return trials
```

# --- From variant_08_idea_0.py (3 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-rotated crossover for non-separable landscapes."""
        n, dim = population.shape

        # Compute population covariance eigenvectors for rotation
        try:
            if n > dim and dim > 1:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                # Regularize
                cov += 1e-10 * np.eye(dim)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # eigenvectors columns are the principal axes
                use_rotation = True
            else:
                use_rotation = False
        except Exception:
            use_rotation = False

        # Boost CR based on generation to increase exploration when stuck
        generation = getattr(self, 'generation', 0)
        cr_boost = min(0.3, generation / 500.0)  # gradually increase CR
        boosted_cr = np.clip(cr_values + cr_boost, 0.0, 1.0)

        if use_rotation and dim > 1:
            # Rotate population and mutants into eigenvector space
            pop_rotated = np.dot(population - center, eigenvectors)
            mut_rotated = np.dot(mutants - center, eigenvectors)

            # Perform binomial crossover in rotated space
            rand_matrix = np.random.random((n, dim))
            cr_matrix = boosted_cr[:, np.newaxis]

            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True

            # For 20% of population, use exponential crossover in rotated space for contiguous blocks
            exp_mask = np.random.random(n) < 0.2
            if np.any(exp_mask):
                exp_idx = np.where(exp_mask)[0]
                for i in exp_idx:
                    L = np.random.randint(0, dim)
                    cross_mask[i, :] = False
                    j = L
                    while True:
                        cross_mask[i, j % dim] = True
                        j += 1
                        if np.random.random() >= boosted_cr[i] or (j - L) >= dim:
                            break

            trial_rotated = np.where(cross_mask, mut_rotated, pop_rotated)

            # Rotate back to original space
            trials = np.dot(trial_rotated, eigenvectors.T) + center
        else:
            # Fallback: standard binomial with boosted CR
            rand_matrix = np.random.random((n, dim))
            cr_matrix = boosted_cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, size=n)
            cross_mask = rand_matrix < cr_matrix
            cross_mask[np.arange(n), j_rand] = True
            trials = np.where(cross_mask, mutants, population)

        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)

        return trials
```

# --- From variant_10_idea_0.py (5 wins) ---
```python
def _crossover_batch(self, population, mutants, cr_values):
        """Eigenvector-guided crossover in rotated space + segment crossover for non-separable functions."""
        n, dim = population.shape
        cr_matrix = cr_values[:, np.newaxis]

        trials = population.copy()

        # Compute covariance-based rotation matrix from population
        use_rotated = dim >= 2 and n >= dim + 1

        if use_rotated:
            try:
                center = np.mean(population, axis=0)
                centered = population - center
                cov = np.dot(centered.T, centered) / max(n - 1, 1)
                cov += 1e-10 * np.eye(dim)  # regularization
                eigvals, eigvecs = np.linalg.eigh(cov)

                # Rotate population and mutants into eigenvector space
                pop_rot = np.dot(population - center, eigvecs)
                mut_rot = np.dot(mutants - center, eigvecs)

                # Apply crossover in rotated space with segment-based approach
                for i in range(n):
                    cr_i = cr_values[i]
                    # Segment crossover: transfer contiguous blocks in rotated space
                    # Block size adapts with CR
                    avg_block_size = max(1, int(cr_i * dim * 0.5))

                    trial_rot = pop_rot[i].copy()
                    j_start = np.random.randint(0, dim)

                    # Geometric distribution for segment length
                    seg_len = min(np.random.geometric(p=max(0.1, 1.0 - cr_i + 0.01)), dim)

                    # Transfer segment from mutant in rotated space
                    indices = [(j_start + k) % dim for k in range(seg_len)]
                    trial_rot[indices] = mut_rot[i][indices]

                    # Also do standard binomial on remaining dims in rotated space
                    remaining = np.ones(dim, dtype=bool)
                    remaining[indices] = False
                    rand_vals = np.random.random(dim)
                    binomial_mask = (rand_vals < cr_i * 0.5) & remaining
                    trial_rot[binomial_mask] = mut_rot[i][binomial_mask]

                    # Rotate back to original space
                    trials[i] = np.dot(trial_rot, eigvecs.T) + center

            except (np.linalg.LinAlgError, ValueError):
                # Fallback to standard binomial crossover
                rand_matrix = np.random.random((n, dim))
                j_rand = np.random.randint(0, dim, size=n)
                cross_mask = rand_matrix < cr_matrix
                cross_mask[np.arange(n), j_rand] = True
                trials = np.where(cross_mask, mutants, population)
        else:
            # For very low dimensions, use exponential crossover
            for i in range(n):
                L = 0
                j = np.random.randint(0, dim)
                while np.random.random() < cr_values[i] and L < dim:
                    trials[i, j] = mutants[i, j]
                    j = (j + 1) % dim
                    L += 1
                if L == 0:
                    trials[i, j] = mutants[i, j]

        return trials
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