Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_sample_population_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.436852e-07            8.431136e-01            1.320887e-07            1.410947e-07            1.426759e-07            1.391784e-07            8.415071e-08            2.078277e-07            1.442386e-07            1.432205e-07            4.081205e+01            
1      2.948768e-07            1.369634e+00            2.835480e-07            2.946653e-07            2.846808e-07            2.398584e-07            2.364988e-07            2.962887e-07            2.966641e-07            3.084844e-07            7.744268e+01            
2      2.401976e-05            2.971030e+00            2.383286e-05            2.496352e-05            2.431110e-05            2.200247e-05            2.096987e-05            2.513761e-05            2.518552e-05            2.603622e-05            3.648559e+02            
3      4.134517e-04            1.927253e+00            4.032146e-04            4.126481e-04            4.117477e-04            3.527089e-04            3.536792e-04            4.144390e-04            4.187926e-04            4.197381e-04            1.872403e+02            
4      2.466914e-02            1.289679e+00            2.485074e-02            2.497216e-02            2.482064e-02            2.372320e-02            2.356971e-02            2.493390e-02            2.498479e-02            2.533361e-02            4.690637e+00            
5      1.000000e-08            6.073733e-01            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            3.346912e+08            
6      5.237108e-07            4.238892e+02            5.284518e-07            5.347644e-07            5.227654e-07            4.152292e-07            4.213348e-07            5.398707e-07            5.585814e-07            5.682396e-07            3.460559e+03            
7      1.725651e-05            2.592480e+00            1.718577e-05            1.742200e-05            1.722900e-05            1.453049e-05            1.519660e-05            1.762171e-05            1.790741e-05            1.808930e-05            2.205765e+02            
8      1.981963e-03            1.900977e+00            2.004983e-03            2.003465e-03            1.981315e-03            1.847517e-03            1.815927e-03            2.053872e-03            2.035245e-03            2.077641e-03            3.571431e+01            
9      5.839769e-04            2.252698e+01            5.718593e-04            5.921591e-04            5.769334e-04            5.768798e+00            6.530638e-04            5.797396e-04            5.994653e-04            5.840760e-04            2.384296e+02            
10     5.076844e+00            4.478253e+00            6.393114e+00            4.715593e+00            5.609926e+00            5.928809e+00            1.171523e+01            9.292538e+00            6.147601e+00            6.504500e+00            1.855688e+02            
11     3.415753e+00            3.753230e+00            2.770552e+00            2.784309e+00            1.617184e+00            3.962843e+00            1.676880e+01            5.308043e+00            2.897289e+00            1.911827e+00            7.334321e+02            
12     1.664444e+00            2.915266e+00            1.453955e+00            1.549938e+00            1.474071e+00            1.926821e+00            1.382189e+01            3.346971e+00            1.459362e+00            1.961420e+00            1.407146e+02            
13     1.229246e+00            2.016455e+00            1.244162e+00            1.249871e+00            1.518349e+00            1.844239e+00            3.815009e+00            2.867701e+00            9.816912e-01            1.363484e+00            3.845386e+01            
14     3.330712e+00            6.017215e+00            3.469924e+00            3.302311e+00            4.379123e+00            2.977315e+00            3.486798e+00            3.832209e+00            3.485960e+00            2.984756e+00            1.437189e+01            
15     1.181775e+00            1.280133e+00            1.118375e+00            1.134848e+00            1.204235e+00            1.158051e+00            2.114387e+00            1.416808e+00            1.121594e+00            1.152378e+00            3.756973e+00            
16     6.986097e+02            7.249612e+02            5.053305e+02            6.072449e+02            5.907975e+02            6.386646e+02            5.106457e+02            5.838884e+02            6.877740e+02            4.983588e+02            9.851819e+03            
17     3.120103e+00            3.120134e+00            3.120103e+00            3.120103e+00            3.120103e+00            3.120103e+00            5.982931e+00            6.233301e+00            3.120103e+00            3.120104e+00            1.118056e+05            
18     1.343281e+01            1.459314e+01            5.420191e+00            1.084395e+01            1.081965e+01            1.428115e+01            2.958850e+01            3.044025e+01            1.236047e+01            1.085110e+01            6.899519e+01            
19     2.930745e+01            5.156131e+01            3.008320e+01            4.091111e+01            3.498902e+01            3.730696e+01            7.681272e+01            6.246693e+01            4.039071e+01            2.569312e+01            2.869032e+02            
20     2.279700e+01            2.286714e+01            2.465428e+01            2.186205e+01            2.366765e+01            2.255493e+01            2.647469e+01            2.728230e+01            2.382734e+01            2.304157e+01            4.580103e+01            
21     4.467927e+00            4.665000e+00            4.495976e+00            4.458744e+00            4.559303e+00            4.506317e+00            4.477903e+00            4.578053e+00            4.450747e+00            4.446660e+00            5.750001e+00            
22     2.268997e+00            3.264592e+00            2.357359e+00            2.473687e+00            2.332622e+00            2.570950e+00            3.422496e+00            2.708562e+00            2.568725e+00            2.770966e+00            1.638344e+01            
23     2.008786e+01            2.905779e+01            1.912895e+01            2.062436e+01            2.106294e+01            2.144414e+01            2.430269e+01            2.097410e+01            1.951675e+01            1.981547e+01            6.924412e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_idea_0.py  (error=8.415071e-08)
Task  1: variant_06_idea_0.py  (error=2.364988e-07)
Task  2: variant_06_idea_0.py  (error=2.096987e-05)
Task  3: variant_05_idea_0.py  (error=3.527089e-04)
Task  4: variant_06_idea_0.py  (error=2.356971e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_05_idea_0.py  (error=4.152292e-07)
Task  7: variant_05_idea_0.py  (error=1.453049e-05)
Task  8: variant_06_idea_0.py  (error=1.815927e-03)
Task  9: variant_02_idea_0.py  (error=5.718593e-04)
Task 10: variant_01_idea_0.py  (error=4.478253e+00)
Task 11: variant_04_idea_0.py  (error=1.617184e+00)
Task 12: variant_02_idea_0.py  (error=1.453955e+00)
Task 13: variant_08_idea_0.py  (error=9.816912e-01)
Task 14: variant_05_idea_0.py  (error=2.977315e+00)
Task 15: variant_02_idea_0.py  (error=1.118375e+00)
Task 16: variant_09_idea_0.py  (error=4.983588e+02)
Task 17: variant_05_idea_0.py  (error=3.120103e+00)
Task 18: variant_02_idea_0.py  (error=5.420191e+00)
Task 19: variant_09_idea_0.py  (error=2.569312e+01)
Task 20: variant_03_idea_0.py  (error=2.186205e+01)
Task 21: variant_09_idea_0.py  (error=4.446660e+00)
Task 22: original.py  (error=2.268997e+00)
Task 23: variant_02_idea_0.py  (error=1.912895e+01)

WIN COUNTS:
  variant_06_idea_0.py: 5 wins
  variant_05_idea_0.py: 5 wins
  variant_02_idea_0.py: 5 wins
  variant_09_idea_0.py: 3 wins
  variant_01_idea_0.py: 1 wins
  variant_04_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins
  variant_03_idea_0.py: 1 wins
  original.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
            """
            Sample a batch using Cholesky decomposition: x = mean + sigma * L @ z
            where C = L @ L.T. More efficient than eigendecomposition for
            large dimensions and avoids periodic B/D updates.
            """
            z = np.random.randn(self.pop_size, self.dim)

            # Ensure Cholesky factor L is valid (C = L @ L.T)
            if not hasattr(self, '_L') or self._L_stale:
                try:
                    # Symmetrize C to handle numerical asymmetries
                    C_sym = np.triu(self.C) + np.triu(self.C, 1).T
                    self._L = np.linalg.cholesky(C_sym)
                    self._L_stale = False
                except np.linalg.LinAlgError:
                    # C not positive definite - use eigendecomposition fallback
                    self._update_eigen_decomposition()
                    # Reconstruct from eigendecomposition: C = B @ diag(D²) @ B.T
                    self._L = self.B @ np.diag(self.D) @ self.B.T
                    self._L_stale = False

            # Sample: x = mean + sigma * L @ z
            # (pop_size, dim) = (dim,) + scalar * ((pop_size, dim) @ (dim, dim).T)
            population = self.mean[np.newaxis, :] + self.sigma * (z @ self._L.T)

            # Clip to bounds for numerical robustness
            population = np.clip(population, self.lb, self.ub)

            return population, z
```

# --- From variant_02_idea_0.py (5 wins) ---
```python
def _sample_population_batch(self):
            """
            Sample a batch of candidates with archive-based restart diversification.
            Maintains an archive of best solutions across restarts to prevent getting
            trapped in local optima by sampling from diverse historical solutions.
            """
            # Initialize archive if first call
            if not hasattr(self, 'archive') or self.archive is None:
                self.archive = []

            # Check if we should force restart diversity sampling
            force_archive_sampling = (self.stagnation_counter > self.dim)

            z = np.random.randn(self.pop_size, self.dim)

            # If using archive sampling, bias toward diverse historical solutions
            if force_archive_sampling and len(self.archive) > 0:
                archive_size = len(self.archive)
                # Sample a portion from archive (proportional to stagnation severity)
                archive_portion = min(len(self.archive), max(1, self.pop_size // 2))
                archive_indices = np.random.choice(archive_size, archive_portion, replace=False)

                for i, idx in enumerate(archive_indices[:archive_portion]):
                    if i < self.pop_size:
                        # Sample from neighborhood of archived solution
                        archive_solution = np.array(self.archive[idx])
                        # Use larger variance for exploration
                        z[i] = np.random.randn(self.dim) * 3.0

            # Ensure D and B are valid
            if self.D is None or self.B is None or np.any(self.D <= 0):
                self.D = np.ones(self.dim)
                self.B = np.eye(self.dim)

            # Transform: x = mean + sigma * B * D * z
            safe_D = np.maximum(self.D, 1e-20)
            scaled = z * safe_D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated

            # Add archived solutions directly to population
            if len(self.archive) > 0 and self.generation % 3 == 0:
                inject_count = min(len(self.archive), max(1, self.pop_size // 4))
                archive_samples = np.array(
                    list(self.archive[:inject_count]) + 
                    [np.random.randn(self.dim) for _ in range(inject_count)]
                )[:inject_count]

                if archive_samples.shape[0] >= inject_count:
                    start_idx = max(0, self.pop_size - inject_count)
                    for i in range(min(inject_count, self.pop_size - start_idx)):
                        if start_idx + i < self.pop_size:
                            population[start_idx + i] = archive_samples[i]

            return population, z
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
        """
        Sample a batch of candidates with restart-triggered exploration.
        When stagnated, inject restart-based diversity to escape local optima.
        """
        z = np.random.randn(self.pop_size, self.dim)

        # Check stagnation to trigger exploration mode
        stagnated = self.stagnation_counter > self.dim

        if stagnated:
            # Restart-based sampling: inject fresh diversity to escape local optima
            n_restart = max(1, int(0.3 * self.pop_size))

            # Initialize restart mean within bounds, blended with current mean
            restart_mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=self.dim)
            blend = np.random.uniform(0.3, 0.7)
            self.mean = blend * restart_mean + (1.0 - blend) * self.mean
            self.mean = np.clip(self.mean, self.lb, self.ub)

            # Sample restart portion from new mean
            restart_pop = self.mean[np.newaxis, :] + self.sigma * np.random.randn(n_restart, self.dim)
            restart_z = (restart_pop - self.mean[np.newaxis, :]) / (self.sigma + 1e-20)

            # Remaining from standard CMA-ES
            z_rest = np.random.randn(self.pop_size - n_restart, self.dim)
            scaled_rest = z_rest * self.D[np.newaxis, :]
            rotated_rest = scaled_rest @ self.B.T
            normal_pop = self.mean[np.newaxis, :] + self.sigma * rotated_rest

            # Combine and concatenate z vectors
            population = np.vstack([restart_pop, normal_pop])
            z = np.vstack([restart_z, z_rest])
        else:
            # Standard CMA-ES sampling
            scaled = z * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated

        population = self._clip_to_bounds_batch(population)
        return population, z
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated

        # Track elite memory for exploration
        if not hasattr(self, '_elite_memory'):
            self._elite_memory = []
        if self.best_x is not None:
            self._elite_memory.append(self.best_x.copy())
            max_memory = 5 * self.mu
            if len(self._elite_memory) > max_memory:
                self._elite_memory = self._elite_memory[-max_memory:]

        # Compute stagnation-based exploration factor
        stagnation_threshold = max(3, int(0.1 * self.dim))
        if self.generation > 5 and self.generation - self.eigen_decomp_gen > stagnation_threshold:
            stagnation_gen = self.generation - self.eigen_decomp_gen - stagnation_threshold
            exploration_factor = min(0.5 * (1.0 - np.exp(-stagnation_gen / (0.3 * self.dim))), 0.4)
        else:
            exploration_factor = 0.0

        # Inject exploration noise when stagnant
        if exploration_factor > 0.01 and len(self._elite_memory) > 0:
            elite_arr = np.array(self._elite_memory)
            elite_std = np.std(elite_arr, axis=0)
            noise_scale = self.sigma * exploration_factor * np.maximum(elite_std, 1e-8)
            exploration_noise = np.random.randn(self.pop_size, self.dim) * noise_scale[np.newaxis, :]
            population = population + exploration_noise

        return population, z
```

# --- From variant_05_idea_0.py (5 wins) ---
```python
def _sample_population_batch(self):
        """
        Adaptive sampling with three modes:
        1. Restart-mode: 60% uniform + 40% CMA-ES when covariance matrix degenerates
        2. Heavy-tailed: t-distribution sampling when stagnation detected (escape local optima)
        3. Standard CMA-ES sampling otherwise
        """
        # Check for covariance matrix degeneracy
        min_D = np.min(self.D)
        max_D = np.max(self.D)
        condition_number = max_D / min_D if min_D > 1e-20 else np.inf

        # Restart-mode: triggered by degenerate covariance OR very small eigenvalues
        # This injects uniform sampling to escape the current basin
        restart_mode = condition_number > 1e5 or min_D < 1e-10

        if restart_mode:
            # 60% uniform sampling for diversity, 40% CMA-ES for exploitation
            alpha = 0.4
            pop_uniform = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
            z = np.random.randn(self.pop_size, self.dim)
            scaled = z * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            pop_cmaes = self.mean[np.newaxis, :] + self.sigma * rotated
            population = alpha * pop_cmaes + (1.0 - alpha) * pop_uniform
        elif self.stagnation_counter > 5:
            # Heavy-tailed sampling: multivariate t-distribution (nu=4) for escaping local optima
            # Generates occasional large jumps by scaling normal samples with sqrt(chi2/nu)
            z = np.random.randn(self.pop_size, self.dim)
            chi2_samples = np.random.chisquare(4, size=self.pop_size)  # nu = 4 degrees of freedom
            t_scales = np.sqrt(4.0 / np.maximum(chi2_samples, 1e-10))  # prevent division by zero
            z_t = z * t_scales[:, np.newaxis]
            scaled = z_t * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated
        else:
            # Standard CMA-ES sampling: x = mean + sigma * B * D * z
            z = np.random.randn(self.pop_size, self.dim)
            scaled = z * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated

        # Numerical robustness: clip to bounds
        population = np.clip(population, self.lb, self.ub)

        return population, z
```

# --- From variant_06_idea_0.py (5 wins) ---
```python
def _sample_population_batch(self):
            """
            Sample a batch of candidates from the multivariate Student's t-distribution
            T(mean, sigma^2 * C, df) for heavy-tailed exploration. Returns array of shape
            (pop_size, dim).
            """
            # Degrees of freedom - lower = heavier tails for escaping local optima
            df = max(3.001, self.dim / 5.0)

            # Sample standard normals
            z = np.random.randn(self.pop_size, self.dim)

            # Sample chi-squared for t-distribution scaling
            chi2_samples = np.random.chisquare(df, self.pop_size)

            # Ensure numerical stability (avoid division by zero)
            chi2_samples = np.maximum(chi2_samples, 1e-6)

            # Compute scaling factor sqrt(df / chi2) for each sample
            scaling = np.sqrt(df / chi2_samples)

            # Apply t-distribution scaling to z-vectors
            t_z = z * scaling[:, np.newaxis]

            # Transform: x = mean + sigma * B * D * t_z
            scaled = t_z * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated

            return population, t_z
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
        """
        Sample a batch of candidates from a mixture of local and diverse exploration.
        When stagnation is detected, inject heavy-tailed solutions across the search space.
        """
        z = np.random.randn(self.pop_size, self.dim)

        # Compute stagnation ratio for adaptive diversity injection
        stagnation_threshold = 15 + self.dim
        stagnation_ratio = min(self.stagnation_counter / max(1, stagnation_threshold), 1.0)

        # Base scaling: local exploitation
        base_scaling = self.D[np.newaxis, :] * self.sigma
        rotated = z * base_scaling
        population = self.mean[np.newaxis, :] + rotated @ self.B.T

        # Inject diverse exploration when stagnant
        if self.stagnation_counter > stagnation_threshold // 2:
            # Compute diversity budget: fraction of population for wide search
            diversity_budget = int(self.pop_size * stagnation_ratio * 0.6)
            diversity_budget = max(1, min(diversity_budget, self.pop_size - 1))

            # Generate heavy-tailed samples using t-distribution
            df = max(2.0, 3.0 - stagnation_ratio)  # Lower df = heavier tails
            t_samples = np.random.standard_t(df, size=(diversity_budget, self.dim))

            # Scale t-samples to match covariance structure but with amplified variance
            amplify = 3.0 + stagnation_ratio * 7.0
            t_scaled = t_samples * self.D[np.newaxis, :] * self.sigma * amplify
            diverse_pop = self.mean[np.newaxis, :] + t_scaled @ self.B.T

            # Also inject completely random solutions for maximum diversity
            random_budget = max(1, diversity_budget // 3)
            random_pop = np.random.uniform(self.lb, self.ub, size=(random_budget, self.dim))

            # Combine: place diverse solutions at front (higher selection pressure)
            combined_diverse = np.vstack([diverse_pop, random_pop])[:diversity_budget]
            population[:diversity_budget] = combined_diverse

            # Expand step size for the entire batch during exploration phase
            expansion_factor = 1.0 + stagnation_ratio * 2.0
            population = self.mean[np.newaxis, :] + expansion_factor * (population - self.mean[np.newaxis, :])

        return population, z
```

# --- From variant_09_idea_0.py (3 wins) ---
```python
def _sample_population_batch(self):
        """
        Hybrid sampling: mix CMA-ES multivariate normal samples with
        uniformly distributed random samples for forced diversity.
        """
        pop_size = self.pop_size
        dim = self.dim

        # Base CMA-ES sampling
        z = np.random.randn(pop_size, dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        cma_pop = self.mean[np.newaxis, :] + self.sigma * rotated

        # Determine mix ratio: use more random samples when stagnating
        if self.stagnation_counter > 10:
            random_fraction = 0.40  # 40% random when stuck
        elif self.stagnation_counter > 5:
            random_fraction = 0.25  # 25% random when slowing
        else:
            random_fraction = 0.15  # 15% random baseline

        n_random = int(np.round(pop_size * random_fraction))
        n_random = max(1, min(n_random, pop_size - 1))  # Keep at least 1 CMA sample

        # Generate random samples uniform over bounds
        random_pop = np.random.uniform(self.lb, self.ub, size=(n_random, dim))

        # Combine: CMA-ES for first (pop_size - n_random), random for last n_random
        combined_pop = np.empty((pop_size, dim), dtype=np.float64)
        combined_pop[:-n_random] = cma_pop[:-n_random] if pop_size > n_random else cma_pop
        combined_pop[-n_random:] = random_pop

        # Shuffle to mix CMA and random samples throughout population
        shuffle_idx = np.random.permutation(pop_size)
        population = combined_pop[shuffle_idx]

        # For z_vectors: approximate for random samples
        z_combined = np.empty((pop_size, dim), dtype=np.float64)
        z_combined[:-n_random] = z[:-n_random] if pop_size > n_random else z
        z_combined[-n_random:] = np.random.randn(n_random, dim)  # fresh z for random samples
        z_vectors = z_combined[shuffle_idx]

        return population, z_vectors
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


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance matrix adaptation using a rank-1 update
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update
    - Periodic restarts when diversity collapses
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = max(4 * dim, 20)
        self.mu = self.pop_size // 2  # number of parents
        self.sigma = 30.0  # initial step size
        self.min_sigma = 1e-12
        self.max_sigma = 100.0

        # Weights for recombination
        self.weights = self._compute_recombination_weights()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # State
        self.mean = None
        self.C = None
        self.p_sigma = None
        self.p_c = None
        self.eigen_decomp_gen = 0
        self.B = None
        self.D = None
        self.invsqrtC = None

        self.best_x = None
        self.best_f = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_f = np.inf

    def _compute_recombination_weights(self):
        """Compute log-based recombination weights for the top mu individuals."""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = raw_weights / np.sum(raw_weights)
        return weights

    def _initialize_state(self):
        """Initialize or reset the distribution parameters (mean, covariance, paths)."""
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=self.dim)
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.sigma = 30.0
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrtC = np.eye(self.dim)
        self.eigen_decomp_gen = 0
        self.stagnation_counter = 0
        self.last_best_f = np.inf

    def _update_eigen_decomposition(self):
        """Perform eigendecomposition of the covariance matrix C for sampling."""
        # Symmetrize C to avoid numerical issues
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        # Clamp eigenvalues to avoid negative values from numerical errors
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        inv_D = 1.0 / self.D
        self.invsqrtC = self.B @ np.diag(inv_D) @ self.B.T
        self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        """
        Sample a batch of candidates from the multivariate normal distribution
        N(mean, sigma^2 * C). Returns array of shape (pop_size, dim).
        """
        # Sample standard normals and transform
        z = np.random.randn(self.pop_size, self.dim)
        # Transform: x = mean + sigma * B * D * z
        scaled = z * self.D[np.newaxis, :]  # (pop_size, dim)
        rotated = scaled @ self.B.T  # (pop_size, dim)
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds [-100, 100]^dim."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        """
        Sort population by fitness (ascending = better) and return sorted arrays.
        Handles NaN by assigning them worst possible fitness.
        """
        safe_fitness = np.where(np.isnan(fitness), np.inf, fitness)
        sorted_indices = np.argsort(safe_fitness)
        return population[sorted_indices], safe_fitness[sorted_indices], z_vectors[sorted_indices]

    def _update_mean(self, sorted_population):
        """
        Update the distribution mean using weighted recombination of the best mu individuals.
        """
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]  # (mu, dim)
        self.mean = self.weights @ selected  # (dim,)
        return old_mean

    def _update_evolution_paths(self, old_mean):
        """
        Update the cumulative step-size path (p_sigma) and the covariance path (p_c).
        Includes stagnation-triggered path reset to escape local optima.
        """
        # Detect stagnation and trigger evolution path reset
        if self.stagnation_counter > 20 + self.dim:
            # Reset evolution paths using current mean shift direction
            mean_shift = (self.mean - old_mean) / self.sigma
            # Preserve covariance matrix C but reset paths for fresh direction
            self.p_sigma = mean_shift * 0.1
            self.p_c = mean_shift * 0.1
            self.stagnation_counter = 0  # Reset counter after path reset

        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Update step-size path
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

        # Heaviside function for p_c update
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # Update covariance path
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using rank-1 and rank-mu updates.
        """
        # Rank-1 update component
        rank_one = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

        # Combined update
        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu

    def _update_step_size(self):
        """Adapt the global step size sigma using cumulative step-size adaptation (CSA)."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)

    def _track_best(self, population, fitness):
        """Update the global best solution found so far, ignoring NaN entries."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_fitness = fitness[valid_mask]
        valid_pop = population[valid_mask]
        local_best_idx = np.argmin(valid_fitness)
        if valid_fitness[local_best_idx] < self.best_f:
            self.best_f = valid_fitness[local_best_idx]
            self.best_x = valid_pop[local_best_idx].copy()

    def _check_stagnation(self, best_gen_fitness):
        """
        Detect if the search has stagnated and trigger a restart if needed.
        Returns True if a restart was performed.
        """
        improvement = self.last_best_f - best_gen_fitness
        if improvement < 1e-12 * (1.0 + abs(self.last_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_f = min(self.last_best_f, best_gen_fitness)

        # Check for various restart conditions
        should_restart = False
        if self.stagnation_counter > 20 + self.dim:
            should_restart = True
        if self.sigma < self.min_sigma * 10:
            should_restart = True
        if np.max(self.D) / np.min(self.D) > 1e7:
            should_restart = True

        if should_restart:
            self._perform_restart()
            return True
        return False

    def _perform_restart(self):
        """
        Restart the search with a new random mean, reset covariance and paths,
        but keep the global best tracked.
        """
        old_best_x = self.best_x.copy() if self.best_x is not None else None
        old_best_f = self.best_f

        self._initialize_state()

        # If we have a known best, bias the new mean slightly towards it
        if old_best_x is not None:
            blend = np.random.uniform(0.0, 0.3)
            self.mean = blend * old_best_x + (1.0 - blend) * self.mean
            self.mean = np.clip(self.mean, self.lb, self.ub)

        self.best_x = old_best_x
        self.best_f = old_best_f

    def _should_update_eigen(self):
        """Determine if eigendecomposition should be recomputed this generation."""
        update_interval = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        return (self.generation - self.eigen_decomp_gen) >= update_interval

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        """
        If func returned fewer values than expected (budget exhaustion),
        truncate population and z_vectors to match.
        """
        n_valid = len(fitness)
        return population[:n_valid], fitness[:n_valid], z_vectors[:n_valid]

    def _inject_best_into_population(self, population, z_vectors):
        """
        Occasionally inject the global best solution into the population to
        preserve elitism across restarts.
        """
        if self.best_x is not None and self.generation % 5 == 0:
            population[-1] = self.best_x.copy()
            # Approximate z_vector (not exact but acceptable)
            z_vectors[-1] = np.zeros(self.dim)
        return population, z_vectors

    def __call__(self, func, stopping_condition):
        """
        Main optimization loop: sample, evaluate, sort, update distribution.
        """
        self._initialize_state()
        self._update_eigen_decomposition()

        # Initial evaluation of a small seed population around mean
        seed_pop = self._clip_to_bounds_batch(
            self.mean[np.newaxis, :] + self.sigma * np.random.randn(self.pop_size, self.dim)
        )
        seed_fit = func(seed_pop)
        if stopping_condition():
            self._track_best(seed_pop, seed_fit[:len(seed_pop)])
            return self.best_f, self.best_x

        self._track_best(seed_pop, seed_fit)

        while not stopping_condition():
            # Eigen decomposition update
            if self._should_update_eigen():
                try:
                    self._update_eigen_decomposition()
                except np.linalg.LinAlgError:
                    self._perform_restart()
                    self._update_eigen_decomposition()

            # Sample new population
            population, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            # Inject best occasionally
            population, z_vectors = self._inject_best_into_population(population, z_vectors)
            population = self._clip_to_bounds_batch(population)

            # Evaluate
            fitness = func(population)
            if stopping_condition():
                if len(fitness) > 0:
                    population, fitness, z_vectors = self._handle_truncated_fitness(
                        population, fitness, z_vectors
                    )
                    self._track_best(population, fitness)
                break

            # Handle truncated returns
            if len(fitness) < len(population):
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._track_best(population, fitness)
                break

            # Track best
            self._track_best(population, fitness)

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(population, fitness, z_vectors)

            # Need at least mu valid individuals
            valid_count = np.sum(np.isfinite(sorted_fit))
            if valid_count < self.mu:
                self.generation += 1
                continue

            # Update mean
            old_mean = self._update_mean(sorted_pop)

            # Update evolution paths
            h_sigma = self._update_evolution_paths(old_mean)

            # Update covariance matrix
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)

            # Update step size
            self._update_step_size()

            # Check stagnation
            best_gen_f = sorted_fit[0]
            if np.isfinite(best_gen_f):
                restarted = self._check_stagnation(best_gen_f)
                if restarted:
                    self._update_eigen_decomposition()

            self.generation += 1

        return self.best_f, self.best_x

```