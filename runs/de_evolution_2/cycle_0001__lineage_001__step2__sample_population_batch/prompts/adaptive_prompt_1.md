Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_sample_population_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.420195e-07            1.299861e-07            1.028223e+02            1.363484e-07            9.075975e-07            6.445351e-08            1.869723e-07            -inf                    2.776042e-06            1.638396e-07            9.183205e-01            
1      2.940412e-07            2.943745e-07            1.855012e+02            3.030267e-07            1.509460e-06            1.906936e-07            2.238107e-07            -inf                    5.173548e-06            3.160545e-07            1.812579e+00            
2      2.424825e-05            2.499350e-05            8.919595e+02            2.476777e-05            9.476535e-05            2.835570e-05            3.484348e-05            -inf                    4.646729e-04            2.618441e-05            2.260030e+01            
3      4.153943e-04            4.060894e-04            3.497418e+02            4.206863e-04            9.062521e-04            3.666233e-04            3.168758e-04            -inf                    4.323132e-03            4.238326e-04            1.929686e+01            
4      2.489961e-02            2.486451e-02            5.944708e+00            2.526032e-02            3.794739e-02            2.382433e-02            2.661381e-02            -inf                    5.884571e-02            2.521753e-02            1.910579e+00            
5      1.000000e-08            1.000000e-08            7.618755e+09            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.150818e+04            
6      5.372434e-07            5.298949e-07            9.894329e+03            5.638407e-07            4.274922e-07            2.006409e+01            9.334390e+01            -inf                    6.043531e+01            5.745447e-07            2.718462e+01            
7      1.753360e-05            1.724700e-05            4.800161e+02            1.800693e-05            5.923577e-05            1.843340e-05            1.368145e-05            -inf                    2.966379e-04            1.791614e-05            1.314872e+01            
8      1.997055e-03            1.965531e-03            5.674718e+01            2.024973e-03            3.912909e-03            2.141736e-03            2.096910e-03            -inf                    9.175703e-03            2.083422e-03            6.641681e+00            
9      5.873080e-04            6.452289e-04            4.195350e+02            5.927763e-04            5.236554e-04            3.292202e+00            7.445325e+00            -inf                    8.035867e+00            6.277012e-04            1.361690e+01            
10     3.413339e+00            4.629204e+00            5.042973e+02            4.377088e+00            8.020873e+00            1.672470e+01            3.311015e+02            -inf                    4.166120e+02            5.398537e+00            7.166733e+00            
11     1.745365e+00            2.186783e+00            1.966748e+03            2.007459e+00            3.582512e+00            4.021028e+01            1.196589e+03            -inf                    1.565909e+03            2.046317e+00            2.385032e+01            
12     1.381109e+00            2.169322e+00            3.229227e+02            2.285042e+00            4.255510e+00            1.634150e+01            2.316207e+02            -inf                    2.642318e+02            1.930789e+00            7.881937e+00            
13     1.343336e+00            1.435556e+00            6.041827e+01            1.563108e+00            1.828527e+00            1.000103e+01            3.507955e+01            -inf                    5.708438e+01            1.750880e+00            6.648430e+00            
14     3.904585e+00            3.399327e+00            2.034465e+01            3.863099e+00            3.810372e+00            3.396935e+00            4.310233e+00            -inf                    1.894829e+01            3.243021e+00            4.796762e+00            
15     1.084512e+00            1.151594e+00            4.473569e+00            1.023780e+00            1.922639e+00            2.739608e+00            4.081755e+00            -inf                    4.218811e+00            1.107072e+00            2.138683e+00            
16     5.913093e+02            5.636808e+02            3.346600e+04            4.079409e+02            5.364875e+02            1.331189e+03            1.111788e+04            -inf                    2.643939e+04            4.910932e+02            5.648043e+02            
17     3.120103e+00            3.120103e+00            3.145198e+05            3.120103e+00            3.120107e+00            4.306186e+03            3.319559e+05            -inf                    3.126680e+05            3.120103e+00            9.928570e+02            
18     5.403979e+00            1.343710e+01            9.432467e+01            5.591664e+00            1.996617e+01            2.891459e+01            1.008432e+02            -inf                    9.935797e+01            1.697969e+01            1.496852e+01            
19     3.151107e+01            3.722701e+01            5.831779e+02            3.400632e+01            3.566238e+01            3.595178e+01            3.411868e+02            -inf                    4.778156e+02            2.904878e+01            4.217036e+01            
20     2.342560e+01            2.266131e+01            7.429848e+01            2.349933e+01            2.403548e+01            2.575272e+01            7.003513e+01            -inf                    6.830413e+01            2.347370e+01            2.425987e+01            
21     4.505597e+00            4.447770e+00            5.997047e+00            4.511121e+00            4.533160e+00            4.649599e+00            6.102265e+00            -inf                    5.998400e+00            4.535657e+00            4.817328e+00            
22     2.319560e+00            2.350091e+00            2.358882e+01            2.695416e+00            2.908407e+00            8.782623e+00            2.168138e+01            -inf                    2.287163e+01            2.585763e+00            5.316779e+00            
23     1.906595e+01            1.919384e+01            1.521888e+02            2.147898e+01            2.635073e+01            3.153522e+01            5.481950e+01            -inf                    5.255600e+01            2.014389e+01            1.854594e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_05_idea_0.py  (error=6.445351e-08)
Task  1: variant_05_idea_0.py  (error=1.906936e-07)
Task  2: original.py  (error=2.424825e-05)
Task  3: variant_06_idea_0.py  (error=3.168758e-04)
Task  4: variant_05_idea_0.py  (error=2.382433e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_04_idea_0.py  (error=4.274922e-07)
Task  7: variant_06_idea_0.py  (error=1.368145e-05)
Task  8: variant_01_idea_0.py  (error=1.965531e-03)
Task  9: variant_04_idea_0.py  (error=5.236554e-04)
Task 10: original.py  (error=3.413339e+00)
Task 11: original.py  (error=1.745365e+00)
Task 12: original.py  (error=1.381109e+00)
Task 13: original.py  (error=1.343336e+00)
Task 14: variant_09_idea_0.py  (error=3.243021e+00)
Task 15: variant_03_idea_0.py  (error=1.023780e+00)
Task 16: variant_03_idea_0.py  (error=4.079409e+02)
Task 17: variant_01_idea_0.py  (error=3.120103e+00)
Task 18: original.py  (error=5.403979e+00)
Task 19: variant_09_idea_0.py  (error=2.904878e+01)
Task 20: variant_01_idea_0.py  (error=2.266131e+01)
Task 21: variant_01_idea_0.py  (error=4.447770e+00)
Task 22: original.py  (error=2.319560e+00)
Task 23: variant_10_idea_0.py  (error=1.854594e+01)

WIN COUNTS:
  original.py: 7 wins
  variant_01_idea_0.py: 4 wins
  variant_05_idea_0.py: 3 wins
  variant_06_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  variant_09_idea_0.py: 2 wins
  variant_03_idea_0.py: 2 wins
  variant_10_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (4 wins) ---
```python
def _sample_population_batch(self):
            """
            Sample a batch of candidates using Cholesky decomposition of the
            covariance matrix: x = mean + sigma * L @ z where C ≈ L @ L.T.
            Returns array of shape (pop_size, dim).
            """
            # Sample standard normals
            z = np.random.randn(self.pop_size, self.dim)

            # Compute Cholesky factor with small regularization for numerical stability
            # This handles near-singular covariance matrices gracefully
            regularized_C = self.C + 1e-12 * np.eye(self.dim)
            try:
                L = np.linalg.cholesky(regularized_C)
            except np.linalg.LinAlgError:
                # Fallback: add more regularization and try again
                L = np.linalg.cholesky(regularized_C + 1e-8 * np.eye(self.dim))

            # Transform: x = mean + sigma * L @ z
            # L @ z computed efficiently using matrix multiplication
            noise = z @ L.T  # (pop_size, dim)
            population = self.mean[np.newaxis, :] + self.sigma * noise

            return population, z
```

# --- From variant_03_idea_0.py (2 wins) ---
```python
def _sample_population_batch(self):
            """
            Sample a batch of candidates using stagnation-triggered diversity injection.
            When stuck in local optima, replace a fraction of CMA-ES samples with fresh
            uniform samples to escape basins. The injection fraction scales with stagnation.
            """
            pop_size = self.pop_size
            dim = self.dim

            # Compute injection rate: proportional to stagnation, capped at 50%
            stagnation_fraction = min(self.stagnation_counter / max(1, 20 + dim), 1.0)
            injection_rate = 0.05 + 0.45 * stagnation_fraction  # 5% baseline, up to 50% when stuck

            n_uniform = int(np.round(pop_size * injection_rate))
            n_cmaes = pop_size - n_uniform

            # Generate CMA-ES samples from N(mean, sigma^2 * C)
            z_cmaes = np.random.randn(n_cmaes, dim)
            scaled = z_cmaes * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            cmaes_pop = self.mean[np.newaxis, :] + self.sigma * rotated

            # Generate uniform injection samples across bounds
            if n_uniform > 0:
                # Sample uniformly in [-100, 100]^dim
                uniform_pop = np.random.uniform(self.lb, self.ub, size=(n_uniform, dim))
                # Bias a few toward the best-known solution if available
                if self.best_x is not None and n_uniform >= 2:
                    bias_count = max(1, n_uniform // 3)
                    uniform_pop[:bias_count] = self.best_x + np.random.randn(bias_count, dim) * self.sigma * 0.5
                    uniform_pop[:bias_count] = np.clip(uniform_pop[:bias_count], self.lb, self.ub)
                population = np.vstack([cmaes_pop, uniform_pop])
                z_vectors = np.vstack([z_cmaes, np.zeros((n_uniform, dim))])
            else:
                population = cmaes_pop
                z_vectors = z_cmaes

            return population, z_vectors
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _sample_population_batch(self):
            """
            Sample a batch using mirrored sampling: half are independent samples,
            half are their systematic reflections. This guarantees symmetric
            exploration in all eigendirections and improves diversity on
            multimodal/deceptive tasks where standard sampling gets trapped.
            """
            half_size = self.pop_size // 2
            z = np.random.randn(half_size, self.dim)
            # Mirror: negate z for the second half
            z_mirrored = -z
            z_full = np.vstack([z, z_mirrored])
            # Transform: x = mean + sigma * B * D * z
            scaled = z_full * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated
            return population, z_full
```

# --- From variant_05_idea_0.py (3 wins) ---
```python
def _sample_population_batch(self):
            """
            Bimodal sampling: half from elite-biased tight distribution,
            half from global diverse distribution. Designed to escape local
            optima on ill-conditioned/deceptive tasks (16,19,20,23,18,10,14).
            """
            # Compute elite center from current covariance structure
            # Use eigendecomposition to get principal axes and scales
            z_all = np.random.randn(self.pop_size, self.dim)

            # Determine exploitation ratio: shrink variance when eigenvalues are extreme
            cond_num = np.max(self.D) / (np.min(self.D) + 1e-20)
            exploit_ratio = np.clip(1.0 / (1.0 + np.log1p(cond_num) / 5.0), 0.1, 0.7)

            # Elite center: mean shifted toward best eigenvector direction
            principal_dir = self.B[:, -1]  # direction of largest variance
            elite_center = self.mean + 0.3 * self.sigma * principal_dir * self.D[-1] * 0.1

            # Mode 1 (exploit): tight sampling around elite center
            n_exploit = self.pop_size // 2
            exploit_z = z_all[:n_exploit]
            exploit_scaled = exploit_z * (self.D * exploit_ratio)[np.newaxis, :]
            exploit_rotated = exploit_scaled @ self.B.T
            pop_exploit = elite_center[np.newaxis, :] + self.sigma * exploit_rotated

            # Mode 2 (explore): standard sampling around global mean
            n_explore = self.pop_size - n_exploit
            explore_z = z_all[n_explore:]
            explore_scaled = explore_z * self.D[np.newaxis, :]
            explore_rotated = explore_scaled @ self.B.T
            pop_explore = self.mean[np.newaxis, :] + self.sigma * explore_rotated

            # Combine both modes
            population = np.vstack([pop_exploit, pop_explore])
            z_vectors = z_all.copy()

            return population, z_vectors
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _sample_population_batch(self):
        """
        Sample a batch using deme-based niching: maintain multiple subpopulations
        distributed across the search space to preserve diversity on multi-modal tasks.
        Returns array of shape (pop_size, dim).
        """
        pop_size = self.pop_size
        dim = self.dim

        # Number of demes: sqrt(pop_size) demes, each with sqrt(pop_size) individuals
        n_demes = max(2, int(np.sqrt(pop_size)))
        deme_size = pop_size // n_demes

        all_candidates = []
        all_z = []

        # Compute current diversity: ratio of max to min eigenvalue
        eig_ratio = np.max(self.D) / np.min(self.D) if np.min(self.D) > 1e-20 else 1e7

        for d in range(n_demes):
            # Each deme has its own center distributed along principal axes
            # Spread demes along the dominant eigenvector to explore elongated valleys
            if eig_ratio > 10.0:
                # Ill-conditioned: spread demes along the major axis
                axis_weight = (d - n_demes / 2) / max(n_demes / 2, 1)
                major_axis = self.B[:, -1]  # eigenvector with largest eigenvalue
                deme_center = self.mean + axis_weight * 2.0 * self.sigma * self.D[-1] * major_axis
            else:
                # Isotropic: spread demes radially from mean
                angle = 2.0 * np.pi * d / n_demes
                spread_radius = self.sigma * min(5.0, np.mean(self.D))
                deme_center = self.mean + spread_radius * np.concatenate([
                    [np.cos(angle)],
                    [np.sin(angle)] + [0.0] * (dim - 2)
                ])

            # Sample individuals around this deme center
            z_deme = np.random.randn(deme_size, dim)
            scaled = z_deme * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            deme_pop = deme_center[np.newaxis, :] + self.sigma * 0.5 * rotated
            all_candidates.append(deme_pop)
            all_z.append(z_deme)

        # Handle remainder (if pop_size not divisible by n_demes)
        remainder = pop_size - len(all_candidates) * deme_size
        if remainder > 0:
            z_rem = np.random.randn(remainder, dim)
            scaled = z_rem * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            rem_pop = self.mean[np.newaxis, :] + self.sigma * rotated
            all_candidates.append(rem_pop)
            all_z.append(z_rem)

        # Concatenate all demes
        population = np.concatenate(all_candidates, axis=0)
        z_vectors = np.concatenate(all_z, axis=0)

        return population, z_vectors
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _sample_population_batch(self):
        """
        Hybrid multi-strategy sampling: CMA-ES + uniform exploration + boundary injection.
        Designed to escape local optima on difficult tasks (16, 19, 20, 23, etc.).
        """
        pop_size = self.pop_size
        dim = self.dim

        # Determine exploration fraction based on covariance condition number
        # High condition number -> more exploration needed
        max_D = self.D[-1] if len(self.D) > 0 else 1.0
        min_D = max(self.D[0], 1e-20) if len(self.D) > 0 else 1.0
        cond = max_D / min_D
        explore_frac = 0.35 if cond > 1e5 else 0.25
        n_explore = max(2, int(pop_size * explore_frac))
        n_boundary = max(1, int(pop_size * 0.10))
        n_cmaes = pop_size - n_explore - n_boundary

        # 1. Standard CMA-ES sampling (main exploitation component)
        z_main = np.random.randn(n_cmaes, dim)
        scaled = z_main * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        pop_main = self.mean[np.newaxis, :] + self.sigma * rotated

        # 2. Wide uniform exploration across bounds
        # This provides truly global search to escape local optima
        z_explore = np.random.randn(n_explore, dim)
        range_arr = np.array([self.ub - self.lb] * dim)
        mean_arr = np.array([(self.ub + self.lb) / 2.0] * dim)
        # Sample from uniform distribution scaled by sigma
        pop_explore = np.random.uniform(
            self.lb, self.ub, size=(n_explore, dim)
        )

        # 3. Boundary injection: sample near search space edges
        # This helps on tasks with optima near boundaries
        z_boundary = np.zeros((n_boundary, dim))
        pop_boundary = np.zeros((n_boundary, dim))
        for i in range(n_boundary):
            # Randomly choose corners/edges
            for d in range(dim):
                if np.random.rand() < 0.3:
                    # Near boundary
                    if np.random.rand() < 0.5:
                        pop_boundary[i, d] = np.random.uniform(self.lb, self.lb + 0.1 * range_arr[d])
                    else:
                        pop_boundary[i, d] = np.random.uniform(self.ub - 0.1 * range_arr[d], self.ub)
                else:
                    # Near current mean with perturbation
                    pop_boundary[i, d] = self.mean[d] + np.random.uniform(-1.0, 1.0) * self.sigma

        # Clip all to bounds
        pop_explore = np.clip(pop_explore, self.lb, self.ub)
        pop_boundary = np.clip(pop_boundary, self.lb, self.ub)

        # Combine all components
        population = np.vstack([pop_main, pop_explore, pop_boundary])
        z_vectors = np.vstack([z_main, z_explore, z_boundary])

        return population, z_vectors
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _sample_population_batch(self):
        """
        Sample a batch combining standard CMA-ES sampling with restart-mode sampling.
        When diversity collapses (large condition number or small sigma), inject
        restart candidates sampled from a wide distribution to escape local optima.
        """
        # Standard CMA-ES sampling
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        cmaes_population = self.mean[np.newaxis, :] + self.sigma * rotated

        # Determine if restart-mode sampling should be triggered
        # Use condition number of covariance as diversity metric
        if self.D is not None and len(self.D) > 0:
            cond_C = np.max(self.D) / np.maximum(np.min(self.D), 1e-30)
            trigger_restart = cond_C > 1e5 or self.sigma < 0.1
        else:
            trigger_restart = False

        if trigger_restart:
            # Restart-mode: sample from a MUCH wider distribution
            # Scale is 10x larger than current sigma to escape local optima
            restart_scale = max(self.sigma * 10.0, 10.0)
            restart_z = np.random.randn(self.pop_size, self.dim)
            # Sample from uniform cube of radius restart_scale in each eigen-direction
            restart_population = self.mean[np.newaxis, :] + restart_scale * restart_z * self.D[np.newaxis, :]
            restart_population = np.clip(restart_population, self.lb, self.ub)

            # Blend: 50% CMA-ES, 50% restart-mode for diversity
            blend_mask = np.random.rand(self.pop_size) < 0.5
            population = np.where(blend_mask[:, np.newaxis], restart_population, cmaes_population)
            # Recompute z for restart candidates (for path tracking)
            z = (population - self.mean[np.newaxis, :]) / max(self.sigma, 1e-30)
            # Project back to z-space using inverse transform
            z = z @ self.B / self.D[np.newaxis, :]
        else:
            population = cmaes_population

        return population, z
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
        """
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