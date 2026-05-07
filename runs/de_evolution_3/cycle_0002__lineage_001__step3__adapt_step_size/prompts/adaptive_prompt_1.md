Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_step_size` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      8.762113e-08            9.867643e-05            2.518834e-07            8.542510e-08            nan                     -inf                    1.010986e+02            2.482633e-07            2.076142e-07            7.714916e-04            1.017735e-07            
1      2.371495e-07            1.923405e-04            7.988841e-07            2.519422e-07            nan                     -inf                    1.900671e+02            6.776698e-07            6.504755e-07            2.183706e-05            4.636620e-07            
2      2.235770e-05            8.003736e-03            7.151879e+00            2.245216e-05            nan                     -inf                    8.280199e+02            5.203513e-05            5.059301e-05            2.941364e+00            2.166817e-04            
3      3.471579e-04            3.350464e-02            1.255778e+01            3.423918e-04            nan                     -inf                    3.401160e+02            2.342979e-03            7.900754e-04            1.012943e+00            8.545052e-03            
4      2.389402e-02            1.479017e-01            3.334611e-02            2.376358e-02            nan                     -inf                    5.937699e+00            4.230569e-02            3.102617e-02            5.359390e-01            3.235845e-02            
5      1.000000e-08            1.000000e-08            7.572183e+02            1.000000e-08            nan                     -inf                    6.604098e+09            1.000000e-08            1.000000e-08            8.541452e-03            1.000000e-08            
6      4.101431e-07            1.244929e-03            5.046864e-06            4.327217e-07            nan                     -inf                    1.084878e+04            1.518536e-06            1.599464e-06            1.569015e+00            1.438941e-06            
7      1.341883e-05            4.606536e-03            1.259409e-04            1.369799e-05            nan                     -inf                    4.717773e+02            3.832857e-05            3.714847e-05            6.699691e+00            1.016300e-04            
8      1.753817e-03            5.487472e-02            2.895632e+00            1.729374e-03            nan                     -inf                    5.595671e+01            3.099980e-03            3.405767e-03            2.528328e+00            3.887886e-01            
9      4.805398e-04            5.150014e-02            9.496705e+00            5.002527e-04            nan                     -inf                    4.421228e+02            4.012451e-03            2.026421e-02            2.705141e+00            1.078044e-03            
10     5.734144e-03            1.642578e-03            3.734780e+02            1.059009e-02            nan                     -inf                    4.096863e+02            5.872690e+00            5.134447e+00            2.066664e+02            4.382725e+00            
11     2.101047e+00            8.420777e-03            1.405330e+03            2.237784e+00            nan                     -inf                    1.675429e+03            3.761542e+00            6.993042e+00            1.587560e+01            1.026884e+00            
12     7.497072e-01            5.121666e-02            2.454216e+02            1.240146e+00            nan                     -inf                    2.723720e+02            2.405428e+00            4.696246e+00            2.302897e+02            2.770216e+00            
13     1.392974e+00            6.324458e-02            8.403594e+00            1.193212e+00            nan                     -inf                    5.746137e+01            1.496698e+00            1.239070e+00            2.398302e+01            1.184189e+00            
14     2.641898e+00            4.529661e-01            1.832426e+01            6.308186e-01            nan                     -inf                    2.047649e+01            8.410296e+00            1.092866e+01            2.535014e+00            1.144430e+01            
15     1.031813e+00            7.514344e-01            4.236027e+00            1.104245e+00            nan                     -inf                    4.347404e+00            1.542050e+00            1.931567e+00            4.300791e+00            1.902855e+00            
16     3.999156e-08            3.588143e-04            2.159987e+04            3.758999e-04            nan                     -inf                    3.173115e+04            5.771247e-06            3.252396e-02            4.996233e-01            1.988774e-08            
17     3.120103e+00            3.139039e+00            3.503330e+00            3.120103e+00            nan                     -inf                    2.725572e+05            3.120105e+00            3.120105e+00            2.721336e+01            3.120105e+00            
18     2.675705e+00            2.830906e+00            2.721898e+00            2.674756e+00            nan                     -inf                    9.190485e+01            2.675539e+00            2.679026e+00            4.655716e+01            2.675975e+00            
19     1.232558e+01            4.596342e+00            4.924160e+02            2.510520e+00            nan                     -inf                    5.475475e+02            1.017955e+02            1.300229e+02            9.226902e+01            1.859320e+02            
20     4.491002e+00            1.401230e+00            6.465810e+01            2.104422e+00            nan                     -inf                    6.903345e+01            6.271302e+00            2.359931e+01            4.979872e+01            1.505511e+01            
21     4.231938e+00            4.123960e+00            5.746214e+00            4.251237e+00            nan                     -inf                    5.967386e+00            4.214262e+00            5.268592e+00            4.139350e+00            4.922087e+00            
22     2.618660e+00            1.477432e+00            2.172465e+01            2.452986e+00            nan                     -inf                    2.280184e+01            1.084541e+01            1.467004e+01            2.198737e+01            1.289072e+01            
23     6.387760e-01            6.012812e-02            1.251026e+02            1.123964e-01            nan                     -inf                    1.414564e+02            1.827823e+01            3.271156e+01            1.695084e+00            1.030285e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_idea_0.py  (error=8.542510e-08)
Task  1: original.py  (error=2.371495e-07)
Task  2: original.py  (error=2.235770e-05)
Task  3: variant_03_idea_0.py  (error=3.423918e-04)
Task  4: variant_03_idea_0.py  (error=2.376358e-02)
Task  5: SKIPPED (trivial)
Task  6: original.py  (error=4.101431e-07)
Task  7: original.py  (error=1.341883e-05)
Task  8: variant_03_idea_0.py  (error=1.729374e-03)
Task  9: original.py  (error=4.805398e-04)
Task 10: variant_01_idea_0.py  (error=1.642578e-03)
Task 11: variant_01_idea_0.py  (error=8.420777e-03)
Task 12: variant_01_idea_0.py  (error=5.121666e-02)
Task 13: variant_01_idea_0.py  (error=6.324458e-02)
Task 14: variant_01_idea_0.py  (error=4.529661e-01)
Task 15: variant_01_idea_0.py  (error=7.514344e-01)
Task 16: variant_10_idea_0.py  (error=1.988774e-08)
Task 17: original.py  (error=3.120103e+00)
Task 18: variant_03_idea_0.py  (error=2.674756e+00)
Task 19: variant_03_idea_0.py  (error=2.510520e+00)
Task 20: variant_01_idea_0.py  (error=1.401230e+00)
Task 21: variant_01_idea_0.py  (error=4.123960e+00)
Task 22: variant_01_idea_0.py  (error=1.477432e+00)
Task 23: variant_01_idea_0.py  (error=6.012812e-02)

WIN COUNTS:
  variant_01_idea_0.py: 10 wins
  variant_03_idea_0.py: 6 wins
  original.py: 6 wins
  variant_10_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (10 wins) ---
```python
def _adapt_step_size(self):
        # Track success rate-based adaptation (fundamentally different from p_sigma path)
        success_window = max(3, self.mu // 2)

        # Use recent fitness history if available
        if not hasattr(self, '_recent_successes'):
            self._recent_successes = []

        # Simple success heuristic: compare median of selected vs full population
        # This is a direct success-ratio approach, not using evolution paths
        if hasattr(self, '_last_median_fit') and np.isfinite(self._last_median_fit):
            valid_fitness = self.fitness_buffer if hasattr(self, 'fitness_buffer') else np.array([np.inf])
            current_median = np.median(valid_fitness[np.isfinite(valid_fitness)]) if np.any(np.isfinite(valid_fitness)) else self._last_median_fit

            # Success ratio: how often do we see improvement?
            if self._last_median_fit > 0:
                ratio = current_median / (self._last_median_fit + 1e-20)
            elif current_median < 0:
                ratio = 1.0 - (current_median / (self._last_best_fit + 1e-20))
            else:
                ratio = 1.0

            # Success rate based on relative fitness change
            if ratio < 0.99:  # Improvement detected
                self._recent_successes.append(1)
            else:
                self._recent_successes.append(0)

            # Keep window bounded
            if len(self._recent_successes) > success_window:
                self._recent_successes.pop(0)

            # Compute success rate
            if len(self._recent_successes) >= success_window:
                success_rate = np.mean(self._recent_successes)
                # Target success rate ~1/5 for CMA-ES like behavior
                target_success = 0.20
                # Adaptation: increase sigma when success rate is high
                adaptation = (success_rate - target_success) / target_success
                self.sigma *= np.exp(0.1 * np.clip(adaptation, -2.0, 2.0))
            else:
                # Insufficient data, use stagnation-based heuristic
                if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 5:
                    self.sigma *= 1.5  # Increase diversity when stagnant
        else:
            self._recent_successes = []

        # Store for next iteration
        valid_fit = self.fitness_buffer if hasattr(self, 'fitness_buffer') and len(self.fitness_buffer) > 0 else np.array([np.inf])
        if np.any(np.isfinite(valid_fit)):
            self._last_median_fit = float(np.median(valid_fit[np.isfinite(valid_fit)]))
            self._last_best_fit = float(np.min(valid_fit[np.isfinite(valid_fit)]))

        # Clip to safe bounds
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```

# --- From variant_03_idea_0.py (6 wins) ---
```python
def _adapt_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        # Primary signal: evolution path normalized deviation (original)
        sig1 = p_sigma_norm / self.chi_n - 1.0

        # Secondary signal: fitness improvement rate
        if hasattr(self, 'prev_fitness_for_sigma') and self.prev_fitness_for_sigma > 0:
            rel_improve = max(self.prev_fitness_for_sigma - self.best_f, 0.0) / (abs(self.prev_fitness_for_sigma) + 1e-20)
            sig2 = np.tanh(5.0 * rel_improve)  # Map improvement to [-1, 1]
        else:
            sig2 = 0.0

        # Tertiary signal: population diversity (rank-based spread)
        if hasattr(self, '_cached_diversity') and self._cached_diversity > 0:
            sig3 = 2.0 * (self._cached_diversity / (self.sigma + 1e-20) - 1.0)
            sig3 = np.clip(sig3, -3.0, 3.0)
        else:
            sig3 = 0.0

        self.prev_fitness_for_sigma = self.best_f if np.isfinite(self.best_f) else self.prev_fitness_for_sigma

        # Adaptive weights: emphasize primary when healthy, secondary/tertiary when stuck
        w1 = 0.7
        w2 = 0.15
        w3 = 0.15
        combined = w1 * sig1 + w2 * sig2 + w3 * sig3

        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * combined)
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        # Track fitness history for momentum
        if not hasattr(self, '_fitness_momentum'):
            self._fitness_momentum = []
        self._fitness_momentum.append(self.best_f)
        if len(self._fitness_momentum) > 5:
            self._fitness_momentum.pop(0)

        # Compute momentum from recent fitness improvements
        momentum = 0.0
        if len(self._fitness_momentum) >= 2:
            improvements = []
            for i in range(1, len(self._fitness_momentum)):
                delta = self._fitness_momentum[i-1] - self._fitness_momentum[i]
                if delta > 0:
                    improvements.append(delta)
            if improvements:
                momentum = np.mean(improvements) / (abs(self.best_f) + 1e-20)

        # Standard CMA-ES adaptation
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        c_sigma_adapt = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)

        # Determine sigma update based on momentum and stagnation
        if self.stagnation_counter > 10:
            # Strong stagnation: aggressive sigma increase
            self.sigma *= 2.5
        elif self.stagnation_counter > 5:
            # Moderate stagnation: moderate sigma increase
            self.sigma *= 1.5
        elif momentum < 1e-6:
            # Low momentum but not stagnating: slight increase
            self.sigma *= np.exp(c_sigma_adapt * 1.5)
        elif momentum > 0.1:
            # Good momentum: reduce step size for refinement
            self.sigma *= np.exp(c_sigma_adapt * 0.5)
        else:
            # Normal adaptation
            self.sigma *= np.exp(c_sigma_adapt)

        # Diversity-based correction: detect population collapse
        if hasattr(self, '_last_pop_spread'):
            pop_spread = np.std(self.mean) if self.dim > 0 else 1.0
            if pop_spread < 0.5 * self._last_pop_spread and self.stagnation_counter > 2:
                self.sigma *= 2.0
        self._last_pop_spread = np.std(self.mean) if self.dim > 0 else 1.0

        self.sigma = np.clip(self.sigma, 1e-20, 1e5)
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
    A simplified CMA-ES inspired optimizer with adaptive operator selection.
    Uses Thompson Sampling to select the best _sample_population_batch strategy
    during optimization from a portfolio of strategies.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))
        self.pop_size = max(self.pop_size, 2 * dim)
        self.mu = self.pop_size // 2
        self.best_f = np.inf
        self.best_x = None
        self._initialize_strategy_params()

        # Adaptive operator selection: Thompson Sampling
        self.n_operators = 5  # number of sampling strategies
        # Beta distribution parameters for Thompson Sampling
        self.op_alpha = np.ones(self.n_operators) * 1.0
        self.op_beta = np.ones(self.n_operators) * 1.0
        self.current_operator = 0
        # Sliding window for credit assignment
        self.window_size = 20
        self.op_history = []  # list of (operator_idx, reward)
        # Track fitness improvements per operator
        self.op_total_reward = np.zeros(self.n_operators)
        self.op_count = np.zeros(self.n_operators)

        # Archive for variant_06
        self.sampling_archive = []

    def _initialize_strategy_params(self):
        dim = self.dim
        mu = self.mu

        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1,
                            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0
        self.C = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.eigenvalues = np.ones(dim)
        self.eigenvectors = np.eye(dim)
        self.gen_since_eigen = 0
        self.stagnation_counter = 0
        self.prev_best_f = np.inf

    def _decompose_covariance(self):
        try:
            self.C = 0.5 * (self.C + self.C.T)
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.gen_since_eigen = 0
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.gen_since_eigen = 0

    def _select_operator(self):
        """Select operator using Thompson Sampling."""
        samples = np.array([
            np.random.beta(self.op_alpha[i], self.op_beta[i])
            for i in range(self.n_operators)
        ])
        self.current_operator = int(np.argmax(samples))
        return self.current_operator

    def _update_operator_reward(self, operator_idx, reward):
        """Update Thompson Sampling parameters based on reward."""
        reward = float(reward)
        operator_idx = int(operator_idx)

        self.op_history.append((operator_idx, reward))

        # Keep sliding window
        if len(self.op_history) > self.window_size:
            old_op, old_reward = self.op_history.pop(0)
            # Decay old contributions
            self.op_alpha[old_op] = max(1.0, self.op_alpha[old_op] - old_reward)
            self.op_beta[old_op] = max(1.0, self.op_beta[old_op] - (1.0 - old_reward))

        # Update with new reward
        self.op_alpha[operator_idx] += reward
        self.op_beta[operator_idx] += (1.0 - reward)

        # Prevent parameters from growing too large (gentle decay)
        max_param = 50.0
        for i in range(self.n_operators):
            total = self.op_alpha[i] + self.op_beta[i]
            if total > max_param:
                scale = max_param / total
                self.op_alpha[i] = max(1.0, self.op_alpha[i] * scale)
                self.op_beta[i] = max(1.0, self.op_beta[i] * scale)

    # --- Operator 0: Original sampling ---
    def _sample_original(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

    # --- Operator 1: Antithetic variates (variant_01) ---
    def _sample_antithetic(self):
        dim = self.dim
        lam = self.pop_size

        if lam % 2 == 0:
            n_pairs = lam // 2
        else:
            n_pairs = (lam - 1) // 2

        z_pairs = np.random.randn(n_pairs, dim)
        z = np.vstack([z_pairs, -z_pairs])

        if lam % 2 == 1:
            z_extra = np.random.randn(1, dim)
            z = np.vstack([z, z_extra])

        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

    # --- Operator 2: Hybrid center (variant_02) ---
    def _sample_hybrid_center(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)

        if self.best_x is not None and np.isfinite(self.best_f):
            diff_norm = np.linalg.norm(self.mean - self.best_x)
            drift_scale = min(diff_norm / (self.sigma + 1e-20), 1.0)
            w_best = 0.3 * drift_scale
            center = (1.0 - w_best) * self.mean + w_best * self.best_x
        else:
            center = self.mean

        sqrt_eig = np.sqrt(np.maximum(self.eigenvalues, 1e-20))
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = center[None, :] + self.sigma * y.T
        return population, z

    # --- Operator 3: Diversity injection (variant_04) ---
    def _sample_diversity_injection(self):
        dim = self.dim
        lam = self.pop_size

        if self.stagnation_counter > 8:
            random_frac = 0.5
        elif self.stagnation_counter > 4:
            random_frac = 0.3
        elif self.stagnation_counter > 2:
            random_frac = 0.2
        else:
            random_frac = 0.1

        n_random = max(1, int(lam * random_frac))
        n_cma = lam - n_random

        z_all = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues + 1e-20)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z_all.T)
        cma_samples = self.mean[None, :] + self.sigma * y.T

        random_samples = np.random.uniform(self.lb, self.ub, (n_random, dim))

        if n_random > 0 and n_cma > 0:
            population = np.vstack([cma_samples[:n_cma], random_samples])
        elif n_random > 0:
            population = random_samples
        else:
            population = cma_samples

        population = np.clip(population, self.lb, self.ub)
        return population, z_all

    # --- Operator 4: Archive-guided with heavy tails (variant_06) ---
    def _sample_archive_guided(self):
        dim = self.dim
        lam = self.pop_size

        if not hasattr(self, 'sampling_archive'):
            self.sampling_archive = []
        if len(self.sampling_archive) > 0 and len(self.sampling_archive[0]) != dim:
            self.sampling_archive = []

        is_stagnant = self.stagnation_counter > max(3, int(5 * self.dim / self.pop_size))

        n_archive = 0
        if is_stagnant and len(self.sampling_archive) >= 5:
            n_archive = int(lam * np.random.uniform(0.4, 0.7))
            n_archive = min(n_archive, len(self.sampling_archive))

        n_cmaes = lam - n_archive

        population = np.empty((lam, dim))
        z = np.empty((lam, dim))

        if n_cmaes > 0:
            z[:n_cmaes] = np.random.randn(n_cmaes, dim)
            heavy_tail_mask = np.random.rand(n_cmaes) < 0.2
            if np.any(heavy_tail_mask):
                z[:n_cmaes][heavy_tail_mask] = np.random.standard_t(2.0, size=(np.sum(heavy_tail_mask), dim))

            sqrt_eig = np.sqrt(self.eigenvalues)
            y = self.eigenvectors @ (sqrt_eig[:, None] * z[:n_cmaes].T)
            population[:n_cmaes] = self.mean[None, :] + self.sigma * y.T

        if n_archive > 0:
            archive_array = np.array(self.sampling_archive)
            archive_indices = np.random.randint(0, len(archive_array), size=n_archive)
            sampled_from_archive = archive_array[archive_indices]

            exploration_noise = np.random.randn(n_archive, dim)
            heavy_tail_pert = np.random.standard_t(2.0, size=(n_archive, dim))
            combined_noise = 0.7 * exploration_noise + 0.3 * heavy_tail_pert

            sqrt_eig = np.sqrt(self.eigenvalues)
            noise = self.eigenvectors @ (sqrt_eig[:, None] * combined_noise.T)
            population[n_cmaes:] = sampled_from_archive + self.sigma * 2.0 * noise.T

            z[n_cmaes:] = combined_noise

        return population, z

    def _sample_population_batch(self):
        """Adaptively select and use the best sampling strategy."""
        op_idx = self._select_operator()

        if op_idx == 0:
            result = self._sample_original()
        elif op_idx == 1:
            result = self._sample_antithetic()
        elif op_idx == 2:
            result = self._sample_hybrid_center()
        elif op_idx == 3:
            result = self._sample_diversity_injection()
        elif op_idx == 4:
            result = self._sample_archive_guided()
        else:
            result = self._sample_original()

        return result

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, z_vectors

        sort_fitness = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fitness)
        return population[order], sort_fitness[order], z_vectors[order]

    def _update_best(self, population, fitness):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_idx = np.where(valid_mask)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.best_f:
            self.best_f = float(fitness[best_idx])
            self.best_x = population[best_idx].copy()

    def _update_mean(self, sorted_population):
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]
        self.mean = self.weights @ selected
        return old_mean

    def _update_evolution_path_sigma(self, old_mean):
        inv_sqrt_eig = 1.0 / np.sqrt(self.eigenvalues + 1e-20)
        inv_sqrt_C = self.eigenvectors @ np.diag(inv_sqrt_eig) @ self.eigenvectors.T

        displacement = (self.mean - old_mean) / self.sigma
        self.p_sigma = ((1.0 - self.c_sigma) * self.p_sigma +
                        np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) *
                        inv_sqrt_C @ displacement)

    def _update_evolution_path_c(self, old_mean):
        dim = self.dim
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (dim + 1.0)) * self.chi_n
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        displacement = (self.mean - old_mean) / self.sigma
        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) *
                     displacement)
        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        dim = self.dim
        selected = sorted_population[:self.mu]

        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        diff = (selected - old_mean[None, :]) / self.sigma
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu)

        self.C = 0.5 * (self.C + self.C.T)

    def _adapt_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)

    def _check_stagnation(self, fitness):
        valid = fitness[np.isfinite(fitness)]
        if len(valid) == 0:
            return False

        current_best = float(np.min(valid))
        improvement = self.prev_best_f - current_best

        if improvement < 1e-12 * (1.0 + abs(self.prev_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        self.prev_best_f = min(self.prev_best_f, current_best)

        stagnation_limit = 10 + int(30 * self.dim / self.pop_size)
        return self.stagnation_counter > stagnation_limit

    def _check_condition_number(self):
        ratio = np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-30)
        return ratio > 1e14

    def _restart(self):
        self.pop_size = min(int(self.pop_size * 1.5), 800)
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        if self.best_x is not None and np.random.rand() < 0.3:
            self.mean = self.best_x + np.random.randn(self.dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            z_vectors = z_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, z_vectors

    def _should_decompose(self):
        self.gen_since_eigen += 1
        decompose_freq = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        decompose_freq = min(decompose_freq, self.pop_size)
        return self.gen_since_eigen >= decompose_freq

    def _update_archive(self, sorted_population, sorted_fitness):
        """Update sampling archive with good solutions for variant_06."""
        if not hasattr(self, 'sampling_archive'):
            self.sampling_archive = []

        # Add top mu solutions to archive
        n_add = min(self.mu, len(sorted_population))
        for i in range(n_add):
            if np.isfinite(sorted_fitness[i]):
                self.sampling_archive.append(sorted_population[i].copy())

        # Keep archive bounded
        max_archive = 200
        if len(self.sampling_archive) > max_archive:
            self.sampling_archive = self.sampling_archive[-max_archive:]

    def __call__(self, func, stopping_condition):
        self._initialize_state()
        self._decompose_covariance()
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        self.sampling_archive = []

        # Reset operator selection
        self.op_alpha = np.ones(self.n_operators) * 1.0
        self.op_beta = np.ones(self.n_operators) * 1.0
        self.op_history = []

        # Initial evaluation to seed the best
        initial_pop = self._clip_to_bounds_batch(
            np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        )
        if not stopping_condition():
            initial_fit = func(initial_pop)
            if stopping_condition():
                self._update_best(initial_pop[:len(initial_fit)], initial_fit)
                return float(self.best_f), self.best_x
            self._update_best(initial_pop[:len(initial_fit)], initial_fit)

            valid_mask = np.isfinite(initial_fit)
            if np.any(valid_mask):
                best_idx = np.argmin(np.where(valid_mask, initial_fit, np.inf))
                self.mean = initial_pop[best_idx].copy()

        generation = 0
        while not stopping_condition():
            # Record fitness before this generation for reward computation
            pre_gen_best = float(self.best_f)

            # Sample new population (operator is selected inside)
            population, z_vectors = self._sample_population_batch()
            used_operator = int(self.current_operator)
            population = self._clip_to_bounds_batch(population)

            # Evaluate
            fitness = func(population)

            if stopping_condition():
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._update_best(population, fitness)
                break

            population, fitness, z_vectors = self._handle_truncated_fitness(
                population, fitness, z_vectors
            )
            self._update_best(population, fitness)

            if len(fitness) < self.mu:
                break

            # Compute reward for operator selection
            post_gen_best = float(self.best_f)
            # Reward based on relative improvement
            if pre_gen_best > 0:
                rel_improvement = (pre_gen_best - post_gen_best) / (abs(pre_gen_best) + 1e-20)
            elif pre_gen_best == 0:
                rel_improvement = 0.0
            else:
                rel_improvement = (pre_gen_best - post_gen_best) / (abs(pre_gen_best) + 1e-20)

            # Also consider the quality of the population (median fitness vs best)
            valid_fit = fitness[np.isfinite(fitness)]
            if len(valid_fit) > 0:
                median_fit = float(np.median(valid_fit))
                best_fit = float(np.min(valid_fit))
                # Reward: normalize improvement to [0, 1]
                # Use sigmoid-like mapping
                reward = float(np.clip(rel_improvement * 100.0, 0.0, 1.0))
                # Bonus if we found improvement
                if post_gen_best < pre_gen_best:
                    reward = max(reward, 0.6)
                # Small baseline reward for not crashing
                reward = max(reward, 0.1)
            else:
                reward = 0.0

            self._update_operator_reward(used_operator, reward)

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(
                population, fitness, z_vectors
            )

            # Update archive for variant_06
            self._update_archive(sorted_pop, sorted_fit)

            # Update distribution
            old_mean = self._update_mean(sorted_pop)
            self._update_evolution_path_sigma(old_mean)
            h_sigma = self._update_evolution_path_c(old_mean)
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)
            self._adapt_step_size()

            # Eigendecomposition if needed
            if self._should_decompose():
                self._decompose_covariance()

            # Check for restart conditions
            need_restart = self._check_stagnation(sorted_fit)
            if not need_restart:
                need_restart = self._check_condition_number()

            if need_restart:
                self._restart()
                self._decompose_covariance()

            generation += 1

        return float(self.best_f), self.best_x

```