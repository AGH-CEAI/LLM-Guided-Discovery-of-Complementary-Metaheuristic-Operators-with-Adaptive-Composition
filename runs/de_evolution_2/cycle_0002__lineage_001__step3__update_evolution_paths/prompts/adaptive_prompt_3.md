Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_evolution_paths` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.420195e-07            1.228319e-06            6.285457e-07            2.938079e-07            1.023118e-07            1.282716e-07            1.218753e-07            1.044010e-07            1.851130e-04            1.401114e-07            1.403597e-07            
1      2.940412e-07            2.627148e-06            8.564851e-07            4.971700e-07            2.847711e-07            2.946142e-07            2.759677e-07            3.056131e-07            3.741035e-04            2.949000e-07            2.849057e-07            
2      2.424825e-05            1.695198e-04            2.449014e-05            1.008062e-04            2.509252e-05            2.445659e-05            2.438751e-05            2.663438e-05            1.394803e-02            2.460300e-05            2.438376e-05            
3      4.153943e-04            1.363667e-03            2.822094e-04            9.612645e-01            4.074310e-04            4.121012e-04            4.056635e-04            4.340604e-04            5.155878e-02            4.127511e-04            4.082429e-04            
4      2.489961e-02            4.501594e-02            2.643802e-02            1.012024e-01            2.491664e-02            2.477445e-02            2.480142e-02            2.530044e-02            1.778556e-01            2.476233e-02            2.468596e-02            
5      1.000000e-08            1.000000e-08            1.000000e-08            4.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.761707e-08            1.000000e-08            1.000000e-08            
6      5.372434e-07            3.065343e-07            3.373707e-07            1.614993e+01            5.249115e-07            5.337507e-07            5.305034e-07            5.240045e-07            2.649398e-03            5.369735e-07            5.436931e-07            
7      1.753360e-05            1.065043e-04            1.359036e-05            2.309066e+00            1.746086e-05            1.755136e-05            1.760934e-05            1.860504e-05            7.819692e-03            1.746749e-05            1.645778e-05            
8      1.997055e-03            5.653615e-03            1.852149e-03            1.123376e+00            1.978333e-03            1.981838e-03            1.982760e-03            2.063323e-03            7.745183e-02            2.000585e-03            1.965815e-03            
9      5.873080e-04            4.316223e-04            7.582243e-04            7.853203e+00            5.811118e-04            5.859224e-04            5.755518e-04            5.787532e-04            8.153616e-02            5.788780e-04            5.784689e-04            
10     3.413339e+00            2.754413e+02            1.226241e+01            5.125000e+00            5.244827e+00            5.127142e+00            4.642865e+00            3.656078e+00            7.649691e-04            4.962449e+00            3.479127e+00            
11     1.745365e+00            1.088105e+03            2.776202e+01            2.514490e+00            2.848831e+00            2.109829e+00            3.076886e+00            2.221706e+00            3.501453e-03            1.658356e+00            3.353835e+00            
12     1.381109e+00            2.257530e+02            1.285325e+01            2.768642e+00            1.441015e+00            1.724960e+00            1.681936e+00            9.460475e-01            2.521045e-01            1.303194e+00            1.986658e+00            
13     1.343336e+00            4.300138e+01            2.312010e+00            1.504969e+00            1.150804e+00            1.923199e+00            1.382584e+00            1.103585e+00            7.678428e-02            1.085237e+00            1.330831e+00            
14     3.904585e+00            1.104028e+01            3.502336e+00            3.739649e+00            3.399345e+00            3.569595e+00            3.602527e+00            3.339568e+00            2.610327e+00            3.404061e+00            3.713113e+00            
15     1.084512e+00            4.077154e+00            2.638237e+00            1.153440e+00            1.067692e+00            1.153591e+00            1.168394e+00            1.104737e+00            1.373317e+00            1.134552e+00            1.132978e+00            
16     5.913093e+02            4.366093e+03            5.917526e+02            5.510813e+02            5.862259e+02            4.090766e+02            3.509725e+02            4.930985e+02            7.755214e-04            5.566571e+02            5.619440e+02            
17     3.120103e+00            3.063473e+05            6.068161e+01            3.120104e+00            3.120103e+00            3.120103e+00            3.120103e+00            3.120103e+00            3.127259e+00            3.120103e+00            3.120103e+00            
18     5.403979e+00            9.440999e+01            2.798763e+01            1.439608e+01            1.047351e+01            1.226536e+01            6.761282e+00            1.012487e+01            2.777162e+00            3.804307e+00            8.449304e+00            
19     3.151107e+01            3.705516e+02            4.385964e+01            3.065927e+01            2.510932e+01            3.175885e+01            4.012831e+01            2.759705e+01            6.139135e+00            3.906178e+01            2.827711e+01            
20     2.342560e+01            6.531090e+01            2.616947e+01            2.312257e+01            2.343769e+01            2.127797e+01            2.450688e+01            2.369599e+01            2.463661e+00            2.290809e+01            2.264145e+01            
21     4.505597e+00            6.070428e+00            4.407695e+00            4.406727e+00            4.497626e+00            4.536744e+00            4.499349e+00            4.530118e+00            4.180422e+00            4.496940e+00            4.484239e+00            
22     2.319560e+00            2.102479e+01            2.508796e+00            2.538915e+00            2.495413e+00            2.245006e+00            2.236801e+00            2.395605e+00            2.315699e+00            2.352219e+00            2.336889e+00            
23     1.906595e+01            1.059602e+02            2.162860e+01            2.002861e+01            2.128672e+01            1.820884e+01            1.156046e+01            2.015015e+01            1.837929e+01            1.935078e+01            2.030620e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_04_idea_0.py  (error=1.023118e-07)
Task  1: variant_06_idea_0.py  (error=2.759677e-07)
Task  2: original.py  (error=2.424825e-05)
Task  3: variant_02_idea_0.py  (error=2.822094e-04)
Task  4: variant_10_idea_0.py  (error=2.468596e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_01_idea_0.py  (error=3.065343e-07)
Task  7: variant_02_idea_0.py  (error=1.359036e-05)
Task  8: variant_02_idea_0.py  (error=1.852149e-03)
Task  9: variant_01_idea_0.py  (error=4.316223e-04)
Task 10: variant_08_idea_0.py  (error=7.649691e-04)
Task 11: variant_08_idea_0.py  (error=3.501453e-03)
Task 12: variant_08_idea_0.py  (error=2.521045e-01)
Task 13: variant_08_idea_0.py  (error=7.678428e-02)
Task 14: variant_08_idea_0.py  (error=2.610327e+00)
Task 15: variant_04_idea_0.py  (error=1.067692e+00)
Task 16: variant_08_idea_0.py  (error=7.755214e-04)
Task 17: variant_06_idea_0.py  (error=3.120103e+00)
Task 18: variant_08_idea_0.py  (error=2.777162e+00)
Task 19: variant_08_idea_0.py  (error=6.139135e+00)
Task 20: variant_08_idea_0.py  (error=2.463661e+00)
Task 21: variant_08_idea_0.py  (error=4.180422e+00)
Task 22: variant_06_idea_0.py  (error=2.236801e+00)
Task 23: variant_06_idea_0.py  (error=1.156046e+01)

WIN COUNTS:
  variant_08_idea_0.py: 10 wins
  variant_06_idea_0.py: 4 wins
  variant_02_idea_0.py: 3 wins
  variant_04_idea_0.py: 2 wins
  variant_01_idea_0.py: 2 wins
  original.py: 1 wins
  variant_10_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _update_evolution_paths(self, old_mean):
            """
            Update evolution paths with smooth adaptive damping instead of hard threshold.
            """
            # Initialize damping tracking if first call
            if not hasattr(self, '_smooth_damping'):
                self._smooth_damping = 1.0

            mean_shift = (self.mean - old_mean) / self.sigma
            transformed = self.invsqrtC @ mean_shift

            # Expected norm for random normal vector (chi distribution approximation)
            expected_norm = self.chi_n

            # Update step-size path with adaptive damping
            c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
            p_sigma_norm = np.linalg.norm(self.p_sigma)

            # Continuous damping: ratio of actual to expected norm
            norm_ratio = p_sigma_norm / max(expected_norm, 1e-20)
            # Smooth damping factor that reduces update when path is too long
            self._smooth_damping = np.clip(norm_ratio / 1.4, 0.1, 1.0)

            self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                           np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * \
                           self._smooth_damping * transformed

            # Update covariance path with same smooth damping
            p_c_norm = np.linalg.norm(self.p_c)
            c_c_complement = np.sqrt(1.0 - (1.0 - self.c_c) ** (2 * (self.generation + 1)))
            expected_c_norm = self.dim ** 0.5 * c_c_complement

            # Continuous damping for p_c based on its norm ratio
            c_damping = np.clip((p_c_norm / max(expected_c_norm, 1e-20)) / 1.4, 0.1, 1.0)

            self.p_c = (1.0 - self.c_c) * self.p_c + \
                       np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * \
                       c_damping * mean_shift

            # Return continuous value for downstream use
            return self._smooth_damping
```

# --- From variant_02_idea_0.py (3 wins) ---
```python
def _update_evolution_paths(self, old_mean):
            """
            Update evolution paths with eigenspace-conditioned adaptation.

            Key changes from standard CMA-ES:
            1. Mean shift projected into eigenspace and scaled by 1/sqrt(eigenvalue)
               to equalize contribution across all eigendirections
            2. Adaptive damping based on covariance matrix condition number
            3. Smoothed Heaviside for continuous p_c updates
            """
            mean_shift = (self.mean - old_mean) / self.sigma

            # Project mean_shift into eigenspace: y = B.T @ mean_shift
            y = self.B.T @ mean_shift

            # Condition the shift by inverse sqrt of eigenvalues
            # This ensures equal exploration across all eigendirections
            # preventing under-exploration in minor directions (key for Tasks 16,19,20,23)
            d_safe = np.maximum(self.D, 1e-10)
            y_conditioned = y / np.sqrt(d_safe)

            # Transform back to original space
            transformed = self.B @ y_conditioned

            # Adaptive condition number damping
            # Prevents instability when eigenvalues are highly skewed
            cond_C = np.max(d_safe) / np.min(d_safe)
            damping = np.sqrt(np.log(cond_C + 1.0) + 1.0)

            # Update step-size path
            c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
            self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                           np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

            # Smoothed Heaviside function (continuous instead of hard threshold)
            p_sigma_norm = np.linalg.norm(self.p_sigma)
            threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
            h_sigma = 1.0 / (1.0 + np.exp(-2.0 * (threshold - p_sigma_norm) / threshold))

            # Update covariance path with adaptive damping
            self.p_c = (1.0 - self.c_c) * self.p_c + \
                       h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff / damping) * mean_shift

            return h_sigma
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _update_evolution_paths(self, old_mean):
            """
            Update the cumulative step-size path (p_sigma) and the covariance path (p_c).
            Uses continuous adaptation instead of hard Heaviside thresholding.
            """
            mean_shift = (self.mean - old_mean) / self.sigma
            transformed = self.invsqrtC @ mean_shift

            # Update step-size path with continuous adaptation
            p_sigma_norm = np.linalg.norm(self.p_sigma)
            mean_shift_norm = np.linalg.norm(mean_shift)

            # Continuous adaptation factor based on ratio of mean shift to path length
            # This adapts faster when escaping local optima vs. fine-tuning
            c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
            self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                           np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

            # Continuous h_sigma: continuous weight from 0 to 1 based on normalized norm
            # Replaces hard threshold (1.4 + 2/(dim+1)) * chi_n * c_s_complement
            expected_norm = self.chi_n * c_s_complement
            norm_ratio = p_sigma_norm / (expected_norm + 1e-20)
            h_sigma = np.tanh(norm_ratio * 0.5)  # Smooth transition, 0 when small, 1 when large

            # Update covariance path with continuous damping
            # The correction term also becomes continuous
            delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

            self.p_c = (1.0 - self.c_c) * self.p_c + \
                       h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

            return h_sigma
```

# --- From variant_06_idea_0.py (4 wins) ---
```python
def _update_evolution_paths(self, old_mean):
        """
        Update evolution paths with fitness-guided adaptive scaling.
        The path momentum is modulated by recent fitness improvement.
        """
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Track smoothed fitness for improvement detection
        if not hasattr(self, 'smoothed_fitness'):
            self.smoothed_fitness = self.best_f
        self.smoothed_fitness = 0.9 * self.smoothed_fitness + 0.1 * self.best_f

        # Compute normalized fitness improvement
        if self.last_best_f < np.inf and self.last_best_f > 1e-30:
            improvement = (self.last_best_f - self.best_f) / (abs(self.last_best_f) + 1e-30)
            improvement = max(-5.0, min(5.0, improvement))
            improvement_factor = 1.0 + 0.5 * improvement
            improvement_factor = max(0.1, min(3.0, improvement_factor))
        else:
            improvement_factor = 1.0

        # Update step-size path with adaptive scaling
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        base_update = np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + improvement_factor * base_update

        # Heaviside function with chi_n-based threshold
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # Update covariance path with adaptive scaling
        base_cov_update = np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * improvement_factor * base_cov_update

        return h_sigma
```

# --- From variant_08_idea_0.py (10 wins) ---
```python
def _update_evolution_paths(self, old_mean):
        """
        Update evolution paths with adaptive momentum and progressive exploration.
        """
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Track normalized mean shift magnitude for adaptive momentum
        shift_norm = np.linalg.norm(mean_shift)
        expected_norm = np.sqrt(self.dim) * self.chi_n
        momentum_boost = max(1.0, min(shift_norm / (expected_norm + 1e-20), 5.0))

        # Progressive exploration: increase adaptation strength over generations
        gen_progress = min(1.0, self.generation / max(1, 200))
        progressive_scale = 1.0 + 1.5 * gen_progress

        # Adaptive damping based on step-size health
        sigma_ratio = self.sigma / max(self.min_sigma, 1e-10)
        adaptive_damping = min(2.0, max(0.5, sigma_ratio))

        # Update step-size path with adaptive momentum
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        sigma_term = np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * momentum_boost * progressive_scale
        self.p_sigma = (1.0 - self.c_sigma / adaptive_damping) * self.p_sigma + sigma_term * transformed

        # Heaviside function for p_c update
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # Update covariance path with adaptive momentum
        cc_term = np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * momentum_boost * progressive_scale
        self.p_c = (1.0 - self.c_c / adaptive_damping) * self.p_c + h_sigma * cc_term * mean_shift

        return h_sigma
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _update_evolution_paths(self, old_mean):
        """
        Update evolution paths with adaptive damping and negative feedback.
        Uses divergence-aware learning rates and counter-steering when paths grow too long.
        """
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Compute convergence indicator
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        chi_n = self.chi_n
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * chi_n * c_s_complement

        # Adaptive damping: detect divergence vs stagnation
        div_ratio = p_sigma_norm / (threshold + 1e-20)

        if div_ratio > 5.0:
            # Severe divergence - use strong damping and negative feedback
            adaptive_c_sigma = self.c_sigma * 0.2
            neg_weight = 0.6
        elif div_ratio > 2.5:
            # Moderate divergence - moderate damping
            adaptive_c_sigma = self.c_sigma * 0.5
            neg_weight = 0.3
        elif div_ratio < 0.3:
            # Stagnation - increase learning rate to escape
            adaptive_c_sigma = min(self.c_sigma * 3.0, 0.5)
            neg_weight = 0.0
        else:
            # Normal operation
            adaptive_c_sigma = self.c_sigma
            neg_weight = 0.0

        # Update step-size path with adaptive learning
        self.p_sigma = (1.0 - adaptive_c_sigma) * self.p_sigma + \
                       np.sqrt(adaptive_c_sigma * (2.0 - adaptive_c_sigma) * self.mu_eff) * transformed

        # Negative feedback: push back when diverging to break local optima cycles
        if neg_weight > 0.0:
            self.p_sigma -= neg_weight * np.sqrt(adaptive_c_sigma * (2.0 - adaptive_c_sigma) * self.mu_eff) * transformed

        # Smooth h_sigma with gradual transition instead of hard Heaviside
        h_sigma = 1.0 if div_ratio < 1.0 else max(0.0, 1.0 - (div_ratio - 1.0) / 4.0)

        # Adaptive covariance path learning rate
        if div_ratio > 3.0:
            adaptive_c_c = self.c_c * 0.4
        elif div_ratio < 0.3:
            adaptive_c_c = min(self.c_c * 2.0, 0.2)
        else:
            adaptive_c_c = self.c_c

        # Update covariance path
        self.p_c = (1.0 - adaptive_c_c) * self.p_c + \
                   h_sigma * np.sqrt(adaptive_c_c * (2.0 - adaptive_c_c) * self.mu_eff) * mean_shift

        return h_sigma
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