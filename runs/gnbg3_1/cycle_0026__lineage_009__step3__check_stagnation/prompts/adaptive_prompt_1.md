Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_check_stagnation` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      8.933761e-08            -inf                    4.414046e-07            3.186966e-06            8.343378e-08            1.171222e-07            1.742719e-05            1.458248e-05            1.810124e-05            1.118616e-07            1.282253e-05            
1      1.726985e-07            -inf                    4.494073e-07            2.095489e-06            1.696057e-07            2.147250e-07            1.262626e-05            7.324211e-06            1.569161e-05            2.308244e-07            7.869437e-06            
2      5.423417e-05            -inf                    3.686599e-05            1.104690e-04            2.434222e-05            3.126601e-05            9.483366e-05            2.777984e-04            4.931680e-05            2.817708e-05            4.251742e-05            
3      3.203505e-04            -inf                    3.450469e-04            5.825286e-04            3.267755e-04            1.584076e-03            9.330653e-04            5.114194e-04            3.378740e-04            3.608881e-04            2.386768e-03            
4      2.320395e-02            -inf                    2.293556e-02            2.373037e-02            2.280258e-02            2.424473e-02            2.302126e-02            2.351685e-02            2.356523e-02            2.514693e-02            2.402556e-02            
5      1.000000e-08            -inf                    1.000000e-08            1.595812e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            3.078324e-08            
6      4.315395e-07            -inf                    6.080873e-06            1.228232e-04            4.150925e-07            7.880622e-07            1.058698e-04            1.146222e-04            7.536959e-05            5.506158e-07            6.151531e-05            
7      1.253606e-05            -inf                    1.459667e-05            1.980732e-05            1.402030e-05            1.541367e-05            5.730602e-05            4.135878e-05            5.735157e-05            1.470194e-05            3.927372e-05            
8      1.799495e-03            -inf                    1.826972e-03            1.981665e-03            1.876405e-03            1.920660e-03            1.969599e-03            2.090243e-03            1.907307e-03            1.912051e-03            2.096345e-03            
9      4.399769e-04            -inf                    5.138180e-04            6.816826e-04            6.200405e-04            5.099816e-04            6.255574e-04            6.913819e-04            5.885973e-04            5.259515e-04            1.362986e-03            
10     3.936457e+01            -inf                    2.136131e+01            1.653739e+01            4.408826e+01            3.885048e+01            2.017541e+01            1.064016e+01            1.436475e+01            3.847339e+01            1.415128e+01            
11     4.228558e+01            -inf                    1.550883e+01            1.663861e+01            5.564952e+01            5.545170e+01            1.596372e+01            1.742092e+01            7.343174e+00            6.601486e+01            1.410364e+01            
12     3.063783e+01            -inf                    1.644247e+01            1.361151e+01            3.129859e+01            2.788051e+01            1.773045e+01            1.851268e+01            1.662605e+01            4.184914e+01            1.117486e+01            
13     4.538411e+00            -inf                    2.297355e+00            2.434899e+00            4.358924e+00            6.298446e+00            2.322207e+00            1.692107e+00            2.028785e+00            1.140438e+01            2.528929e+00            
14     2.676015e+00            -inf                    2.663209e+00            2.714986e+00            2.857827e+00            2.747978e+00            2.663550e+00            2.765358e+00            2.586153e+00            3.036948e+00            2.657740e+00            
15     3.141322e+00            -inf                    2.801932e+00            2.402373e+00            2.823397e+00            2.866200e+00            2.641537e+00            2.523548e+00            2.776961e+00            2.993562e+00            2.488404e+00            
16     7.471930e+01            -inf                    7.682886e+01            5.887098e+01            6.163678e+01            1.193152e+02            6.670102e+01            7.228130e+01            8.400174e+01            2.205485e+02            6.062854e+01            
17     5.387587e+02            -inf                    1.059719e+02            2.842673e+02            1.689583e+03            2.322467e+03            2.320154e+02            9.004386e+01            4.504600e+00            3.373457e+03            8.173242e+01            
18     1.516647e+01            -inf                    2.052299e+01            1.577635e+01            2.084613e+01            2.222773e+01            1.692343e+01            1.467410e+01            1.643689e+01            3.059551e+01            1.576747e+01            
19     3.034549e+01            -inf                    1.114316e+01            1.279585e+01            1.626744e+01            1.188786e+01            1.234949e+01            1.430687e+01            9.212902e+00            3.802775e+01            1.415928e+01            
20     2.204231e+01            -inf                    1.942516e+01            1.944468e+01            2.370482e+01            2.154730e+01            1.950806e+01            1.898961e+01            1.915552e+01            2.644424e+01            1.693917e+01            
21     4.490789e+00            -inf                    4.282116e+00            4.460937e+00            4.556617e+00            4.464805e+00            4.475253e+00            4.479567e+00            4.397185e+00            4.651797e+00            4.326558e+00            
22     1.001211e+01            -inf                    1.007684e+01            3.474981e+00            8.403926e+00            8.243089e+00            8.083063e+00            7.594701e+00            5.296513e+00            9.247117e+00            8.676983e+00            
23     2.245931e+01            -inf                    1.953465e+01            2.220301e+01            1.500595e+01            2.196334e+01            1.911095e+01            2.002937e+01            1.850464e+01            1.984456e+01            1.810019e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_04_idea_0.py  (error=8.343378e-08)
Task  1: variant_04_idea_0.py  (error=1.696057e-07)
Task  2: variant_04_idea_0.py  (error=2.434222e-05)
Task  3: original.py  (error=3.203505e-04)
Task  4: variant_04_idea_0.py  (error=2.280258e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_04_idea_0.py  (error=4.150925e-07)
Task  7: original.py  (error=1.253606e-05)
Task  8: original.py  (error=1.799495e-03)
Task  9: original.py  (error=4.399769e-04)
Task 10: variant_07_idea_0.py  (error=1.064016e+01)
Task 11: variant_08_idea_0.py  (error=7.343174e+00)
Task 12: variant_10_idea_0.py  (error=1.117486e+01)
Task 13: variant_07_idea_0.py  (error=1.692107e+00)
Task 14: variant_08_idea_0.py  (error=2.586153e+00)
Task 15: variant_03_idea_0.py  (error=2.402373e+00)
Task 16: variant_03_idea_0.py  (error=5.887098e+01)
Task 17: variant_08_idea_0.py  (error=4.504600e+00)
Task 18: variant_07_idea_0.py  (error=1.467410e+01)
Task 19: variant_08_idea_0.py  (error=9.212902e+00)
Task 20: variant_10_idea_0.py  (error=1.693917e+01)
Task 21: variant_02_idea_0.py  (error=4.282116e+00)
Task 22: variant_03_idea_0.py  (error=3.474981e+00)
Task 23: variant_04_idea_0.py  (error=1.500595e+01)

WIN COUNTS:
  variant_04_idea_0.py: 6 wins
  original.py: 4 wins
  variant_08_idea_0.py: 4 wins
  variant_07_idea_0.py: 3 wins
  variant_03_idea_0.py: 3 wins
  variant_10_idea_0.py: 2 wins
  variant_02_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (1 wins) ---
```python
def _check_stagnation(self):
        """Track stagnation counter with scale-adaptive threshold.

        Key insight: The original fixed threshold (1e-12) is orders of magnitude
        too small for high-error tasks (error ~10-500). For Task 17 (error ~538),
        even a 0.001 improvement is meaningful, but 1e-12 is numerically invisible.

        This version uses: threshold = max(1e-6 * |f_opt|, 1e-10)
        - High-error tasks (|f_opt|~100): threshold ~1e-4 → proper stagnation detection
        - Low-error tasks (|f_opt|~1e-8): threshold ~1e-10 → fine-grained detection
        """
        improvement = self.f_opt_prev - self.f_opt

        # Adaptive threshold: scale with problem difficulty
        base_scale = max(abs(self.f_opt), 1.0)
        rel_threshold = 1e-6 * base_scale
        threshold = max(rel_threshold, 1e-10)

        if improvement > threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Scale max_stagnation with problem difficulty for harder tasks
        if abs(self.f_opt) > 10.0:
            self.max_stagnation = 150 + self.dim * 5
        elif abs(self.f_opt) > 1.0:
            self.max_stagnation = 100 + self.dim * 4
        else:
            self.max_stagnation = 50 + self.dim * 3

        self.f_opt_prev = self.f_opt
```

# --- From variant_03_idea_0.py (3 wins) ---
```python
def _check_stagnation(self):
        """Track stagnation using momentum-based improvement and covariance condition monitoring."""
        if not hasattr(self, 'stagnation_momentum'):
            self.stagnation_momentum = 0.0
            self.fit_history = []
            self.cond_history = []

        # Track fitness history for momentum calculation
        self.fit_history.append(self.f_opt)
        if len(self.fit_history) > 20:
            self.fit_history.pop(0)

        # Track covariance condition number history
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        cond = np.max(eigvals) / np.min(eigvals)
        self.cond_history.append(cond)
        if len(self.cond_history) > 20:
            self.cond_history.pop(0)

        # Compute relative improvement momentum
        if len(self.fit_history) >= 5:
            recent_window = self.fit_history[-5:]
            early_window = self.fit_history[:5] if len(self.fit_history) >= 5 else self.fit_history

            # Relative improvement: how much did fitness improve relative to scale?
            scale = max(abs(self.f_opt), 1.0, abs(self.fit_history[0]))
            recent_improvement = (np.mean(early_window) - np.mean(recent_window)) / scale
            recent_improvement = max(recent_improvement, 0.0)

            # Momentum: EMA of improvement rate
            self.stagnation_momentum = 0.7 * self.stagnation_momentum + 0.3 * recent_improvement
        else:
            self.stagnation_momentum = 0.0

        # Stagnation detected if: no momentum AND condition is problematic
        momentum_threshold = 1e-6
        cond_healthy = np.mean(self.cond_history) < 1e7 if self.cond_history else True

        if self.stagnation_momentum < momentum_threshold and not cond_healthy:
            self.stagnation_counter += 3  # Accelerated stagnation on ill-conditioned state
        elif self.stagnation_momentum < momentum_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        self.f_opt_prev = self.f_opt
```

# --- From variant_04_idea_0.py (6 wins) ---
```python
def _check_stagnation(self):
        """Multi-dimensional stagnation detection with diversity and condition awareness."""
        # Track fitness improvement
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
            self.stagnation_start_f = self.f_opt
        else:
            self.stagnation_counter += 1

        self.f_opt_prev = self.f_opt

        # Initialize stagnation tracking
        if not hasattr(self, 'stagnation_start_f'):
            self.stagnation_start_f = self.f_opt
        if not hasattr(self, 'diversity_stagnation_counter'):
            self.diversity_stagnation_counter = 0
        if not hasattr(self, 'condition_stagnation_counter'):
            self.condition_stagnation_counter = 0

        # Compute current diversity and condition
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        current_diversity = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        current_diversity = np.clip(current_diversity, 0.0, 2.0)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        current_cond = eig_max / max(eig_min, 1e-15)

        # Track diversity stagnation
        if current_diversity < 0.05:
            self.diversity_stagnation_counter += 1
        else:
            self.diversity_stagnation_counter = max(0, self.diversity_stagnation_counter - 1)

        # Track condition stagnation
        if current_cond > 1e6:
            self.condition_stagnation_counter += 1
        else:
            self.condition_stagnation_counter = max(0, self.condition_stagnation_counter - 1)

        # Compute relative improvement rate (for detecting slow progress on multimodal tasks)
        f_range = abs(self.stagnation_start_f) + 1e-10
        relative_stagnation = abs(self.f_opt - self.stagnation_start_f) / f_range

        # Store metrics for restart logic (accessed by _restart_if_needed)
        self.relative_stagnation = relative_stagnation
        self.current_diversity = current_diversity
        self.current_condition = current_cond
```

# --- From variant_07_idea_0.py (3 wins) ---
```python
def _check_stagnation(self):
        """Multi-signal stagnation detection with covariance health monitoring."""
        # Signal 1: Improvement-based stagnation (adaptive threshold)
        fitness_scale = max(abs(self.f_opt), 1.0)
        adaptive_threshold = 1e-6 * fitness_scale
        if self.f_opt < self.f_opt_prev - adaptive_threshold:
            self.stagnation_counter = 0
            if hasattr(self, 'improvement_streak'):
                self.improvement_streak += 1
            else:
                self.improvement_streak = 1
        else:
            self.stagnation_counter += 1
            if hasattr(self, 'improvement_streak'):
                self.improvement_streak = 0

        # Signal 2: Covariance health (detect ill-conditioning before collapse)
        try:
            eigvals = np.linalg.eigvalsh(self.C)
            eigvals = np.clip(eigvals, 1e-15, None)
            cond = np.max(eigvals) / np.min(eigvals)
            cond_stagnation = cond > 1e6
        except:
            cond_stagnation = True

        # Signal 3: Step-size stagnation (sigma not changing)
        if hasattr(self, 'prev_sigma'):
            sigma_ratio = self.sigma / max(self.prev_sigma, 1e-15)
            sigma_stagnation = 0.95 < sigma_ratio < 1.05
        else:
            sigma_stagnation = False
        self.prev_sigma = self.sigma

        # Signal 4: Diversity collapse (population converging)
        pop_spread = np.mean(np.std(self.population, axis=0))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_stagnation = pop_spread < 0.1 * expected_spread

        # Signal 5: Fitness plateau (no improvement in many generations)
        plateau_stagnation = self.stagnation_counter > self.max_stagnation // 3

        # Combined stagnation score (any strong signal triggers restart consideration)
        combined_stagnation = (
            self.stagnation_counter > self.max_stagnation // 2 or
            cond_stagnation or
            (sigma_stagnation and plateau_stagnation) or
            diversity_stagnation
        )

        # If multiple mild signals, also consider stagnant
        mild_signals = int(plateau_stagnation) + int(cond_stagnation) + int(sigma_stagnation)
        if mild_signals >= 2:
            combined_stagnation = True

        # Update f_opt_prev
        self.f_opt_prev = self.f_opt

        # Trigger aggressive restart check in _restart_if_needed by setting flag
        if combined_stagnation and not hasattr(self, 'aggressive_restart_pending'):
            self.aggressive_restart_pending = True
        elif not combined_stagnation and hasattr(self, 'aggressive_restart_pending'):
            self.aggressive_restart_pending = False
```

# --- From variant_08_idea_0.py (4 wins) ---
```python
def _check_stagnation(self):
        """Track stagnation using adaptive relative threshold and diversity signal."""
        # Adaptive threshold: scale to problem difficulty
        # On high-error tasks (error ~90), require proportionally larger improvements
        # On low-error tasks (error ~1e-08), use tighter threshold
        scale = max(1.0, abs(self.f_opt))
        threshold = max(1e-12, 1e-6 * scale)

        # Primary signal: best fitness improvement
        improved = self.f_opt < self.f_opt_prev - threshold

        # Secondary signal: population diversity (avoid restarting when still exploring)
        if hasattr(self, 'population') and len(self.population) > 1:
            pop_diffs = self.population - self.mean
            max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
            expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
            spread = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
            # Still exploring if population has meaningful spread
            still_exploring = spread > 0.05
        else:
            still_exploring = False

        # Reset stagnation if improved OR still exploring (don't restart during exploration)
        if improved or still_exploring:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.f_opt_prev = self.f_opt
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _check_stagnation(self):
        """Multi-signal stagnation detection with adaptive thresholds and failure mode recognition."""
        # Use relative improvement threshold scaled to current fitness magnitude
        # This adapts to problem difficulty (harder problems get larger thresholds)
        rel_threshold = 1e-6 * max(1.0, abs(self.f_opt))
        actual_threshold = max(rel_threshold, 1e-12)

        if self.f_opt < self.f_opt_prev - actual_threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

            # Detect fitness collapse: if population fitness range is tiny relative to best,
            # we have converged to a local optimum and need to restart NOW
            if len(self.fitness) >= 2:
                fitness_range = np.max(self.fitness) - np.min(self.fitness)
                # Scale threshold by problem difficulty indicator
                if fitness_range < 1e-8 * max(1.0, abs(self.f_opt)):
                    self.stagnation_counter += 2  # Accelerate restart

            # Detect covariance collapse: ill-conditioned C means we've lost exploration ability
            try:
                eigvals = np.linalg.eigvalsh(self.C)
                eig_min = np.min(eigvals)
                eig_max = np.max(eigvals)
                if eig_max > 0 and eig_min / eig_max < 1e-10:
                    self.stagnation_counter += 2  # Accelerate restart on rank deficiency
            except:
                pass

        self.f_opt_prev = self.f_opt
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


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Sobol quasi-random sequences."""
        # Generate Sobol sequence samples (quasi-random, low-discrepancy)
        # Falls back to stratified sampling if Sobol unavailable
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            # Map from [0,1]^dim to [lb, ub]
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            # Fallback: stratified sampling with jitter
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            # Shuffle each dimension independently
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        # More efficient than eigendecomposition for sampling
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        # Sample from standard normal and transform
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivors_batch(self):
        """Select survivors via elitist (mu, lambda)-selection."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = self.fitness[i] - self.trial_fitness[i]
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Initialize cumulative tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay to accumulated reward (recent rewards weighted more)
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Log-scaled improvement reward (more sensitive to small improvements)
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Explicit stagnation detection
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Exploration bonus when stagnant and diversity is low
        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        # Update cumulative weighted reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize by decay sum to get comparable reward values
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_08()
        else:
            self._adapt_covariance_variant_09()
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_01(self):
        """Archive-guided covariance perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Initialize archive for tracking distinct best solutions
        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        # Add current best to archive if distinct enough
        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for arch_pos in self.archive_positions:
            dist = np.linalg.norm(self.x_opt - arch_pos)
            if dist < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.archive_positions) < 10:
            self.archive_positions.append(self.x_opt.copy())
            self.archive_fitness.append(self.f_opt)
            self.archive_generations.append(self.generation)
        elif is_distinct and len(self.archive_positions) >= 10:
            # Replace worst entry
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        # Compute diversity and condition metrics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        # Compute archive-based escape direction
        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            # Direction from mean toward archive centroid, weighted by diversity
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                # Also consider directions to individual archive members
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        # Perturb along diverse directions
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            # Add global exploration along principal axes
            escape_perturb += 0.1 * np.eye(self.dim)

        # Adaptive learning rates with exploration boost when trapped
        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with archive-based perturbation
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Inject escape perturbation when trapped
        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        # Full restart on extreme ill-conditioning
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_06(self):
        """Diversity-sensitive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
        
        if is_converging:
            scale = 10.0
        else:
            scale = 1.0 + 5.0 * min(rel_var, 0.1)
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Evolution path update (unchanged)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with positive/negative decomposition
        rank_mu_pos = np.zeros((self.dim, self.dim))
        rank_mu_neg = np.zeros((self.dim, self.dim))

        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            weight = self.weights[i]
            outer = np.outer(diff, diff)

            if weight > 0:
                rank_mu_pos += weight * outer
            else:
                rank_mu_neg += abs(weight) * outer

        # Normalize by sum of positive and negative weights separately
        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        # Combine positive and negative updates (active CMA-ES core)
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking for escaping local optima."""
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt - self.f_opt
        self.prev_f_opt = self.f_opt

        fitness_gradient = max(abs(delta_f_opt), 1e-15)
        temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
        self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * self.exploration_temp
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute diversity using percentile-robust and condition-aware metrics.

        Key differences from ESS-based approach:
        - Uses MAX-normalized spread (catches catastrophic collapse better)
        - Uses fitness percentile range (detects local optima trapping)
        - Uses condition number directly (detects rank deficiency)
        - Uses population spread ratio (detects severe elongation)

        This makes the metric more sensitive to extreme states that cause
        catastrophic failure on multimodal/ill-conditioned tasks.
        """
        # Max-normalized spread: captures catastrophic collapse better than trace
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        spread_norm = np.clip(spread_norm, 0.0, 2.0)

        # Fitness percentile range: detects local optima trapping
        sorted_fit = np.sort(self.fitness)
        fit_range = sorted_fit[-1] - sorted_fit[0]
        fit_median = sorted_fit[len(sorted_fit) // 2]
        fit_p10 = sorted_fit[max(0, len(sorted_fit) // 10 - 1)]
        fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
        fit_percentile_range = np.clip(fit_percentile_range, 0.0, 1.0)

        # Condition number: directly detects rank deficiency
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        cond_norm = 1.0 / np.log1p(cond)
        cond_norm = np.clip(cond_norm, 0.0, 1.0)

        # Spread ratio: detects severe elongation (ratio of max to min axis lengths)
        diag = np.diag(self.C)
        diag = np.maximum(diag, 1e-15)
        max_var = np.max(diag)
        min_var = np.min(diag)
        spread_ratio = min_var / max_var
        spread_ratio = np.clip(spread_ratio, 0.0, 1.0)

        # Weighted combination: emphasize condition and spread ratio for ill-conditioned tasks
        diversity = (
            0.25 * spread_norm +
            0.35 * fit_percentile_range +
            0.25 * cond_norm +
            0.15 * spread_ratio
        )

        return float(np.clip(diversity, 1e-15, None))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0

```