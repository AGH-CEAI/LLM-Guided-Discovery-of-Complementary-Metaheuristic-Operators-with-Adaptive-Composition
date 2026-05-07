Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_step_size` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.295391e-02            -inf                    6.097524e-02            -inf                    5.403108e-02            7.790306e-02            7.212959e-02            6.359921e-02            5.184540e-02            5.654968e-02            -inf                    
1      4.461412e-02            -inf                    4.822814e-02            -inf                    4.875933e-02            1.541634e-01            1.963313e-01            8.671633e-02            4.407292e-02            4.476627e-02            -inf                    
2      1.937323e-01            -inf                    3.663605e-01            -inf                    7.974538e-01            4.535180e+00            2.470977e+00            3.367871e+00            5.858323e-01            2.699199e+00            -inf                    
3      7.926602e-01            -inf                    6.175728e-01            -inf                    6.048507e-01            3.367647e+00            3.138047e+00            2.802956e+00            1.695787e+00            4.482400e-01            -inf                    
4      5.316473e-01            -inf                    5.543732e-01            -inf                    5.559774e-01            9.964251e-01            1.028583e+00            7.074641e-01            5.295858e-01            8.411093e-01            -inf                    
5      3.900719e-04            -inf                    2.471466e-03            -inf                    1.508717e-04            9.834038e+00            6.704788e+00            1.348718e-01            6.980811e-04            2.171323e-01            -inf                    
6      1.760676e+01            -inf                    9.457141e+00            -inf                    1.274116e+01            2.556132e+01            1.057654e+01            2.522832e+01            2.173405e+01            2.227961e+01            -inf                    
7      1.347344e-01            -inf                    1.481494e-01            -inf                    1.364861e-01            2.223190e+00            1.293624e+00            4.403268e-01            1.250501e-01            6.533746e-01            -inf                    
8      4.863669e-01            -inf                    1.350900e+00            -inf                    5.480394e-01            2.306035e+00            2.241648e+00            1.614425e+00            2.856891e-01            1.348476e+00            -inf                    
9      3.833155e+00            -inf                    5.581337e+00            -inf                    5.549830e+00            6.992838e+00            5.442893e+00            7.398213e+00            4.297304e+00            5.249948e+00            -inf                    
10     3.019513e+01            -inf                    2.229084e+01            -inf                    2.936808e+01            2.339451e+01            1.606741e+01            8.824809e+01            2.877493e+01            3.392040e+01            -inf                    
11     4.066815e+01            -inf                    1.443563e+01            -inf                    4.930664e+01            5.665091e+01            4.923594e+01            3.120917e+02            3.506120e+01            5.425378e+01            -inf                    
12     2.716651e+01            -inf                    1.709311e+01            -inf                    3.034728e+01            4.135334e+01            2.133822e+01            7.704045e+01            2.643860e+01            3.239569e+01            -inf                    
13     4.800987e+00            -inf                    3.109477e+00            -inf                    5.157537e+00            3.338600e+00            4.190274e+00            2.245205e+01            4.017866e+00            9.504327e+00            -inf                    
14     2.778819e+00            -inf                    2.684433e+00            -inf                    2.662104e+00            2.806305e+00            2.746297e+00            3.565223e+00            2.641928e+00            3.168903e+00            -inf                    
15     2.777393e+00            -inf                    2.671280e+00            -inf                    2.784509e+00            2.980715e+00            2.437473e+00            3.418266e+00            2.755245e+00            3.021042e+00            -inf                    
16     1.563362e+02            -inf                    1.107814e+02            -inf                    1.466506e+02            7.351753e+01            1.302880e+02            3.586525e+02            2.174078e+02            1.403386e+02            -inf                    
17     1.046338e+03            -inf                    2.511763e+02            -inf                    1.977499e+03            2.533642e+03            8.325550e+02            1.573095e+04            1.122812e+03            2.864311e+03            -inf                    
18     2.198914e+01            -inf                    2.080437e+01            -inf                    2.569198e+01            2.173510e+01            1.715886e+01            3.703448e+01            2.102765e+01            2.630018e+01            -inf                    
19     1.609313e+01            -inf                    1.775860e+01            -inf                    1.472069e+01            1.566726e+01            1.423545e+01            5.259274e+01            2.137228e+01            2.335278e+01            -inf                    
20     2.051589e+01            -inf                    2.240796e+01            -inf                    2.176927e+01            2.068074e+01            1.925302e+01            2.560619e+01            2.321259e+01            2.356406e+01            -inf                    
21     4.539340e+00            -inf                    4.396156e+00            -inf                    4.503758e+00            4.511185e+00            4.535153e+00            4.618550e+00            4.567649e+00            4.565536e+00            -inf                    
22     9.518951e+00            -inf                    8.466937e+00            -inf                    1.078838e+01            9.325067e+00            6.447091e+00            1.187047e+01            9.272049e+00            9.768174e+00            -inf                    
23     1.488969e+01            -inf                    1.934936e+01            -inf                    2.189862e+01            1.657824e+01            1.906992e+01            2.101955e+01            2.142185e+01            2.708037e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_08_idea_0.py  (error=5.184540e-02)
Task  1: variant_08_idea_0.py  (error=4.407292e-02)
Task  2: original.py  (error=1.937323e-01)
Task  3: variant_09_idea_0.py  (error=4.482400e-01)
Task  4: variant_08_idea_0.py  (error=5.295858e-01)
Task  5: variant_04_idea_0.py  (error=1.508717e-04)
Task  6: variant_02_idea_0.py  (error=9.457141e+00)
Task  7: variant_08_idea_0.py  (error=1.250501e-01)
Task  8: variant_08_idea_0.py  (error=2.856891e-01)
Task  9: original.py  (error=3.833155e+00)
Task 10: variant_06_idea_0.py  (error=1.606741e+01)
Task 11: variant_02_idea_0.py  (error=1.443563e+01)
Task 12: variant_02_idea_0.py  (error=1.709311e+01)
Task 13: variant_02_idea_0.py  (error=3.109477e+00)
Task 14: variant_08_idea_0.py  (error=2.641928e+00)
Task 15: variant_06_idea_0.py  (error=2.437473e+00)
Task 16: variant_05_idea_0.py  (error=7.351753e+01)
Task 17: variant_02_idea_0.py  (error=2.511763e+02)
Task 18: variant_06_idea_0.py  (error=1.715886e+01)
Task 19: variant_06_idea_0.py  (error=1.423545e+01)
Task 20: variant_06_idea_0.py  (error=1.925302e+01)
Task 21: variant_02_idea_0.py  (error=4.396156e+00)
Task 22: variant_06_idea_0.py  (error=6.447091e+00)
Task 23: original.py  (error=1.488969e+01)

WIN COUNTS:
  variant_08_idea_0.py: 6 wins
  variant_02_idea_0.py: 6 wins
  variant_06_idea_0.py: 6 wins
  original.py: 3 wins
  variant_09_idea_0.py: 1 wins
  variant_04_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_02_idea_0.py (6 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using short-term success history (no cumulation)."""
        recent_improved = 0
        recent_total = 0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation = np.clip(adaptation, -0.5, 0.5)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim))
        damping_adaptive = np.clip(damping_adaptive, 0.1, 100.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using multi-feedback signals with diversity-sensitive damping."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Signal 1: Evolution path (CSA baseline)
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        path_norm = np.linalg.norm(self.ps) / np.sqrt(self.dim)

        # Signal 2: Fitness gradient direction
        if hasattr(self, 'prev_f_opt'):
            f_improved = self.f_opt < self.f_opt_prev - 1e-12
            fit_grad = np.sign(self.f_opt_prev - self.f_opt) if hasattr(self, 'prev_f_opt') else 0.0
        else:
            f_improved = False
            fit_grad = 0.0

        # Signal 3: Diversity ratio for adaptive damping
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = pop_variance / (expected_var + 1e-10)
        diversity_ratio = np.clip(diversity_ratio, 1e-4, 10.0)

        # Stagnation detection
        stagnation = self.stagnation_counter > self.max_stagnation // 2

        # Adaptive damping: increase when diverse (explore more), decrease when converged (exploit)
        base_damping = self.damping
        if stagnation:
            adaptive_damping = base_damping * 0.3  # Lower damping = faster sigma growth
        elif diversity_ratio < 0.1:
            adaptive_damping = base_damping * 0.5  # Low diversity: encourage exploration
        elif diversity_ratio > 0.5:
            adaptive_damping = base_damping * 1.5  # High diversity: be more conservative
        else:
            adaptive_damping = base_damping

        # Combined step-size update with multiple feedback signals
        # Primary: evolution path feedback
        csa_term = (path_norm - 1.0) * self.cs / adaptive_damping

        # Secondary: fitness-based adjustment
        if f_improved:
            fitness_adjustment = -0.1 / adaptive_damping  # Small reward for improvement
        else:
            fitness_adjustment = 0.05 / adaptive_damping  # Small penalty for no improvement

        # Tertiary: diversity-based adjustment (prevent collapse or explosion)
        if diversity_ratio < 0.05:
            diversity_adjustment = 0.2 / adaptive_damping  # Prevent collapse
        elif diversity_ratio > 2.0:
            diversity_adjustment = -0.1 / adaptive_damping  # Slow down if too spread
        else:
            diversity_adjustment = 0.0

        # Combine all signals
        total_adjustment = csa_term + fitness_adjustment + diversity_adjustment

        # Apply with safeguards
        self.sigma *= np.exp(np.clip(total_adjustment, -2.0, 2.0))
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using success history and diversity-weighted control."""
        improvement = self.f_opt_prev - self.f_opt

        if not hasattr(self, 'success_history'):
            self.success_history = []
        if not hasattr(self, 'sigma_history'):
            self.sigma_history = []

        is_success = improvement > 0
        self.success_history.append(1.0 if is_success else 0.0)
        self.sigma_history.append(self.sigma)

        max_history = max(5, self.dim // 10 + 5)
        if len(self.success_history) > max_history:
            self.success_history.pop(0)
            self.sigma_history.pop(0)

        success_rate = np.mean(self.success_history)

        pop_spread = np.mean(np.std(self.population, axis=0))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_ratio = np.clip(pop_spread / (expected_spread + 1e-10), 0.1, 3.0)

        target_success = 0.25

        if success_rate > target_success:
            adjustment = 1.0 + 0.3 * (success_rate - target_success) / (1.0 - target_success + 1e-10)
        else:
            adjustment = 0.7 + 0.3 * success_rate / (target_success + 1e-10)

        if diversity_ratio < 0.3:
            adjustment *= 1.5
        elif diversity_ratio < 0.6:
            adjustment *= 1.2

        if len(self.sigma_history) >= 3:
            sigma_trend = self.sigma_history[-1] / (np.mean(self.sigma_history[-3:]) + 1e-10)
            if sigma_trend < 0.8:
                adjustment *= 1.1
            elif sigma_trend > 1.2:
                adjustment *= 0.9

        self.sigma *= adjustment
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_06_idea_0.py (6 wins) ---
```python
def _adapt_step_size(self):
            """Adapt step-size using explicit success-rate tracking (no CSA evolution path)."""
            if not hasattr(self, '_step_history'):
                self._step_history = []

            n_successful = sum(1 for f in self.trial_fitness if f < self.f_opt_prev)
            current_success_rate = n_successful / max(len(self.trial_fitness), 1)
            self._step_history.append(current_success_rate)

            if len(self._step_history) > 5:
                self._step_history.pop(0)

            smoothed_success = sum(self._step_history) / len(self._step_history)

            expected_success = 0.2
            success_diff = smoothed_success - expected_success

            damping_adjusted = self.damping * np.sqrt(self.dim)

            adapt_rate = self.cs * 1.5
            sigma_multiplier = np.exp(adapt_rate * success_diff / damping_adjusted)

            self.sigma *= sigma_multiplier

            pop_spread = np.mean(np.std(self.population, axis=0))
            if pop_spread > 1e-10:
                target_spread = (self.ub[0] - self.lb[0]) * 0.05
                if self.sigma * pop_spread > target_spread * 10:
                    self.sigma *= 0.9
                elif self.sigma * pop_spread < target_spread * 0.1:
                    self.sigma *= 1.1

            self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_08_idea_0.py (6 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using mirrored sampling variance ratio (fundamentally different from CSA)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Mirrored sampling: generate paired samples in opposite directions
        z_mirror = np.random.randn(self.dim)
        z_mirror = z_mirror / np.maximum(np.linalg.norm(z_mirror), 1e-10)

        # Compute variance ratio: current vs expected under isotropic normal
        z_var = np.var(z_mirror)
        expected_var = 1.0 / self.dim
        variance_ratio = np.clip(z_var / (expected_var + 1e-10), 0.1, 10.0)

        # Scale evolution path by variance ratio for adaptive search
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean * np.sqrt(variance_ratio)

        # Compute adaptive sigma change using combined norm and variance ratio
        norm_ps = np.linalg.norm(self.ps) / np.sqrt(self.dim)
        sigma_factor = np.exp((norm_ps - 1.0) * self.cs / self.damping * (1.0 / np.sqrt(variance_ratio) + 0.5 * (variance_ratio - 1.0)))

        self.sigma *= sigma_factor
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Multi-signal adaptive step-size with fitness landscape awareness."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Signal 1: Evolution path (standard CSA)
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        # Signal 2: Diversity-adjusted direction
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Signal 3: Fitness gradient
        fit_improvement = max(0.0, self.f_opt_prev - self.f_opt)
        fit_scale = max(abs(self.f_opt), 1.0)
        fit_gradient = fit_improvement / (fit_scale + 1e-10)

        # Weighted combination of signals
        w1, w2, w3 = 0.6, 0.25, 0.15
        combined_signal = w1 * np.linalg.norm(self.ps) / np.sqrt(self.dim)
        combined_signal += w2 * (1.0 - diversity) * np.linalg.norm(y_mean) / (np.linalg.norm(y_mean) + 1e-10)
        combined_signal += w3 * min(fit_gradient * 100.0, 2.0)

        # Adaptive damping based on fitness variance
        fit_variance = np.var(self.fitness)
        fit_var_normalized = fit_variance / (fit_scale ** 2 + 1e-10)
        adaptive_damping = self.damping * (1.0 + 2.0 * np.exp(-fit_var_normalized * 10.0))

        # Step size update with combined signal
        sigma_change = np.exp((combined_signal - 1.0) * self.cs / adaptive_damping)
        self.sigma *= sigma_change

        # Adaptive clipping based on convergence stage
        if fit_var_normalized < 0.01:
            sigma_min, sigma_max = 1e-14, 0.1
        elif fit_var_normalized > 1.0:
            sigma_min, sigma_max = 1e-8, 50.0
        else:
            sigma_min, sigma_max = 1e-10, 10.0

        self.sigma = np.clip(self.sigma, sigma_min, sigma_max)
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
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
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
        """Sample new trial population from multivariate normal."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
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
        """Adapt step-size using evolution path (CSA - Cumulation)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damping)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
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
        """Cholesky factor update - maintains C = LL^T directly."""
        if self.L is None:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        z_1 = y_mean / np.maximum(np.linalg.norm(y_mean), 1e-10)
        
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += w
        
        if total_weight > 0:
            rank_mu /= total_weight
        
        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        
        C_new = (1.0 - ccov_1 - ccov_mu) * self.C
        C_new += ccov_1 * np.outer(z_1, z_1)
        C_new += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])
        
        self.C = self._ensure_positive_definite(C_new)
        
        try:
            v = np.linalg.solve(self.L.T, z_1)
            alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
            self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
    
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
        """Dual active evolution paths with exponential history weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        alpha = 0.05
        if not hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        self.pc_weighted = (1.0 - alpha) * self.pc_weighted + alpha * y_mean
        
        beta = 0.02
        if not hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            cross_contribution = np.sqrt(beta) * (y_mean - self.prev_y_mean)
            self.p_cross = (1.0 - beta) * self.p_cross + cross_contribution
        self.prev_y_mean = y_mean.copy()
        
        rank_one_primary = np.outer(self.pc, self.pc)
        rank_one_weighted = np.outer(self.pc_weighted, self.pc_weighted)
        rank_one_cross = np.outer(self.p_cross, self.p_cross)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_primary = self.ccov
        ccov_weighted = 0.3 * self.ccov
        ccov_cross = 0.1 * self.ccov
        
        self.C = ((1.0 - ccov_primary) * self.C + 
                  ccov_primary * rank_one_primary +
                  ccov_weighted * rank_one_weighted +
                  ccov_cross * rank_one_cross +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
        
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1e7:
            self.C *= 0.5
    
    def _adapt_covariance_variant_09(self):
        """Diversity-sensitive and stagnation-aware scaling (most robust)."""
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)
        
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
        
        improvement = max(1e-10, self.prev_f_opt - self.f_opt)
        self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
        self.prev_f_opt = self.f_opt
        
        stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
        
        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * (1.5 if stagnation else 1.0) * (1.5 if diversity < 0.3 else 1.0)
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        rank_mu_scale = 2.0 if (stagnation or diversity < 0.3) else 1.0
        
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  ccov_adaptive * rank_mu_scale * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute population diversity as mean per-dimension std."""
        return np.mean(np.std(self.population, axis=0))
    
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