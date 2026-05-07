Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_step_size` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.980876e-02            5.647368e-02            6.202597e-02            -inf                    5.532805e-02            5.319574e-02            2.207312e-01            2.922329e-01            -inf                    5.845961e-02            5.843475e-02            
1      4.573482e-02            4.478334e-02            1.041030e-01            -inf                    3.987060e-02            4.396745e-02            4.833623e-01            6.723807e-01            -inf                    4.722298e-02            4.390214e-02            
2      1.482209e-01            2.303396e-01            1.912219e+00            -inf                    1.508675e-01            9.320328e-01            6.264183e+00            7.349436e+00            -inf                    1.230416e+00            2.733927e-01            
3      2.967010e-01            9.213855e-01            2.658094e+00            -inf                    1.354671e-01            2.036299e+00            5.909405e+00            7.105296e+00            -inf                    1.666733e+00            1.809569e-01            
4      5.227077e-01            5.188165e-01            6.833353e-01            -inf                    5.237153e-01            5.423551e-01            1.188300e+00            1.325574e+00            -inf                    6.266507e-01            5.218279e-01            
5      1.052188e-04            1.766477e-04            1.364272e+00            -inf                    1.640399e-04            2.227684e-04            8.975384e+01            1.725682e+02            -inf                    2.073907e-02            1.169040e+03            
6      8.556890e-02            2.465140e+01            6.067233e+00            -inf                    6.414725e-02            2.607837e+01            2.904712e+01            3.201140e+01            -inf                    2.142046e+01            3.111077e-02            
7      1.067957e-01            2.330114e-01            6.247395e-01            -inf                    1.300783e-01            9.044659e-02            3.261560e+00            4.123911e+00            -inf                    7.972482e-01            8.018207e-02            
8      7.086230e-01            4.396458e-01            1.539151e+00            -inf                    2.563622e-01            6.648499e-01            2.912943e+00            3.099276e+00            -inf                    1.142415e+00            2.592597e-01            
9      7.996643e-02            6.857090e+00            6.814360e+00            -inf                    1.012241e-01            6.829225e+00            9.490423e+00            1.063209e+01            -inf                    6.843212e+00            6.980678e+00            
10     2.331381e+01            2.864001e+01            4.092481e+01            -inf                    1.511323e+01            2.743776e+01            4.380212e+01            4.702484e+01            -inf                    4.095249e+01            2.433617e+01            
11     2.693611e+01            9.338521e+01            3.012266e+01            -inf                    8.958428e+00            6.458743e+01            6.943921e+01            1.376590e+02            -inf                    3.255151e+01            3.337468e+01            
12     2.414232e+01            3.137839e+01            2.221309e+01            -inf                    2.145673e+01            2.496437e+01            3.796738e+01            3.406862e+01            -inf                    2.873894e+01            2.653553e+01            
13     2.824585e+00            4.216219e+00            3.696625e+00            -inf                    3.474037e+00            3.480561e+00            4.003259e+00            7.801149e+00            -inf                    3.256973e+00            3.561317e+00            
14     2.956938e+00            3.019153e+00            2.889673e+00            -inf                    2.881758e+00            2.907145e+00            3.067295e+00            3.341905e+00            -inf                    3.066528e+00            3.004454e+00            
15     2.719692e+00            2.864946e+00            2.791986e+00            -inf                    2.651334e+00            2.852837e+00            2.735818e+00            3.163964e+00            -inf                    2.760017e+00            2.794555e+00            
16     1.380376e+02            1.505276e+02            1.240093e+02            -inf                    1.176738e+02            1.270736e+02            2.034153e+02            1.150430e+02            -inf                    1.016976e+02            1.750881e+02            
17     5.573199e+02            2.625833e+03            4.185074e+02            -inf                    4.057364e+01            4.906425e+02            4.337921e+03            3.457914e+03            -inf                    3.936235e+02            3.056418e+03            
18     2.002852e+01            2.323762e+01            2.150344e+01            -inf                    2.094346e+01            2.317060e+01            2.252241e+01            2.281895e+01            -inf                    2.389468e+01            2.130653e+01            
19     2.966696e+01            2.893281e+01            3.268464e+01            -inf                    2.590477e+01            3.836786e+01            2.634359e+01            3.026707e+01            -inf                    3.879025e+01            2.184302e+01            
20     2.266535e+01            2.144870e+01            2.197912e+01            -inf                    1.975945e+01            2.091753e+01            2.073798e+01            1.970385e+01            -inf                    2.197555e+01            2.033781e+01            
21     4.424215e+00            4.578718e+00            4.416883e+00            -inf                    4.552056e+00            4.526399e+00            4.542508e+00            4.540559e+00            -inf                    4.465095e+00            4.549864e+00            
22     7.767410e+00            1.166090e+01            1.110346e+01            -inf                    7.952405e+00            1.096318e+01            9.411264e+00            1.129021e+01            -inf                    1.034095e+01            7.940954e+00            
23     2.208430e+01            2.270731e+01            2.221018e+01            -inf                    1.911667e+01            2.078034e+01            2.074355e+01            2.225562e+01            -inf                    2.174336e+01            1.891922e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_05_idea_0.py  (error=5.319574e-02)
Task  1: variant_04_idea_0.py  (error=3.987060e-02)
Task  2: original.py  (error=1.482209e-01)
Task  3: variant_04_idea_0.py  (error=1.354671e-01)
Task  4: variant_01_idea_0.py  (error=5.188165e-01)
Task  5: original.py  (error=1.052188e-04)
Task  6: variant_10_idea_0.py  (error=3.111077e-02)
Task  7: variant_10_idea_0.py  (error=8.018207e-02)
Task  8: variant_04_idea_0.py  (error=2.563622e-01)
Task  9: original.py  (error=7.996643e-02)
Task 10: variant_04_idea_0.py  (error=1.511323e+01)
Task 11: variant_04_idea_0.py  (error=8.958428e+00)
Task 12: variant_04_idea_0.py  (error=2.145673e+01)
Task 13: original.py  (error=2.824585e+00)
Task 14: variant_04_idea_0.py  (error=2.881758e+00)
Task 15: variant_04_idea_0.py  (error=2.651334e+00)
Task 16: variant_09_idea_0.py  (error=1.016976e+02)
Task 17: variant_04_idea_0.py  (error=4.057364e+01)
Task 18: original.py  (error=2.002852e+01)
Task 19: variant_10_idea_0.py  (error=2.184302e+01)
Task 20: variant_07_idea_0.py  (error=1.970385e+01)
Task 21: variant_02_idea_0.py  (error=4.416883e+00)
Task 22: original.py  (error=7.767410e+00)
Task 23: variant_10_idea_0.py  (error=1.891922e+01)

WIN COUNTS:
  variant_04_idea_0.py: 9 wins
  original.py: 6 wins
  variant_10_idea_0.py: 4 wins
  variant_05_idea_0.py: 1 wins
  variant_01_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins
  variant_02_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using CMA-ES Cumulative Step-size Adaptation (CSA).

        This approach uses an evolution path ps that accumulates successful steps.
        If ||ps|| is larger than expected under random selection, sigma increases;
        if smaller, sigma decreases. This provides smoother, more robust adaptation
        than immediate success counting.
        """
        # Compute mean shift in sigma-scaled coordinates
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Update evolution path ps (cumulative conjugate gradient direction)
        # This path should behave like a random walk under success
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * y_mean

        # Expected length of ps under random normal distribution (chi_d)
        # chi_d ≈ sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim^2))
        chi_d = self.dim ** 0.5 * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        # Compute norm of evolution path
        ps_norm = np.linalg.norm(self.ps)

        # Avoid division by zero or extreme values
        ps_norm_safe = max(ps_norm, 1e-10)
        chi_d_safe = max(chi_d, 1e-10)

        # Step-size adaptation: if path is long, increase sigma; if short, decrease
        adaptation = (ps_norm_safe / chi_d_safe) - 1.0
        adaptation = np.clip(adaptation, -1.0, 1.0)

        # Apply exponential update with damping
        self.sigma *= np.exp((self.cs / self.damping) * adaptation)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_02_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
            """Adapt step-size using momentum-weighted improvement with diversity correction."""
            recent_improved = 0
            recent_total = 0
            improvement_sum = 0.0

            for i in range(self.NP):
                if i < len(self.trial_fitness) and i < len(self.fitness):
                    if self.trial_fitness[i] < self.fitness[i]:
                        recent_improved += 1
                        improvement_sum += self.fitness[i] - self.trial_fitness[i]
                    recent_total += 1

            if recent_total == 0:
                recent_total = max(1, self.NP // 4)

            success_rate = recent_improved / recent_total
            success_rate = np.clip(success_rate, 0.0, 1.0)

            # Momentum-based improvement tracking (EMA of fitness gain)
            if not hasattr(self, 'improvement_ema'):
                self.improvement_ema = 1.0

            avg_improvement = improvement_sum / max(recent_improved, 1)
            self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * avg_improvement

            # Diversity-based correction to prevent premature convergence
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.01, 1.0)

            # Target rate with non-linear diversity modulation
            target_rate = 0.25 * (1.0 + 0.5 * np.log1p(1.0 / (diversity_ratio + 1e-10)))
            target_rate = np.clip(target_rate, 0.05, 0.6)

            # Non-linear adaptation: steeper response when far from target
            rate_ratio = success_rate / max(target_rate, 1e-10)
            if rate_ratio < 0.5:
                # Stagnation detected: use non-linear boost to escape
                adaptation = -0.5 * (1.0 + np.log1p(0.5 / (rate_ratio + 1e-10)))
            elif rate_ratio > 1.5:
                # Very high success: can be more aggressive
                adaptation = 0.5 * np.log1p(rate_ratio)
            else:
                # Normal regime
                adaptation = (success_rate - target_rate) / target_rate

            adaptation = np.clip(adaptation, -0.8, 0.8)

            # Momentum-weighted damping (slower when improvement momentum is low)
            momentum_factor = np.clip(np.log1p(self.improvement_ema * 1e6) / 10.0, 0.2, 2.0)

            # Dimension and diversity-aware damping
            damping_adaptive = self.damping * momentum_factor * (0.3 + 0.7 * np.log1p(self.dim) / diversity_ratio)
            damping_adaptive = np.clip(damping_adaptive, 0.05, 150.0)

            # Adaptive learning rate (faster early, slower late)
            generation_factor = np.exp(-self.generation / (50 + self.dim * 2))
            cs_adaptive = self.cs * (1.0 + generation_factor)

            self.sigma *= np.exp(adaptation * cs_adaptive / damping_adaptive)
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_04_idea_0.py (9 wins) ---
```python
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
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using evolution path length control (CMA-ES style)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Evolution path for step-size: cumulative sum of normalized mean steps
        if not hasattr(self, 'p_sigma') or self.p_sigma is None:
            self.p_sigma = np.zeros(self.dim)

        self.p_sigma = (1.0 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        path_norm = np.linalg.norm(self.p_sigma)
        expected_norm = np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * max(1, self.generation)))
        expected_norm = max(expected_norm, 0.1)

        # Control signal: path too short -> increase sigma, path too long -> decrease
        control = path_norm / expected_norm - 1.0
        control = np.clip(control, -0.5, 0.5)

        # Adaptive damping based on dimension and stagnation
        stagnation_boost = 1.0 + 0.5 * min(self.stagnation_counter / max(1, self.max_stagnation), 1.0)
        damping_eff = self.damping * stagnation_boost
        damping_eff = np.clip(damping_eff, 0.1, 200.0)

        # Exponential update with controlled damping
        self.sigma *= np.exp(control * self.cs / damping_eff)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using momentum, fitness gradient, and diversity awareness."""
        # Compute fitness improvement
        f_mean_prev = getattr(self, 'f_mean_prev', float(np.mean(self.fitness)))
        f_mean_curr = float(np.mean(self.fitness))
        f_improvement = f_mean_prev - f_mean_curr

        # Initialize momentum if needed
        if not hasattr(self, 'sigma_momentum'):
            self.sigma_momentum = 0.0
            self.sigma_history = []

        # Update sigma history
        self.sigma_history.append(float(self.sigma))
        if len(self.sigma_history) > 10:
            self.sigma_history.pop(0)

        # Compute momentum: positive when improving, negative when stagnating
        improvement_rate = f_improvement / (abs(f_mean_prev) + 1e-10)
        target_momentum = float(np.clip(improvement_rate * 10.0, -1.0, 1.0))

        # Smooth momentum update with inertia
        momentum_alpha = 0.3
        self.sigma_momentum = (1.0 - momentum_alpha) * self.sigma_momentum + momentum_alpha * target_momentum
        self.sigma_momentum = float(np.clip(self.sigma_momentum, -0.8, 0.8))

        # Compute population diversity (normalized spread)
        pop_spread = float(np.mean(np.std(self.population, axis=0)))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_ratio = pop_spread / (expected_spread + 1e-10)
        diversity_ratio = float(np.clip(diversity_ratio, 0.0, 1.0))

        # Compute success rate for additional signal
        recent_improved = 0
        recent_total = 0
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(recent_improved / recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        # Diversity-aware damping: low diversity -> high damping (force exploration)
        min_diversity_thresh = 0.05
        if diversity_ratio < min_diversity_thresh:
            diversity_factor = 0.3  # Low damping to boost sigma
        else:
            diversity_factor = 1.0

        # Combined adaptation signal
        # Momentum dominates, success rate and diversity provide correction
        adaptation = self.sigma_momentum + 0.2 * (success_rate - 0.25)

        # Force sigma increase when diversity is critically low
        if diversity_ratio < min_diversity_thresh:
            adaptation = max(adaptation, 0.3)

        adaptation = float(np.clip(adaptation, -0.6, 0.6))

        # Adaptive damping with dimension scaling
        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) * diversity_factor
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 100.0))

        # Apply momentum-driven sigma update
        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-12, 15.0))

        # Prevent sigma collapse: if sigma is too small relative to population spread
        if pop_spread > 1e-10 and self.sigma < pop_spread * 1e-4:
            self.sigma = pop_spread * 1e-3

        # Store current mean fitness for next iteration
        self.f_mean_prev = f_mean_curr
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using CMA-ES-style cumulative evolution path with momentum."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Track cumulative evolution path (momentum of mean movement)
        if not hasattr(self, 'ps_path'):
            self.ps_path = np.zeros(self.dim)

        # Evolution path update with cumulation
        self.ps_path = (1.0 - self.cs) * self.ps_path + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        # Path length as measure of step-size adequacy
        path_norm = np.linalg.norm(self.ps_path)

        # Expected path length under random selection (approx sqrt(dim))
        expected_path = np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * self.generation + 1))
        expected_path = max(expected_path, 0.1)

        # Success measure from path length ratio
        success_ratio = path_norm / expected_path if expected_path > 1e-15 else 1.0
        success_ratio = np.clip(success_ratio, 0.01, 10.0)

        # Logarithmic adaptation for multiplicative step-size control
        adaptation = np.log(success_ratio) / np.log(2.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        # Adaptive damping based on dimension and path length
        damping_adaptive = self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(path_norm + 1e-10))
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        # Exponential step-size update with momentum
        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-12, 20.0)
```

# --- From variant_10_idea_0.py (4 wins) ---
```python
def _adapt_step_size(self):
        """Adapt step-size using magnitude-aware improvement with stagnation detection."""
        # Calculate improvement metrics
        if len(self.trial_fitness) > 0 and len(self.fitness) > 0:
            improvements = np.maximum(0.0, self.fitness - self.trial_fitness)
            recent_improved = np.sum(improvements > 0)
            total_improvement = np.sum(improvements)
            max_improvement = np.max(improvements)
        else:
            recent_improved = 0
            total_improvement = 0.0
            max_improvement = 0.0

        recent_total = max(1, len(self.trial_fitness))
        success_rate = recent_improved / recent_total

        # Dynamic target rate based on dimension
        target_rate = 0.2 + 0.1 * np.tanh(self.dim / 20.0)

        # Magnitude-aware adaptation: use log-scale improvement
        expected_improvement = max(abs(self.f_opt), 1.0) * 0.01
        if total_improvement > 0 and expected_improvement > 0:
            improvement_ratio = total_improvement / (expected_improvement * recent_total + 1e-15)
            magnitude_factor = np.tanh(np.log1p(improvement_ratio) / 5.0)
        else:
            magnitude_factor = -0.5

        # Stagnation detection: if no significant improvement, increase sigma aggressively
        is_stagnant = (max_improvement < abs(self.f_opt) * 1e-6 + 1e-10) or (self.f_opt > 1e3)

        if is_stagnant:
            # Aggressive expansion to escape local optima
            stagnation_factor = 2.0
            self.sigma = np.clip(self.sigma * stagnation_factor, 1e-10, 50.0)
        else:
            # Combine success rate and magnitude for adaptation
            rate_adaptation = (success_rate - target_rate) / (target_rate + 1e-15)
            combined_adaptation = 0.7 * rate_adaptation + 0.3 * magnitude_factor
            combined_adaptation = np.clip(combined_adaptation, -0.8, 0.8)

            # Adaptive damping based on dimension and current sigma
            damping_adaptive = self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(100))
            damping_adaptive = np.clip(damping_adaptive, 0.05, 50.0)

            # Exponential update with combined factors
            self.sigma *= np.exp(combined_adaptation * self.cs / damping_adaptive)
            self.sigma = np.clip(self.sigma, 1e-10, 50.0)
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