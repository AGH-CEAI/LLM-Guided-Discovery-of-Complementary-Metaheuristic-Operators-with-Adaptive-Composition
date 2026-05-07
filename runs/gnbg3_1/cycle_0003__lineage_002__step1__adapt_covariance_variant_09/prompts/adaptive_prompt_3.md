Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_09` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.740224e-02            5.783364e-02            5.648747e-02            5.980876e-02            5.826304e-02            6.583817e-02            6.007327e-02            -inf                    6.222619e-02            6.692689e-02            6.039978e-02            
1      4.773438e-02            4.446444e-02            4.378946e-02            4.573482e-02            4.692157e-02            6.942647e-02            4.666375e-02            -inf                    4.553331e-02            4.707029e-02            4.673820e-02            
2      5.090398e-01            7.045896e-01            7.003173e-01            1.482209e-01            7.733191e-01            4.012138e-01            4.919460e-01            -inf                    9.644828e-01            5.272323e-01            6.235798e-01            
3      9.301433e-01            6.588962e-01            8.578735e-01            2.967010e-01            9.048367e-01            1.817114e+00            1.289510e+00            -inf                    2.011710e+00            5.043374e+00            1.008610e+00            
4      5.464598e-01            5.280889e-01            6.415936e-01            5.227077e-01            5.231518e-01            6.668948e-01            5.627670e-01            -inf                    6.634969e-01            5.437377e-01            5.766317e-01            
5      1.084660e-03            2.279099e-04            9.334466e-04            1.052188e-04            4.700943e-02            2.097585e-02            2.299401e-02            -inf                    2.886047e-02            6.959511e-02            5.901939e-03            
6      4.844004e+00            3.328588e-01            2.293605e+01            8.556890e-02            2.458021e+01            2.625646e+01            3.073040e+01            -inf                    2.363916e+01            3.113882e+01            2.681100e+01            
7      1.628064e-01            1.265499e-01            2.636888e-01            1.067957e-01            3.026155e-01            4.608234e-01            2.389031e-01            -inf                    3.270295e-01            8.607845e-01            2.778143e-01            
8      4.653467e-01            4.058865e-01            1.214948e+00            7.086230e-01            7.044732e-01            1.278786e+00            7.525355e-01            -inf                    5.284933e-01            1.865097e+00            7.610734e-01            
9      6.669777e+00            7.509416e+00            7.510251e+00            7.996643e-02            6.480584e+00            7.807832e+00            7.594858e+00            -inf                    7.554665e+00            7.944713e+00            7.663704e+00            
10     1.724373e+01            2.079812e+01            1.143589e+01            2.331381e+01            2.111345e+01            2.935985e+01            2.493970e+01            -inf                    1.105771e+01            3.575515e+01            2.351190e+01            
11     2.881134e+01            2.372506e+01            7.497936e+00            2.693611e+01            2.668818e+01            8.531263e+01            2.505012e+01            -inf                    1.844124e+01            2.635192e+01            1.972258e+01            
12     2.487322e+01            2.691562e+01            1.306736e+01            2.414232e+01            3.207889e+01            3.393197e+01            1.705464e+01            -inf                    1.789484e+01            1.993468e+01            2.087846e+01            
13     2.294564e+00            2.240324e+00            1.965162e+00            2.824585e+00            3.279583e+00            1.286604e+01            2.791636e+00            -inf                    2.546579e+00            2.952095e+00            2.671858e+00            
14     2.610978e+00            3.058112e+00            3.006951e+00            2.956938e+00            2.850208e+00            3.072444e+00            2.979289e+00            -inf                    3.084261e+00            2.913579e+00            2.987757e+00            
15     2.711519e+00            2.592145e+00            2.477485e+00            2.719692e+00            2.591966e+00            2.997424e+00            2.629006e+00            -inf                    2.775015e+00            2.657621e+00            2.832525e+00            
16     1.085934e+02            2.009754e+02            2.420996e+02            1.380376e+02            1.203734e+02            2.629240e+02            2.110487e+02            -inf                    3.242855e+02            2.763304e+02            2.992357e+02            
17     7.079749e+02            5.771931e+02            2.125609e+01            5.573199e+02            4.990370e+01            1.772968e+02            5.140755e+01            -inf                    2.307938e+02            1.350054e+03            1.679983e+03            
18     1.968582e+01            2.121362e+01            1.896326e+01            2.002852e+01            2.634903e+01            2.475427e+01            1.857382e+01            -inf                    2.130703e+01            2.577658e+01            2.273331e+01            
19     1.239370e+01            2.256751e+01            3.046533e+01            2.966696e+01            2.501055e+01            2.654570e+01            2.628398e+01            -inf                    3.038057e+01            1.778104e+01            3.456607e+01            
20     2.129990e+01            2.118644e+01            2.267095e+01            2.266535e+01            2.440674e+01            2.549464e+01            2.310923e+01            -inf                    2.190714e+01            2.432928e+01            2.494237e+01            
21     4.450044e+00            4.539207e+00            4.555070e+00            4.424215e+00            4.506285e+00            4.575942e+00            4.552381e+00            -inf                    4.560003e+00            4.635456e+00            4.567370e+00            
22     9.265769e+00            1.065897e+01            6.677396e+00            7.767410e+00            1.097995e+01            9.815735e+00            1.097162e+01            -inf                    1.049882e+01            1.104730e+01            1.066381e+01            
23     1.663020e+01            1.980036e+01            1.903688e+01            2.208430e+01            1.961683e+01            1.993394e+01            2.393140e+01            -inf                    1.967339e+01            3.397038e+01            2.318301e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_02_idea_0.py  (error=5.648747e-02)
Task  1: variant_02_idea_0.py  (error=4.378946e-02)
Task  2: variant_03_idea_0.py  (error=1.482209e-01)
Task  3: variant_03_idea_0.py  (error=2.967010e-01)
Task  4: variant_03_idea_0.py  (error=5.227077e-01)
Task  5: variant_03_idea_0.py  (error=1.052188e-04)
Task  6: variant_03_idea_0.py  (error=8.556890e-02)
Task  7: variant_03_idea_0.py  (error=1.067957e-01)
Task  8: variant_01_idea_0.py  (error=4.058865e-01)
Task  9: variant_03_idea_0.py  (error=7.996643e-02)
Task 10: variant_08_idea_0.py  (error=1.105771e+01)
Task 11: variant_02_idea_0.py  (error=7.497936e+00)
Task 12: variant_02_idea_0.py  (error=1.306736e+01)
Task 13: variant_02_idea_0.py  (error=1.965162e+00)
Task 14: original.py  (error=2.610978e+00)
Task 15: variant_02_idea_0.py  (error=2.477485e+00)
Task 16: original.py  (error=1.085934e+02)
Task 17: variant_02_idea_0.py  (error=2.125609e+01)
Task 18: variant_06_idea_0.py  (error=1.857382e+01)
Task 19: original.py  (error=1.239370e+01)
Task 20: variant_01_idea_0.py  (error=2.118644e+01)
Task 21: variant_03_idea_0.py  (error=4.424215e+00)
Task 22: variant_02_idea_0.py  (error=6.677396e+00)
Task 23: original.py  (error=1.663020e+01)

WIN COUNTS:
  variant_02_idea_0.py: 8 wins
  variant_03_idea_0.py: 8 wins
  original.py: 4 wins
  variant_01_idea_0.py: 2 wins
  variant_08_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Active CMA-ES: Negative weights shrink bad directions."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Active CMA-ES: Use both positive and negative weights
        n_active = self.mu
        active_weights = np.zeros(self.NP)
        for i in range(self.NP):
            if i < self.mu:
                active_weights[i] = np.log(self.mu + 0.5) - np.log(i + 1)
            elif i < n_active:
                active_weights[i] = -np.log(n_active + 0.5) + np.log(i + 1)

        sum_pos = np.sum(active_weights[active_weights > 0])
        sum_neg = np.sum(active_weights[active_weights < 0])

        if sum_pos > 0:
            active_weights[active_weights > 0] /= sum_pos
        if sum_neg < 0:
            active_weights[active_weights < 0] /= -sum_neg

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(n_active):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += active_weights[i] * np.outer(diff, diff)

        rank_one = np.outer(self.pc, self.pc)

        # Adaptive learning rate based on condition number
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
        ccov_base = min(self.ccov, 0.3 / (1.0 + np.log1p(cond)))

        # Combine rank-1 and rank-μ updates
        self.C = ((1.0 - ccov_base) * self.C + 
                  ccov_base * rank_one + 
                  ccov_base * (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu)

        # Ensure positive definiteness
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_02_idea_0.py (8 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Exponential moving average covariance with natural gradient damping."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute natural gradient direction (C^-1 @ y_mean)
        try:
            y_cov_norm = np.linalg.solve(self.C, y_mean)
        except np.linalg.LinAlgError:
            y_cov_norm = np.linalg.solve(self.C + 1e-6 * np.eye(self.dim), y_mean)

        nat_grad_norm = np.linalg.norm(y_cov_norm)
        damping_factor = min(1.0 + nat_grad_norm * 0.1, 2.0)

        cc_damped = self.cc / damping_factor
        self.pc = (1.0 - cc_damped) * self.pc + np.sqrt(cc_damped * (2.0 - cc_damped)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Initialize EMA tracking for smooth covariance evolution
        if not hasattr(self, 'ema_rank_one'):
            self.ema_rank_one = np.zeros((self.dim, self.dim))
            self.ema_rank_mu = np.zeros((self.dim, self.dim))
            self.ema_beta = 0.95

        self.ema_rank_one = self.ema_beta * self.ema_rank_one + (1.0 - self.ema_beta) * rank_one
        self.ema_rank_mu = self.ema_beta * self.ema_rank_mu + (1.0 - self.ema_beta) * rank_mu

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)

        # Stagnation detection for adaptive scaling
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0

        improvement = max(1e-10, self.prev_f_opt - self.f_opt)
        self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
        self.prev_f_opt = self.f_opt

        stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))

        # Adaptive learning rate based on state
        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        ccov_scale = 1.0
        if stagnation:
            ccov_scale *= 2.0
        if diversity < 0.3:
            ccov_scale *= 1.5

        ccov_adaptive = np.clip(base_ccov * ccov_scale, 1e-10, 0.5)

        # Blend current and EMA updates for stability
        ema_weight = 0.3 if stagnation else 0.5
        rank_one_blend = ema_weight * self.ema_rank_one + (1.0 - ema_weight) * rank_one
        rank_mu_blend = ema_weight * self.ema_rank_mu + (1.0 - ema_weight) * rank_mu

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one_blend + 
                  ccov_adaptive * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu_blend)

        # Eigenvalue bounds for conditioning control
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min, eig_max = np.min(eigvals), np.max(eigvals)
        cond = eig_max / (eig_min + 1e-10)

        if cond > 1e6:
            shrink = np.sqrt(1e6 / cond)
            self.C *= shrink

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_03_idea_0.py (8 wins) ---
```python
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
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Eigenvalue floor with active conditioning - prevents premature collapse."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Base covariance update
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Enforce minimum eigenvalue floor to prevent collapse
        eigvals, eigvecs = np.linalg.eigh(self.C)
        min_eig = np.min(eigvals)
        eig_floor = 1e-6 * np.max(eigvals)

        if min_eig < eig_floor:
            # Raise small eigenvalues to floor
            eigvals[eigvals < eig_floor] = eig_floor
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Active condition number control
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
        max_cond = 1e6

        if cond > max_cond:
            # Compress condition number by scaling extreme eigenvalues
            target_spread = max_cond
            min_target = np.min(eigvals)
            max_target = min_target * target_spread

            # Compress large eigenvalues
            mask_large = eigvals > max_target
            if np.any(mask_large):
                eigvals[mask_large] = max_target

            # Expand small eigenvalues if needed
            if np.min(eigvals) < min_target:
                eigvals[eigvals < min_target] = min_target

            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _adapt_covariance_variant_09(self):
        """Eigenvalue-drift control with active conditioning number management."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        pop_spread = np.max(np.std(self.population, axis=0)) / (np.min(np.std(self.population, axis=0)) + 1e-10)
        target_cond = min(1e6, self.dim * 10.0)

        if not hasattr(self, 'eigen_history'):
            self.eigen_history = []

        eigvals, eigvecs = np.linalg.eigh(self.C)
        self.eigen_history.append(eigvals.copy())
        if len(self.eigen_history) > 5:
            self.eigen_history.pop(0)

        if len(self.eigen_history) >= 3:
            eigen_drift = np.max(np.abs(self.eigen_history[-1] / (self.eigen_history[0] + 1e-10) - 1.0))
        else:
            eigen_drift = 0.0

        cond_C = np.max(eigvals) / (np.min(eigvals) + 1e-10)

        if cond_C > target_cond:
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            target_min = max_eig / target_cond

            if min_eig < target_min * 0.9:
                self.C += eigvecs @ np.diag(np.maximum(0, target_min - eigvals)) @ eigvecs.T

        ccov_adaptive = base_ccov
        if eigen_drift > 0.5:
            ccov_adaptive *= 0.7
        elif eigen_drift < 0.1 and cond_C < target_cond:
            ccov_adaptive *= 1.3

        if pop_spread < 0.1:
            ccov_adaptive *= 1.5
        elif pop_spread > 10.0:
            ccov_adaptive *= 0.8

        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  ccov_adaptive * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        eigvals_final = np.linalg.eigvalsh(self.C)
        if np.min(eigvals_final) < 1e-12:
            self.C += (1e-10 - np.min(eigvals_final)) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
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