Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_original` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.812975e-02            -inf                    5.736492e-02            5.693587e-02            5.698015e-02            5.644365e-02            5.619701e-02            5.431900e-02            5.778450e-02            5.810940e-02            
1      4.496215e-02            -inf                    4.406181e-02            4.276369e-02            4.587642e-02            4.504228e-02            4.338716e-02            4.415680e-02            4.288379e-02            4.428389e-02            
2      1.578556e-01            -inf                    1.629739e-01            1.495499e-01            1.554141e-01            1.526198e-01            2.016044e-01            1.390500e-01            1.525873e-01            1.797591e-01            
3      1.291268e-01            -inf                    1.298322e-01            1.368949e-01            1.338006e-01            1.357429e-01            1.567545e+00            1.325922e-01            1.828215e-01            1.772737e-01            
4      5.148583e-01            -inf                    5.190724e-01            5.250389e-01            5.320754e-01            5.181029e-01            5.166171e-01            5.065503e-01            5.150921e-01            5.106138e-01            
5      1.126927e-04            -inf                    8.733112e-05            1.004936e-04            9.752135e-05            1.172349e-04            8.743569e-05            6.693666e-05            1.425056e-04            9.059178e-05            
6      1.546407e-02            -inf                    1.543397e-02            1.552748e-02            1.511093e-02            1.782422e-02            1.527267e-02            1.470655e-02            1.511103e-02            1.511651e-02            
7      7.948120e-02            -inf                    7.762079e-02            7.987785e-02            7.955351e-02            8.332294e-02            8.177743e-02            7.492109e-02            8.424861e-02            8.776848e-02            
8      2.617683e-01            -inf                    2.705688e-01            2.947133e-01            2.498051e-01            2.616766e-01            2.514284e-01            2.423052e-01            2.621317e-01            2.615661e-01            
9      7.222536e-02            -inf                    1.318022e-01            1.432964e-01            1.059211e-01            1.834278e-01            8.954317e-02            8.705959e-02            1.466715e-01            1.188250e-01            
10     1.243494e+01            -inf                    3.363222e+01            1.081971e+01            2.401414e+01            2.830054e+01            2.045154e+01            3.095832e+01            1.095520e+01            2.103502e+01            
11     1.626618e+01            -inf                    6.509486e+01            1.679804e+01            5.717116e+01            3.912368e+01            8.271160e+01            1.296352e+02            2.111230e+01            2.252607e+01            
12     1.768508e+01            -inf                    2.113599e+01            1.826819e+01            3.099856e+01            2.101202e+01            3.418652e+01            2.578360e+01            1.567369e+01            1.180666e+01            
13     2.751837e+00            -inf                    4.040182e+00            2.449287e+00            5.059325e+00            3.734180e+00            3.315951e+00            2.471968e+00            2.367473e+00            3.077932e+00            
14     2.703046e+00            -inf                    3.518580e+00            2.700364e+00            2.764886e+00            2.764208e+00            3.123562e+00            2.631088e+00            2.716747e+00            2.763996e+00            
15     2.692457e+00            -inf                    3.513394e+00            2.719678e+00            2.833179e+00            2.730923e+00            3.016949e+00            2.889712e+00            2.641761e+00            2.717966e+00            
16     8.564369e+01            -inf                    8.339990e+01            9.320663e+01            9.010806e+01            9.175305e+01            9.642769e+01            5.915072e+01            7.122007e+01            7.231778e+01            
17     3.727647e+01            -inf                    3.176854e+02            7.661598e+01            1.730879e+03            3.230367e+02            5.385608e+01            2.233495e+02            3.317605e+01            1.042151e+02            
18     1.807794e+01            -inf                    2.200542e+01            1.367876e+01            2.232267e+01            2.310024e+01            1.959489e+01            1.808378e+01            1.590151e+01            1.946975e+01            
19     1.386868e+01            -inf                    2.133703e+01            1.199393e+01            2.603747e+01            1.486595e+01            2.852219e+01            1.458523e+01            1.702172e+01            1.484302e+01            
20     1.881006e+01            -inf                    2.036379e+01            2.001058e+01            2.127444e+01            2.179940e+01            2.406986e+01            1.911290e+01            2.081253e+01            2.027284e+01            
21     4.279396e+00            -inf                    4.587566e+00            4.516712e+00            4.524229e+00            4.507379e+00            4.540472e+00            4.361365e+00            4.437390e+00            4.301783e+00            
22     8.372090e+00            -inf                    1.202118e+01            1.016092e+01            9.223454e+00            9.074620e+00            1.037697e+01            1.124061e+01            9.598075e+00            8.083250e+00            
23     2.045874e+01            -inf                    2.448175e+01            2.251457e+01            2.603701e+01            1.891881e+01            2.132793e+01            1.688869e+01            1.897264e+01            2.206002e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_08_idea_0.py  (error=5.431900e-02)
Task  1: variant_03_idea_0.py  (error=4.276369e-02)
Task  2: variant_08_idea_0.py  (error=1.390500e-01)
Task  3: original.py  (error=1.291268e-01)
Task  4: variant_08_idea_0.py  (error=5.065503e-01)
Task  5: variant_08_idea_0.py  (error=6.693666e-05)
Task  6: variant_08_idea_0.py  (error=1.470655e-02)
Task  7: variant_08_idea_0.py  (error=7.492109e-02)
Task  8: variant_08_idea_0.py  (error=2.423052e-01)
Task  9: original.py  (error=7.222536e-02)
Task 10: variant_03_idea_0.py  (error=1.081971e+01)
Task 11: original.py  (error=1.626618e+01)
Task 12: variant_10_idea_0.py  (error=1.180666e+01)
Task 13: variant_09_idea_0.py  (error=2.367473e+00)
Task 14: variant_08_idea_0.py  (error=2.631088e+00)
Task 15: variant_09_idea_0.py  (error=2.641761e+00)
Task 16: variant_08_idea_0.py  (error=5.915072e+01)
Task 17: variant_09_idea_0.py  (error=3.317605e+01)
Task 18: variant_03_idea_0.py  (error=1.367876e+01)
Task 19: variant_03_idea_0.py  (error=1.199393e+01)
Task 20: original.py  (error=1.881006e+01)
Task 21: original.py  (error=4.279396e+00)
Task 22: variant_10_idea_0.py  (error=8.083250e+00)
Task 23: variant_08_idea_0.py  (error=1.688869e+01)

WIN COUNTS:
  variant_08_idea_0.py: 10 wins
  original.py: 5 wins
  variant_03_idea_0.py: 4 wins
  variant_09_idea_0.py: 3 wins
  variant_10_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_03_idea_0.py (4 wins) ---
```python
def _adapt_covariance_original(self):
        """Mean-stagnation eigenscale reset: push exploration outward when stuck."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Detect mean stagnation: is the mean barely moving relative to step size?
        mean_displacement = np.linalg.norm(self.mean - self.old_mean)
        stagnation_threshold = 1e-3 * self.sigma

        if mean_displacement < stagnation_threshold and mean_displacement > 1e-15:
            # Mean is stagnant — apply exponential eigenscale perturbation to covariance
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-10)

                # Exponential scaling: push outward along each principal axis
                # Stronger expansion along low-eigenvalue (narrow) directions
                log_eigvals = np.log(eigvals + 1e-10)
                log_scale = log_eigvals - np.min(log_eigvals) + 1.0
                scale_factors = np.power(log_scale, -1.5)
                scale_factors = np.clip(scale_factors, 0.1, 10.0)

                # Perturb eigvals: expand narrow directions, contract wide ones
                perturbed_eigvals = eigvals * scale_factors
                perturbed_eigvals = np.maximum(perturbed_eigvals, 1e-10)

                # Reconstruct perturbed covariance
                self.C = eigvecs @ np.diag(perturbed_eigvals) @ eigvecs.T

            except np.linalg.LinAlgError:
                pass

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_08_idea_0.py (10 wins) ---
```python
def _adapt_covariance_original(self):
        """Covariance adaptation with stagnation-triggered restart for escaping local optima."""
        stagnation_threshold = max(20, self.dim * 2)
        cond_threshold = 1e7
        eigmin_threshold = 1e-8

        need_restart = (self.stagnation_counter > stagnation_threshold)

        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / max(np.min(eigvals), 1e-15)
        need_restart = need_restart or (cond > cond_threshold) or (np.min(eigvals) < eigmin_threshold)

        if need_restart:
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc *= 0.0
            self.stagnation_counter = 0

        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'momentum_ema'):
            self.momentum_ema = np.zeros(self.dim)
        self.momentum_ema = 0.7 * self.momentum_ema + 0.3 * y_mean
        y_momentum = self.momentum_ema / max(np.linalg.norm(self.momentum_ema), 1e-10) * np.linalg.norm(y_mean)

        cc_adapt = self.cc * (1.5 if need_restart else 1.0)
        cc_adapt = np.clip(cc_adapt, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_momentum

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_boost = 3.0 if need_restart else 1.0
        ccov_adapt = min(self.ccov * ccov_boost, 0.4)

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        if need_restart:
            self.C += 0.1 * np.eye(self.dim) * (self.sigma ** 2)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_09_idea_0.py (3 wins) ---
```python
def _adapt_covariance_original(self):
        """Eigenspace restart with orthogonal perturbation for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Detect trapping: low fitness variance AND stagnation
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if not hasattr(self, 'prev_f_opt_trap'):
            self.prev_f_opt_trap = self.f_opt
            self.stagnation_gen = 0
        else:
            f_improved = self.prev_f_opt_trap - self.f_opt
            if f_improved > 1e-10:
                self.stagnation_gen = 0
            else:
                self.stagnation_gen += 1
            self.prev_f_opt_trap = self.f_opt

        # Trigger eigenspace restart if trapped
        is_trapped = (rel_var < 1e-3) and (self.stagnation_gen > self.dim * 2)

        if is_trapped:
            # Eigendecompose current covariance
            eigvals, eigvecs = np.linalg.eigh(self.C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # Mix eigenvectors to create new basis (prevents collapse)
            theta = 0.3 * np.pi
            mix_matrix = np.cos(theta) * np.eye(self.dim) + np.sin(theta) * (np.ones((self.dim, self.dim)) / self.dim - np.eye(self.dim))
            eigvecs_mixed = eigvecs @ mix_matrix

            # Inject exploration noise along weakest eigendirections
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            eigvals_scaled = eigvals.copy()

            # Boost weak directions, cap strong directions
            for i in range(self.dim):
                ratio = eigvals[i] / (max_eig + 1e-10)
                if ratio < 0.1:
                    eigvals_scaled[i] = min_eig + 0.2 * (max_eig - min_eig)
                elif ratio > 0.9:
                    eigvals_scaled[i] = 0.7 * max_eig

            # Reconstruct covariance in mixed basis
            self.C = eigvecs_mixed @ np.diag(eigvals_scaled) @ eigvecs_mixed.T
            self.C = 0.5 * (self.C + self.C.T)

            # Reset evolution path to align with new covariance
            self.pc *= 0.1
            self.stagnation_gen = 0

            # Increase learning rate temporarily
            ccov_boost = min(self.ccov * 3.0, 0.5)
            self.C = (1.0 - ccov_boost) * self.C + ccov_boost * np.eye(self.dim)
        else:
            self.stagnation_gen = max(0, self.stagnation_gen - 1)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _adapt_covariance_original(self):
        """Eigenspace perturbation with condition-aware adaptation for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Compute eigendecomposition for condition monitoring
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            eigvals = np.diag(self.C)
            eigvecs = np.eye(self.dim)

        eig_min = np.maximum(np.min(eigvals), 1e-12)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-12)

        # Detect catastrophic convergence (primary failure mode on hardest tasks)
        is_trapped = (cond > 1e4) or (eig_spread < 1e-4) or (eig_min < 1e-8)

        # Adaptive learning rates based on condition number
        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.5
        elif cond > 1e3:
            ccov_scale = 0.3
            cc_scale = 0.7
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        ccov_eff = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
        cc_eff = np.clip(self.cc * cc_scale, 0.001, 0.3)

        # Rebuild evolution path with scaled learning rate
        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Standard rank-mu update
        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        # Perturb eigenspectrum if trapped
        if is_trapped:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)

                # Clip eigenvalues to prevent collapse
                eigvals = np.clip(eigvals, 1e-10, None)

                # Inject exploration along minor eigendirections
                min_eig = np.min(eigvals)
                max_eig = np.max(eigvals)
                exploration_budget = 0.1 * max_eig

                # Distribute exploration across bottom 50% of eigendirections
                n_explore = max(1, self.dim // 2)
                sorted_indices = np.argsort(eigvals)
                for idx in sorted_indices[:n_explore]:
                    eigvals[idx] += exploration_budget / n_explore

                # Rebuild covariance from perturbed spectrum
                self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            except np.linalg.LinAlgError:
                # Fallback: add isotropic exploration
                self.C += 0.05 * np.eye(self.dim)

        # Ensure positive definiteness
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