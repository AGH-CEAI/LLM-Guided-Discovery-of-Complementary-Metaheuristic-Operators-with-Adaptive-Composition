Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_sample_trials_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.841111e-02            5.806510e-02            1.735312e+00            6.061998e-02            6.022886e-02            6.162695e-02            5.752454e-02            1.673292e-01            7.881094e-02            1.174572e-01            3.059973e+01            
1      4.353874e-02            4.799558e-02            2.327073e+00            4.849397e-02            4.469807e-02            4.687259e-02            4.498536e-02            2.274512e-01            7.370143e-02            3.416057e-01            5.636677e+01            
2      3.272469e-01            3.872134e-01            5.357363e+00            3.953353e-01            8.084285e-01            1.417014e+00            2.776225e+00            6.301045e+00            3.651041e+00            4.658699e+00            3.104521e+02            
3      1.077107e+00            7.284251e-01            4.090680e+00            1.501477e+00            2.537116e+00            4.455491e-01            1.895181e+00            5.274953e+00            3.434068e+00            3.406856e+00            1.598724e+02            
4      5.405105e-01            5.373697e-01            1.409808e+00            5.744152e-01            5.444195e-01            5.864369e-01            1.204443e+00            1.280928e+00            1.061914e+00            1.250901e+00            4.171179e+00            
5      4.832773e-04            3.807195e-04            1.609162e+02            1.992646e-03            1.187336e-03            2.815927e-03            1.091843e+01            7.108440e+01            1.785955e+00            9.887231e+00            7.520552e+07            
6      1.191362e+01            1.301844e+00            4.533383e+01            1.306341e+01            1.231615e+01            1.366115e+01            3.521875e+01            9.172093e+01            4.203389e+01            5.820956e+01            1.505121e+02            
7      2.440828e-01            1.163609e-01            5.335329e+00            4.819585e-01            2.224741e-01            2.215211e-01            3.249695e+00            3.644235e+00            3.790246e-01            3.549372e+00            1.301444e+02            
8      9.965210e-01            8.041910e-01            2.750050e+00            8.819162e-01            7.373809e-01            7.790755e-01            1.962302e+00            2.991203e+00            2.265708e+00            2.072647e+00            3.075487e+01            
9      5.864569e+00            7.005194e+00            7.170495e+00            6.916916e+00            6.795096e+00            6.040309e+00            7.807297e+00            9.002129e+00            7.416799e+00            9.548140e+00            8.071850e+00            
10     2.286568e+01            1.496870e+01            4.719619e+01            4.603761e+01            2.305896e+01            1.352149e+01            1.655746e+01            5.273213e+01            4.703037e+01            1.414718e+01            1.479465e+02            
11     3.019150e+01            1.437512e+01            1.317531e+02            1.112575e+02            5.541881e+01            4.371804e+01            2.109481e+01            1.609868e+02            6.960024e+01            1.180996e+01            5.702488e+02            
12     1.641908e+01            2.598057e+01            3.878076e+01            6.319590e+01            2.843647e+01            2.232277e+01            1.910286e+01            5.782235e+01            3.011806e+01            1.528368e+01            1.150977e+02            
13     2.429108e+00            2.820615e+00            1.397357e+01            1.434911e+01            3.424201e+00            4.838286e+00            2.875368e+00            1.979339e+01            1.466067e+01            3.779842e+00            3.292239e+01            
14     2.625924e+00            2.822777e+00            3.372262e+00            2.705119e+00            2.564468e+00            2.800404e+00            3.710871e+00            3.957858e+00            2.910750e+00            3.581842e+00            5.345884e+00            
15     2.714215e+00            2.764336e+00            2.981461e+00            3.097497e+00            2.769429e+00            2.793396e+00            2.603614e+00            3.193606e+00            3.008412e+00            2.763019e+00            3.697754e+00            
16     1.022392e+02            1.373026e+02            2.224738e+02            1.809605e+02            1.761636e+02            2.108532e+02            3.510929e+02            2.683540e+03            3.040917e+02            6.133682e+02            5.369322e+02            
17     5.506900e+02            5.271915e+02            9.199799e+03            8.631371e+03            4.750436e+01            2.355548e+03            1.185791e+02            3.796794e+04            1.385147e+04            6.975825e+02            7.793868e+04            
18     2.309127e+01            2.137911e+01            2.691740e+01            3.390799e+01            2.220838e+01            2.438614e+01            1.843327e+01            3.788072e+01            2.891046e+01            2.190716e+01            5.620013e+01            
19     1.450811e+01            1.449410e+01            4.071093e+01            2.741974e+01            1.760964e+01            2.693640e+01            2.146929e+01            5.741658e+01            2.263879e+01            4.513468e+01            1.209146e+02            
20     2.157294e+01            2.341112e+01            2.484807e+01            2.611903e+01            2.064210e+01            2.322626e+01            2.014272e+01            2.699080e+01            2.490848e+01            2.480615e+01            3.271227e+01            
21     4.411725e+00            4.485484e+00            4.571463e+00            4.504436e+00            4.514588e+00            4.614118e+00            4.495096e+00            4.719565e+00            4.645250e+00            4.603630e+00            5.333560e+00            
22     7.208806e+00            9.350569e+00            1.185576e+01            1.144280e+01            8.706743e+00            1.049279e+01            3.970817e+00            1.173283e+01            1.156149e+01            9.518405e+00            1.371484e+01            
23     1.800047e+01            2.207464e+01            2.565392e+01            1.602778e+01            1.716082e+01            2.121418e+01            2.147521e+01            2.947109e+01            2.841916e+01            3.452321e+01            3.681744e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_idea_0.py  (error=5.752454e-02)
Task  1: original.py  (error=4.353874e-02)
Task  2: original.py  (error=3.272469e-01)
Task  3: variant_05_idea_0.py  (error=4.455491e-01)
Task  4: variant_01_idea_0.py  (error=5.373697e-01)
Task  5: variant_01_idea_0.py  (error=3.807195e-04)
Task  6: variant_01_idea_0.py  (error=1.301844e+00)
Task  7: variant_01_idea_0.py  (error=1.163609e-01)
Task  8: variant_04_idea_0.py  (error=7.373809e-01)
Task  9: original.py  (error=5.864569e+00)
Task 10: variant_05_idea_0.py  (error=1.352149e+01)
Task 11: variant_09_idea_0.py  (error=1.180996e+01)
Task 12: variant_09_idea_0.py  (error=1.528368e+01)
Task 13: original.py  (error=2.429108e+00)
Task 14: variant_04_idea_0.py  (error=2.564468e+00)
Task 15: variant_06_idea_0.py  (error=2.603614e+00)
Task 16: original.py  (error=1.022392e+02)
Task 17: variant_04_idea_0.py  (error=4.750436e+01)
Task 18: variant_06_idea_0.py  (error=1.843327e+01)
Task 19: variant_01_idea_0.py  (error=1.449410e+01)
Task 20: variant_06_idea_0.py  (error=2.014272e+01)
Task 21: original.py  (error=4.411725e+00)
Task 22: variant_06_idea_0.py  (error=3.970817e+00)
Task 23: variant_03_idea_0.py  (error=1.602778e+01)

WIN COUNTS:
  original.py: 6 wins
  variant_06_idea_0.py: 5 wins
  variant_01_idea_0.py: 5 wins
  variant_04_idea_0.py: 3 wins
  variant_05_idea_0.py: 2 wins
  variant_09_idea_0.py: 2 wins
  variant_03_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (5 wins) ---
```python
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
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _sample_trials_batch(self):
        """Sample trials with restart-triggered hybrid global+local exploration."""
        # Use wider sampling when stagnating to escape local optima
        is_stagnating = self.stagnation_counter > self.max_stagnation // 3

        if is_stagnating:
            # Hybrid: 40% global uniform + 60% local Gaussian with inflated sigma
            n_global = int(0.4 * self.NP)
            n_local = self.NP - n_global

            # Global exploration: uniform across entire bounds
            global_samples = np.random.uniform(self.lb, self.ub, (n_global, self.dim))

            # Local search: Gaussian around mean with wider spread
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(n_local, self.dim)
            local_samples = self.mean + 2.5 * self.sigma * (z @ L.T)

            self.trials = np.vstack([global_samples, local_samples])
        else:
            # Standard sampling
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * (z @ L.T)

        self.trials = self._clip_to_bounds(self.trials)
```

# --- From variant_04_idea_0.py (3 wins) ---
```python
def _sample_trials_batch(self):
        """Sample with stagnation-triggered heavy-tailed exploration to escape local optima."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        # Track generations since best improvement (independent of main stagnation counter)
        if not hasattr(self, '_best_at_last_improvement'):
            self._best_at_last_improvement = self.f_opt
            self._gen_at_last_improvement = self.generation

        gens_since_best = self.generation - self._gen_at_last_improvement

        # Update tracking if improved
        if self.f_opt < self._best_at_last_improvement - 1e-14:
            self._best_at_last_improvement = self.f_opt
            self._gen_at_last_improvement = self.generation
            gens_since_best = 0

        # Stagnation threshold scales with dimension
        stagnation_threshold = max(8, self.dim // 2)

        if gens_since_best >= stagnation_threshold:
            # Exploration mode: t-distribution with heavier tails for escaping local optima
            # Degrees of freedom decreases with dim for heavier tails in high dimensions
            df = max(2.0, self.dim / 5.0)
            z_raw = np.random.standard_t(df, size=(self.NP, self.dim))

            # Normalize each row, then scale by random factor in [1.5, 4.0] for varied jump sizes
            row_norms = np.linalg.norm(z_raw, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-8)
            z_normalized = z_raw / row_norms
            jump_factors = np.random.uniform(1.5, 4.0, size=(self.NP, 1))
            z_explore = z_normalized * jump_factors

            # Use larger sigma during exploration
            sigma_explore = self.sigma * 3.0
            self.trials = self.mean + sigma_explore * (z_explore @ L.T)
        else:
            # Normal mode: standard multivariate normal sampling
            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * (z @ L.T)

        self.trials = self._clip_to_bounds(self.trials)
```

# --- From variant_05_idea_0.py (2 wins) ---
```python
def _sample_trials_batch(self):
        """Sample with pulsed explosive exploration on stagnation."""
        # Check for stagnation to trigger exploration burst
        is_stagnant = (self.stagnation_counter > self.max_stagnation // 2)

        # Track exploration burst state
        if not hasattr(self, 'in_explosion'):
            self.in_explosion = False
            self.explosion_gen = 0
            self.sigma_backup = None

        # Trigger explosive exploration phase
        if is_stagnant and not self.in_explosion:
            self.in_explosion = True
            self.explosion_gen = 0
            # Backup current sigma for later restoration
            self.sigma_backup = self.sigma
            # Explosive expansion: increase sigma by factor of 10-20
            self.sigma = min(self.sigma * 15.0, (self.ub[0] - self.lb[0]) * 0.4)
            # Reset evolution path to remove misleading momentum
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
            # Reset covariance to near-identity (isotropic exploration)
            self.C = 0.5 * np.eye(self.dim) + 0.5 * np.diag(np.diag(self.C))

        if self.in_explosion:
            self.explosion_gen += 1
            # Decay explosion over ~20 generations
            explosion_decay = max(0.05, 1.0 - self.explosion_gen / 20.0)

            # Sample from current covariance (reset to near-isotropic)
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Wide isotropic sampling with decaying radius
            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * explosion_decay * (z @ L.T)
            self.trials = self._clip_to_bounds(self.trials)

            # Exit explosion phase after decay
            if self.explosion_gen >= 20 and self.sigma_backup is not None:
                self.in_explosion = False
                self.sigma = self.sigma_backup
                self.explosion_gen = 0
                self.sigma_backup = None
        else:
            # Standard CMA-ES sampling
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * (z @ L.T)
            self.trials = self._clip_to_bounds(self.trials)
```

# --- From variant_06_idea_0.py (5 wins) ---
```python
def _sample_trials_batch(self):
        """Sample with covariance restart on degeneracy detection."""
        eigvals, eigvecs = np.linalg.eigh(self.C)

        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        cond_num = max_eig / max(min_eig, 1e-30)

        # Restart covariance to isotropic if severely degenerate
        if cond_num > 1000.0 * self.dim:
            self.C = self.sigma * np.eye(self.dim)
            eigvals, eigvecs = np.linalg.eigh(self.C)

        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _sample_trials_batch(self):
        """Adaptive multi-component hybrid sampling for escaping local optima."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        z_cov = np.random.randn(self.NP, self.dim)
        samples_cov = z_cov @ L_cov.T

        z_iso = np.random.randn(self.NP, self.dim)
        samples_iso = z_iso

        if hasattr(self, 'success_directions') and len(self.success_directions) > 0:
            stacked = np.array(self.success_directions)
            stacked = stacked - np.mean(stacked, axis=0)
            Q, _ = np.linalg.qr(stacked.T)
            z_hist = np.random.randn(self.NP, min(self.dim, len(self.success_directions)))
            samples_hist = z_hist @ Q[:len(self.success_directions), :].T
        else:
            samples_hist = np.zeros((self.NP, self.dim))

        pop_var = np.mean(np.var(self.population, axis=0)) if hasattr(self, 'population') and len(self.population) > 1 else 1.0
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_var / (expected_var + 1e-10), 0.01, 1.0)

        iso_ratio = 0.3 * (1.0 - diversity) + 0.1
        iso_ratio = np.clip(iso_ratio, 0.05, 0.5)

        hist_ratio = 0.2 * diversity + 0.05
        hist_ratio = np.clip(hist_ratio, 0.05, 0.3)

        cov_ratio = max(0.0, 1.0 - iso_ratio - hist_ratio)

        combined = (cov_ratio * samples_cov + 
                    iso_ratio * samples_iso + 
                    hist_ratio * samples_hist)

        self.trials = self.mean + self.sigma * combined
        self.trials = self._clip_to_bounds(self.trials)
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