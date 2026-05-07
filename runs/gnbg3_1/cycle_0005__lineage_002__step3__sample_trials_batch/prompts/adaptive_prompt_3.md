Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_sample_trials_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.532805e-02            5.721924e-02            1.959189e-01            6.742745e-02            9.650202e-02            1.120839e+01            -inf                    5.951174e-02            -inf                    1.050988e+01            -inf                    
1      3.987060e-02            4.196099e-02            1.713381e-01            1.640166e+00            1.523327e-01            1.099115e+01            -inf                    5.236852e-02            -inf                    1.047122e+01            -inf                    
2      1.508675e-01            3.642094e+00            5.321771e+00            3.860030e+00            3.063265e+00            8.691116e+01            -inf                    4.393072e+00            -inf                    4.713256e+01            -inf                    
3      1.354671e-01            2.839861e+00            4.169012e+00            2.528424e+00            4.124560e+00            6.857693e+01            -inf                    2.580112e+00            -inf                    5.447448e+01            -inf                    
4      5.237153e-01            5.702695e-01            1.378315e+00            1.322013e+00            1.654417e+00            2.026422e+00            -inf                    1.036635e+00            -inf                    2.461145e+00            -inf                    
5      1.640399e-04            1.909958e+00            4.820705e-01            9.903989e+01            8.519764e+01            1.824272e+03            -inf                    8.605920e-04            -inf                    6.037558e+02            -inf                    
6      6.414725e-02            2.535868e+01            3.665368e+01            6.545085e+02            1.276512e+02            9.033754e+02            -inf                    2.534268e+01            -inf                    8.445216e+02            -inf                    
7      1.300783e-01            1.715792e+00            4.587874e+00            2.560531e+00            3.888133e+00            7.107792e+01            -inf                    5.681575e+00            -inf                    4.737829e+01            -inf                    
8      2.563622e-01            2.014788e+00            2.515909e+00            1.587631e+00            2.440747e+00            9.293121e+00            -inf                    1.769484e+00            -inf                    9.797113e+00            -inf                    
9      1.012241e-01            7.551118e+00            8.068925e+00            7.749145e+01            7.128459e+00            1.461399e+02            -inf                    7.406058e+00            -inf                    1.267456e+02            -inf                    
10     1.511323e+01            2.023551e+01            6.090790e+01            1.170551e+01            8.074669e+01            7.492835e+01            -inf                    5.890984e+01            -inf                    1.496803e+02            -inf                    
11     8.958428e+00            4.554624e+01            1.756790e+02            2.414315e+01            2.663594e+02            2.162981e+02            -inf                    1.064711e+02            -inf                    5.081544e+02            -inf                    
12     2.145673e+01            2.060514e+01            5.885070e+01            1.994403e+01            7.871185e+01            5.882219e+01            -inf                    3.975865e+01            -inf                    1.044765e+02            -inf                    
13     3.474037e+00            3.286394e+00            1.262618e+01            2.542692e+00            1.628558e+01            2.949959e+01            -inf                    2.372949e+01            -inf                    4.129723e+01            -inf                    
14     2.881758e+00            3.064567e+00            3.666464e+00            4.528716e+00            4.453884e+00            3.776385e+00            -inf                    3.195819e+00            -inf                    4.899352e+00            -inf                    
15     2.651334e+00            2.723299e+00            3.041977e+00            2.822168e+00            3.580167e+00            3.459385e+00            -inf                    3.675500e+00            -inf                    3.981944e+00            -inf                    
16     1.176738e+02            1.333782e+02            3.827491e+02            9.729420e+02            4.169852e+02            8.618726e+03            -inf                    1.830025e+02            -inf                    8.753046e+03            -inf                    
17     4.057364e+01            6.885273e+01            5.176193e+03            2.594138e+01            1.231462e+04            4.890397e+04            -inf                    1.846450e+04            -inf                    1.191772e+05            -inf                    
18     2.094346e+01            2.184319e+01            2.612514e+01            2.066528e+01            2.978728e+01            4.608440e+01            -inf                    4.205587e+01            -inf                    7.098058e+01            -inf                    
19     2.590477e+01            1.428120e+01            7.275808e+01            6.153811e+01            8.035488e+01            1.797448e+02            -inf                    5.847594e+01            -inf                    1.932197e+02            -inf                    
20     1.975945e+01            2.242769e+01            2.359373e+01            2.486853e+01            2.540710e+01            3.321100e+01            -inf                    2.922767e+01            -inf                    3.638619e+01            -inf                    
21     4.552056e+00            4.533663e+00            4.781872e+00            4.594245e+00            5.021031e+00            5.675247e+00            -inf                    4.625980e+00            -inf                    5.656757e+00            -inf                    
22     7.952405e+00            7.946335e+00            1.034719e+01            1.001119e+01            1.291815e+01            1.434087e+01            -inf                    1.441941e+01            -inf                    1.784120e+01            -inf                    
23     1.911667e+01            2.339194e+01            3.390517e+01            3.085083e+01            5.550404e+01            4.378217e+01            -inf                    3.612365e+01            -inf                    4.166417e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=5.532805e-02)
Task  1: original.py  (error=3.987060e-02)
Task  2: original.py  (error=1.508675e-01)
Task  3: original.py  (error=1.354671e-01)
Task  4: original.py  (error=5.237153e-01)
Task  5: original.py  (error=1.640399e-04)
Task  6: original.py  (error=6.414725e-02)
Task  7: original.py  (error=1.300783e-01)
Task  8: original.py  (error=2.563622e-01)
Task  9: original.py  (error=1.012241e-01)
Task 10: variant_03_idea_0.py  (error=1.170551e+01)
Task 11: original.py  (error=8.958428e+00)
Task 12: variant_03_idea_0.py  (error=1.994403e+01)
Task 13: variant_03_idea_0.py  (error=2.542692e+00)
Task 14: original.py  (error=2.881758e+00)
Task 15: original.py  (error=2.651334e+00)
Task 16: original.py  (error=1.176738e+02)
Task 17: variant_03_idea_0.py  (error=2.594138e+01)
Task 18: variant_03_idea_0.py  (error=2.066528e+01)
Task 19: variant_01_idea_0.py  (error=1.428120e+01)
Task 20: original.py  (error=1.975945e+01)
Task 21: variant_01_idea_0.py  (error=4.533663e+00)
Task 22: variant_01_idea_0.py  (error=7.946335e+00)
Task 23: original.py  (error=1.911667e+01)

WIN COUNTS:
  original.py: 16 wins
  variant_03_idea_0.py: 5 wins
  variant_01_idea_0.py: 3 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _sample_trials_batch(self):
        """Sample new trial population using eigendecomposition."""
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Eigendecomposition: C = V @ D @ V.T
        eigvals, eigvecs = np.linalg.eigh(self.C)

        # Robustness: ensure positive eigenvalues and bound condition number
        eigvals = np.maximum(eigvals, 1e-10)
        max_eig = np.max(eigvals)
        if max_eig > 1e6 * np.min(eigvals):
            eigvals = np.clip(eigvals, np.min(eigvals), 1e6 * np.min(eigvals))

        # Sample from standard normal and transform: mean + sigma * V @ sqrt(D) @ z
        z = np.random.randn(self.dim, self.NP)
        D_sqrt = np.sqrt(eigvals)
        self.trials = self.mean[:, np.newaxis] + self.sigma * (eigvecs * D_sqrt) @ z
        self.trials = self.trials.T

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
```

# --- From variant_03_idea_0.py (5 wins) ---
```python
def _sample_trials_batch(self):
        """Sample with eigenvalue-weighted diversity and stagnation-triggered mixing."""
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Eigendecomposition for eigenvalue-weighted sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)

        # Detect stagnation: high condition number means elongated covariance
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1000:
            # Mix with identity to isotropically expand exploration
            mix = min(0.5, (cond - 1000) / 10000)
            C_sampled = (1 - mix) * self.C + mix * np.mean(eigvals) * np.eye(self.dim)
            eigvals_s, eigvecs = np.linalg.eigh(C_sampled)
            eigvals_s = np.maximum(eigvals_s, 1e-10)
        else:
            eigvals_s, eigvecs = eigvals, eigvecs

        # Eigenvalue-weighted sampling: sample more in smaller eigenvalue directions
        # Use power-law weights: smaller eigvals get higher probability
        weights = 1.0 / (eigvals_s ** 0.25 + 1e-10)
        weights /= np.sum(weights)

        # Sample eigenvalue indices for each individual
        n_sample = max(1, int(0.1 * self.NP))
        sampled_eigen_indices = np.random.choice(self.dim, size=n_sample, p=weights, replace=True)

        # Standard normal samples
        z = np.random.randn(self.NP, self.dim)

        # Apply eigenvalue scaling: scale each direction by sqrt(eigenvalue)
        sqrt_eigvals = np.sqrt(eigvals_s)
        scales = np.ones(self.dim)
        for idx in sampled_eigen_indices:
            scales[idx] *= 1.5
        scales = np.clip(scales, 0.5, 2.0)

        # Transform samples: z * sqrt(eigvals) * scales
        transformed = z * (sqrt_eigvals * scales)

        # Apply to mean
        self.trials = self.mean + self.sigma * (transformed @ eigvecs.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)

        # Stagnation detection for restarts
        if hasattr(self, 'stagnation_counter'):
            if self.stagnation_counter > self.max_stagnation // 2:
                # Increase sigma to escape local optima
                self.sigma = min(self.sigma * 2.0, 10.0)
                # Reinitialize poorly-performing individuals
                n_bad = min(self.NP // 2, len(self.population))
                if n_bad > 0:
                    bad_indices = np.argsort(self.fitness)[-n_bad:]
                    for idx in bad_indices:
                        if idx > 0:  # Keep best individual
                            self.population[idx] = np.random.uniform(self.lb, self.ub)
                            self.fitness[idx] = float('inf')
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