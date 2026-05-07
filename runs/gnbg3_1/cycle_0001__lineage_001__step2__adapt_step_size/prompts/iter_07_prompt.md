This is iteration 7 of 10. Propose ONE replacement implementation for `_adapt_step_size`.

Requirements:
- Keep the EXACT function signature: `def _adapt_step_size(self):`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that reaches error <= 1e-08 on EACH task.
- Tasks marked UNSOLVED below are critical gaps. The bigger the remaining
  error, the higher the priority — your variant should specifically target
  the WORST unsolved tasks at the top of the priority list.

TASK COVERAGE SUMMARY: 0 SOLVED (<= 1e-08), 24 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 17: *** UNSOLVED *** — best=2.512e+02 by variant_02_idea_0.py      (target=1e-08, ~+10.4 decades above target)
  Task 16: *** UNSOLVED *** — best=7.352e+01 by variant_05_idea_0.py      (target=1e-08, ~+9.9 decades above target)
  Task 20: *** UNSOLVED *** — best=1.925e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.3 decades above target)
  Task 18: *** UNSOLVED *** — best=1.716e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 12: *** UNSOLVED *** — best=1.709e+01 by variant_02_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 10: *** UNSOLVED *** — best=1.607e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 23: *** UNSOLVED *** — best=1.489e+01 by original.py               (target=1e-08, ~+9.2 decades above target)
  Task 11: *** UNSOLVED *** — best=1.444e+01 by variant_02_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 19: *** UNSOLVED *** — best=1.424e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task  6: *** UNSOLVED *** — best=9.457e+00 by variant_02_idea_0.py      (target=1e-08, ~+9.0 decades above target)
  Task 22: *** UNSOLVED *** — best=6.447e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 21: *** UNSOLVED *** — best=4.396e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.6 decades above target)
  Task  9: *** UNSOLVED *** — best=3.833e+00 by original.py               (target=1e-08, ~+8.6 decades above target)
  Task 13: *** UNSOLVED *** — best=3.109e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.5 decades above target)
  Task 14: *** UNSOLVED *** — best=2.662e+00 by variant_04_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 15: *** UNSOLVED *** — best=2.437e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task  3: *** UNSOLVED *** — best=6.049e-01 by variant_04_idea_0.py      (target=1e-08, ~+7.8 decades above target)
  Task  4: *** UNSOLVED *** — best=5.316e-01 by original.py               (target=1e-08, ~+7.7 decades above target)
  Task  8: *** UNSOLVED *** — best=4.864e-01 by original.py               (target=1e-08, ~+7.7 decades above target)
  Task  2: *** UNSOLVED *** — best=1.937e-01 by original.py               (target=1e-08, ~+7.3 decades above target)
  Task  7: *** UNSOLVED *** — best=1.347e-01 by original.py               (target=1e-08, ~+7.1 decades above target)
  Task  0: *** UNSOLVED *** — best=5.295e-02 by original.py               (target=1e-08, ~+6.7 decades above target)
  Task  1: *** UNSOLVED *** — best=4.461e-02 by original.py               (target=1e-08, ~+6.6 decades above target)
  Task  5: *** UNSOLVED *** — best=1.509e-04 by variant_04_idea_0.py      (target=1e-08, ~+4.2 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  
---------------------------------------------------------------------------------------------------------------------------------------------------------------
0    5.2954e-02            -inf                  6.0975e-02            -inf                  5.4031e-02            7.7903e-02            7.2130e-02            
1    4.4614e-02            -inf                  4.8228e-02            -inf                  4.8759e-02            1.5416e-01            1.9633e-01            
2    1.9373e-01            -inf                  3.6636e-01            -inf                  7.9745e-01            4.5352e+00            2.4710e+00            
3    7.9266e-01            -inf                  6.1757e-01            -inf                  6.0485e-01            3.3676e+00            3.1380e+00            
4    5.3165e-01            -inf                  5.5437e-01            -inf                  5.5598e-01            9.9643e-01            1.0286e+00            
5    3.9007e-04            -inf                  2.4715e-03            -inf                  1.5087e-04            9.8340e+00            6.7048e+00            
6    1.7607e+01            -inf                  9.4571e+00            -inf                  1.2741e+01            2.5561e+01            1.0577e+01            
7    1.3473e-01            -inf                  1.4815e-01            -inf                  1.3649e-01            2.2232e+00            1.2936e+00            
8    4.8637e-01            -inf                  1.3509e+00            -inf                  5.4804e-01            2.3060e+00            2.2416e+00            
9    3.8332e+00            -inf                  5.5813e+00            -inf                  5.5498e+00            6.9928e+00            5.4429e+00            
10   3.0195e+01            -inf                  2.2291e+01            -inf                  2.9368e+01            2.3395e+01            1.6067e+01            
11   4.0668e+01            -inf                  1.4436e+01            -inf                  4.9307e+01            5.6651e+01            4.9236e+01            
12   2.7167e+01            -inf                  1.7093e+01            -inf                  3.0347e+01            4.1353e+01            2.1338e+01            
13   4.8010e+00            -inf                  3.1095e+00            -inf                  5.1575e+00            3.3386e+00            4.1903e+00            
14   2.7788e+00            -inf                  2.6844e+00            -inf                  2.6621e+00            2.8063e+00            2.7463e+00            
15   2.7774e+00            -inf                  2.6713e+00            -inf                  2.7845e+00            2.9807e+00            2.4375e+00            
16   1.5634e+02            -inf                  1.1078e+02            -inf                  1.4665e+02            7.3518e+01            1.3029e+02            
17   1.0463e+03            -inf                  2.5118e+02            -inf                  1.9775e+03            2.5336e+03            8.3256e+02            
18   2.1989e+01            -inf                  2.0804e+01            -inf                  2.5692e+01            2.1735e+01            1.7159e+01            
19   1.6093e+01            -inf                  1.7759e+01            -inf                  1.4721e+01            1.5667e+01            1.4235e+01            
20   2.0516e+01            -inf                  2.2408e+01            -inf                  2.1769e+01            2.0681e+01            1.9253e+01            
21   4.5393e+00            -inf                  4.3962e+00            -inf                  4.5038e+00            4.5112e+00            4.5352e+00            
22   9.5190e+00            -inf                  8.4669e+00            -inf                  1.0788e+01            9.3251e+00            6.4471e+00            
23   1.4890e+01            -inf                  1.9349e+01            -inf                  2.1899e+01            1.6578e+01            1.9070e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 2.512e+02  (target = 1e-08)
  Task 16: best error so far = 7.352e+01  (target = 1e-08)
  Task 20: best error so far = 1.925e+01  (target = 1e-08)
  Task 18: best error so far = 1.716e+01  (target = 1e-08)
  Task 12: best error so far = 1.709e+01  (target = 1e-08)
  Task 10: best error so far = 1.607e+01  (target = 1e-08)
  Task 23: best error so far = 1.489e+01  (target = 1e-08)
  Task 11: best error so far = 1.444e+01  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to crush the error on the WORST unsolved
tasks above. It is acceptable — even expected — for the new variant to be
worse than existing variants on already-SOLVED tasks; the adaptive selector
will handle that. Reason explicitly about the priority targets:
1. What property of those WORST unsolved tasks (multimodality, ill-conditioning,
   separability, ruggedness, noise, deceptive local optima, narrow basins,
   non-separable rotation, etc.) is preventing existing operators from reaching
   1e-08? Use the per-task error magnitudes as evidence — errors
   stuck at ~1e+1 vs ~1e-3 vs ~1e-6 imply different failure modes.
2. What specific mechanism in your proposed operator is designed to break
   through that exact obstacle and push the error several orders of magnitude
   lower?
3. Why is this approach fundamentally different from the prior variants —
   especially from whichever variant currently holds the best (but still
   insufficient) error on the priority tasks?

Current implementation:
```python
def _adapt_step_size(self):
        """Adapt step-size using evolution path (CSA - Cumulation)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damping)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```

Full algorithm for context:
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

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _adapt_step_size(self, ...):
    ...
```