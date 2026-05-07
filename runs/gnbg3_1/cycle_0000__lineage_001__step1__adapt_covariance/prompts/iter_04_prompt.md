This is iteration 4 of 10. Propose ONE replacement implementation for `_adapt_covariance`.

Requirements:
- Keep the EXACT function signature: `def _adapt_covariance(self):`
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
  Task 17: *** UNSOLVED *** — best=1.246e+03 by variant_01_idea_0.py      (target=1e-08, ~+11.1 decades above target)
  Task 16: *** UNSOLVED *** — best=3.628e+02 by original.py               (target=1e-08, ~+10.6 decades above target)
  Task  6: *** UNSOLVED *** — best=2.366e+02 by variant_01_idea_0.py      (target=1e-08, ~+10.4 decades above target)
  Task 23: *** UNSOLVED *** — best=3.114e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.5 decades above target)
  Task 19: *** UNSOLVED *** — best=2.555e+01 by original.py               (target=1e-08, ~+9.4 decades above target)
  Task 20: *** UNSOLVED *** — best=2.441e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.4 decades above target)
  Task 11: *** UNSOLVED *** — best=2.211e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.3 decades above target)
  Task 18: *** UNSOLVED *** — best=2.091e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.3 decades above target)
  Task 10: *** UNSOLVED *** — best=2.065e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.3 decades above target)
  Task 12: *** UNSOLVED *** — best=1.480e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 22: *** UNSOLVED *** — best=1.066e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.0 decades above target)
  Task  9: *** UNSOLVED *** — best=7.444e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.9 decades above target)
  Task 13: *** UNSOLVED *** — best=5.854e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 21: *** UNSOLVED *** — best=4.529e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task  2: *** UNSOLVED *** — best=4.121e+00 by original.py               (target=1e-08, ~+8.6 decades above target)
  Task 15: *** UNSOLVED *** — best=2.665e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 14: *** UNSOLVED *** — best=2.665e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task  7: *** UNSOLVED *** — best=2.208e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task  8: *** UNSOLVED *** — best=1.783e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task  3: *** UNSOLVED *** — best=1.369e+00 by original.py               (target=1e-08, ~+8.1 decades above target)
  Task  4: *** UNSOLVED *** — best=9.403e-01 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task  5: *** UNSOLVED *** — best=3.556e-01 by original.py               (target=1e-08, ~+7.6 decades above target)
  Task  1: *** UNSOLVED *** — best=5.639e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.8 decades above target)
  Task  0: *** UNSOLVED *** — best=5.338e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.7 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  
---------------------------------------------------------------------------------------------
0    1.6361e-01            5.3375e-02            6.8063e+01            -inf                  
1    1.5493e-01            5.6385e-02            1.2719e+02            -inf                  
2    4.1211e+00            4.9589e+00            4.8911e+02            -inf                  
3    1.3685e+00            2.6584e+00            2.4652e+02            -inf                  
4    9.4031e-01            1.3164e+00            5.1513e+00            -inf                  
5    3.5560e-01            2.9859e+01            6.4854e+08            -inf                  
6    3.7816e+02            2.3659e+02            3.1689e+03            -inf                  
7    2.2083e+00            2.8868e+00            3.2697e+02            -inf                  
8    1.7834e+00            1.9401e+00            4.6640e+01            -inf                  
9    7.9509e+00            7.4439e+00            2.2631e+02            -inf                  
10   3.5795e+01            2.0650e+01            2.4550e+02            -inf                  
11   3.3814e+01            2.2115e+01            8.8057e+02            -inf                  
12   2.2037e+01            1.4801e+01            1.8975e+02            -inf                  
13   8.5349e+00            5.8545e+00            4.5310e+01            -inf                  
14   2.6646e+00            3.2961e+00            1.4484e+01            -inf                  
15   2.7827e+00            2.6651e+00            3.9617e+00            -inf                  
16   3.6278e+02            4.3253e+02            1.7415e+04            -inf                  
17   1.9828e+03            1.2462e+03            1.9023e+05            -inf                  
18   2.4233e+01            2.0914e+01            7.7297e+01            -inf                  
19   2.5546e+01            3.4931e+01            2.9986e+02            -inf                  
20   2.7818e+01            2.4409e+01            5.0212e+01            -inf                  
21   4.5807e+00            4.5292e+00            5.8306e+00            -inf                  
22   1.1590e+01            1.0661e+01            1.8274e+01            -inf                  
23   3.3042e+01            3.1139e+01            7.9897e+01            -inf                  

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 1.246e+03  (target = 1e-08)
  Task 16: best error so far = 3.628e+02  (target = 1e-08)
  Task  6: best error so far = 2.366e+02  (target = 1e-08)
  Task 23: best error so far = 3.114e+01  (target = 1e-08)
  Task 19: best error so far = 2.555e+01  (target = 1e-08)
  Task 20: best error so far = 2.441e+01  (target = 1e-08)
  Task 11: best error so far = 2.211e+01  (target = 1e-08)
  Task 18: best error so far = 2.091e+01  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0

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
def _adapt_covariance(self):
        """
        Adapt covariance matrix using rank-1 and rank-mu updates.
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        # Ensure positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-10:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
```

Full algorithm for context:
```python
import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    A novel optimizer combining CMA-ES-style covariance adaptation
    with a particle-swarm-inspired mean tracking mechanism.
    
    Key features:
    - Simplified full covariance matrix adaptation (not diagonal-only)
    - Dual evolution paths for step-size and covariance
    - Rank-based weighted recombination
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        # Population size: 8 * dim (balanced for dim=30 -> NP=240)
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
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
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
            self._adapt_covariance()
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        # Apply LHS for better spread
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Initialize covariance from population spread
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        # Evaluate initial population
        self.fitness = self.func(self.population)
        
        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
    
    def _sample_trials_batch(self):
        """Sample new trial population from multivariate normal."""
        # Eigendecomposition for stable sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Sample from N(mean, sigma^2 * C)
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = np.clip(self.trials, self.lb, self.ub)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        if self.trial_fitness[trial_best_idx] < self.f_opt:
            self.f_opt = self.trial_fitness[trial_best_idx]
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
        """
        Adapt step-size using evolution path (CSA - Cumulation).
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damping)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_covariance(self):
        """
        Adapt covariance matrix using rank-1 and rank-mu updates.
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        # Ensure positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-10:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
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
            
            # Keep best individual, reinitialize rest
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            # Inject elite into new population
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = elite_fit
            self.x_opt = elite.copy()
            self.f_opt_prev = elite_fit

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _adapt_covariance(self, ...):
    ...
```