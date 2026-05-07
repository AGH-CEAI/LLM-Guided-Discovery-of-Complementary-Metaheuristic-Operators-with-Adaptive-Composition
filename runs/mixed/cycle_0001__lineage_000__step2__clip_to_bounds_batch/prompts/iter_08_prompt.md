This is iteration 8 of 10. Propose ONE replacement implementation for `_clip_to_bounds_batch`.

Requirements:
- Keep the EXACT function signature: `def _clip_to_bounds_batch(self, pop):`
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
  Task 12: *** UNSOLVED *** — best=1.920e+03 by variant_04_idea_0.py      (target=1e-08, ~+11.3 decades above target)
  Task  8: *** UNSOLVED *** — best=3.478e+02 by variant_04_idea_0.py      (target=1e-08, ~+10.5 decades above target)
  Task 13: *** UNSOLVED *** — best=1.245e+02 by variant_04_idea_0.py      (target=1e-08, ~+10.1 decades above target)
  Task 21: *** UNSOLVED *** — best=5.664e+01 by variant_05_idea_0.py      (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=3.208e+01 by variant_04_idea_0.py      (target=1e-08, ~+9.5 decades above target)
  Task 16: *** UNSOLVED *** — best=3.004e+01 by variant_03_idea_0.py      (target=1e-08, ~+9.5 decades above target)
  Task  2: *** UNSOLVED *** — best=7.950e+00 by variant_03_idea_0.py      (target=1e-08, ~+8.9 decades above target)
  Task 20: *** UNSOLVED *** — best=5.684e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 14: *** UNSOLVED *** — best=5.415e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 18: *** UNSOLVED *** — best=2.687e+00 by variant_04_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task  5: *** UNSOLVED *** — best=1.895e+00 by variant_03_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task 17: *** UNSOLVED *** — best=1.838e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task  4: *** UNSOLVED *** — best=1.719e+00 by variant_03_idea_0.py      (target=1e-08, ~+8.2 decades above target)
  Task 19: *** UNSOLVED *** — best=9.904e-01 by variant_01_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task  3: *** UNSOLVED *** — best=9.635e-01 by variant_03_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task 22: *** UNSOLVED *** — best=8.854e-01 by variant_05_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task  1: *** UNSOLVED *** — best=8.847e-01 by variant_06_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task 11: *** UNSOLVED *** — best=8.060e-01 by variant_01_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task 10: *** UNSOLVED *** — best=7.461e-01 by variant_03_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task  9: *** UNSOLVED *** — best=5.301e-01 by variant_03_idea_0.py      (target=1e-08, ~+7.7 decades above target)
  Task  7: *** UNSOLVED *** — best=2.014e-01 by variant_06_idea_0.py      (target=1e-08, ~+7.3 decades above target)
  Task  6: *** UNSOLVED *** — best=1.665e-01 by variant_06_idea_0.py      (target=1e-08, ~+7.2 decades above target)
  Task  0: *** UNSOLVED *** — best=8.990e-02 by variant_03_idea_0.py      (target=1e-08, ~+7.0 decades above target)
  Task 15: *** UNSOLVED *** — best=5.253e-02 by variant_06_idea_0.py      (target=1e-08, ~+6.7 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  variant_07_idea_0.py  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    5.0641e-01            4.8178e-01            1.0233e+00            8.9902e-02            5.1276e-01            5.5122e-01            9.7948e-02            5.4584e-01            
1    9.7110e-01            9.7047e-01            1.0020e+00            8.8879e-01            9.7219e-01            9.6862e-01            8.8472e-01            9.7322e-01            
2    2.1086e+04            6.5995e+01            2.9345e+04            7.9501e+00            4.5659e+01            1.7473e+01            1.0675e+01            1.3381e+03            
3    2.9553e+00            3.0166e+00            6.7095e+00            9.6352e-01            2.3215e+00            3.1373e+00            1.0286e+00            2.9773e+00            
4    1.8181e+00            1.8120e+00            1.8876e+00            1.7190e+00            1.7958e+00            1.8321e+00            1.7399e+00            1.7966e+00            
5    1.9327e+00            1.9543e+00            1.9862e+00            1.8946e+00            1.9356e+00            1.9447e+00            1.8996e+00            1.9320e+00            
6    7.3677e-01            5.8042e-01            1.5570e+00            1.8564e-01            6.3155e-01            6.5488e-01            1.6654e-01            6.2744e-01            
7    7.9918e-01            7.2037e-01            1.3734e+00            2.3160e-01            6.2447e-01            7.9776e-01            2.0138e-01            6.5832e-01            
8    6.0777e+02            4.5969e+02            7.2459e+02            4.3060e+02            3.4780e+02            4.0659e+02            3.8271e+02            4.1885e+02            
9    1.1093e+00            6.4968e-01            6.2144e+02            5.3012e-01            9.4339e-01            8.3199e-01            6.3765e-01            9.2839e-01            
10   1.1843e+01            8.0512e-01            5.4529e+03            7.4606e-01            8.9902e-01            8.1719e-01            1.2191e+01            1.9792e+00            
11   2.9487e+00            8.0604e-01            4.6402e+03            9.9868e+00            8.6981e-01            9.3521e-01            9.7518e-01            1.2046e+00            
12   2.9767e+03            2.2632e+03            9.5857e+03            2.3882e+03            1.9196e+03            2.3863e+03            2.4050e+03            2.2225e+03            
13   2.1242e+03            1.9679e+02            8.3151e+03            1.5259e+03            1.2446e+02            1.5234e+02            1.9446e+03            4.8599e+02            
14   5.6346e+00            5.4364e+00            5.7554e+00            5.6290e+00            5.4485e+00            5.4152e+00            5.5766e+00            5.7717e+00            
15   1.4140e+00            7.6320e-01            7.5658e+00            1.1607e-01            2.0996e+00            3.7765e+00            5.2531e-02            6.9198e-01            
16   7.3523e+02            7.2492e+02            2.2295e+02            3.0039e+01            1.3122e+02            5.6427e+01            4.2110e+01            7.5268e+02            
17   3.4115e+02            4.9262e+00            5.9308e+03            6.2254e+02            1.2498e+01            1.8381e+00            7.7322e+02            2.5533e+00            
18   6.1411e+02            7.2562e+00            3.5589e+03            4.3776e+02            2.6872e+00            3.6854e+02            9.9684e+02            8.5881e+00            
19   4.5711e+00            9.9043e-01            1.2966e+01            3.2407e+00            9.9963e-01            1.0477e+00            2.9864e+00            1.0264e+00            
20   5.7468e+00            5.7318e+00            5.6840e+00            5.7728e+00            5.6926e+00            5.7242e+00            5.7148e+00            5.7306e+00            
21   1.7418e+03            6.1412e+01            1.2604e+04            3.6311e+04            6.9374e+01            5.6645e+01            4.0551e+04            6.3522e+01            
22   9.8645e-01            9.1464e-01            2.7558e+01            9.5777e-01            9.6296e-01            8.8538e-01            1.7876e+00            1.0215e+00            
23   6.1696e+01            3.7508e+01            9.6574e+01            6.1035e+01            3.2076e+01            3.7337e+01            5.9450e+01            4.7576e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 12: best error so far = 1.920e+03  (target = 1e-08)
  Task  8: best error so far = 3.478e+02  (target = 1e-08)
  Task 13: best error so far = 1.245e+02  (target = 1e-08)
  Task 21: best error so far = 5.664e+01  (target = 1e-08)
  Task 23: best error so far = 3.208e+01  (target = 1e-08)
  Task 16: best error so far = 3.004e+01  (target = 1e-08)
  Task  2: best error so far = 7.950e+00  (target = 1e-08)
  Task 20: best error so far = 5.684e+00  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

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
def _clip_to_bounds_batch(self, pop):
        """Clip population to search bounds."""
        return np.clip(pop, self.lower, self.upper)
```

Full algorithm for context:
```python
import numpy as np


class QuantumRingSwarmOptimizer:
    """
    A PSO-based optimizer with dynamic ring topology and quantum tunneling mutation.
    
    Key components:
    - Ring lattice topology with k-nearest neighborhood
    - Quantum tunneling mutation triggered by stagnation
    - Adaptive inertia weight based on progress ratio
    - Diversity-based reinitialization of stagnant particles
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = kwargs.get('pop_size', 6 * dim)  # 180 for dim=30
        self.lower = kwargs.get('lower', -100.0)
        self.upper = kwargs.get('upper', 100.0)
        
        self.k_neighbors = kwargs.get('k_neighbors', 3)
        self.w_max = kwargs.get('w_max', 0.9)
        self.w_min = kwargs.get('w_min', 0.4)
        self.c1 = kwargs.get('c1', 1.5)  # personal weight
        self.c2 = kwargs.get('c2', 1.5)  # social weight
        self.c3 = kwargs.get('c3', 2.0)  # neighborhood weight
        
        self.tunnel_prob = kwargs.get('tunnel_prob', 0.1)
        self.tunnel_scale = kwargs.get('tunnel_scale', 0.3)
        self.stagnation_limit = kwargs.get('stagnation_limit', 15)
        self.diversity_threshold = kwargs.get('diversity_threshold', 1e-6)
        
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fit = None
        self.neighborhood_best = None
        self.global_best = None
        self.global_best_fit = None
        self.stagnation_counter = np.zeros(self.pop_size, dtype=np.int32)
        self.fitness_history = []
        self.step_size = 1.0
        
    def _initialize_population(self):
        """Initialize population uniformly in search space with zero velocity."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.pop_size, self.dim)
        )
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fit = np.full(self.pop_size, np.inf)
        self.neighborhood_best = np.zeros((self.pop_size, self.dim))
        self.global_best = None
        self.global_best_fit = np.inf
        self.stagnation_counter[:] = 0
        self.fitness_history = []
        self.step_size = (self.upper - self.lower) * 0.1
        
    def _clip_to_bounds_batch(self, pop):
        """Clip population to search bounds."""
        return np.clip(pop, self.lower, self.upper)
    
    def _compute_ring_topology(self):
        """Compute ring topology indices: each particle connects to k neighbors on each side."""
        indices = np.arange(self.pop_size)
        for i in range(self.pop_size):
            left_neighbors = (indices[i - self.k_neighbors: i] % self.pop_size)
            right_neighbors = (indices[i + 1: i + self.k_neighbors + 1] % self.pop_size)
            neighbors = np.concatenate([left_neighbors, right_neighbors])
            all_neighbors = np.concatenate([neighbors, [i]])
            best_idx = np.argmin(self.personal_best_fit[all_neighbors])
            self.neighborhood_best[i] = self.personal_best[all_neighbors[best_idx]]
            
    def _evaluate_batch(self, pop, func):
        """Evaluate fitness for entire population in batch."""
        clipped = self._clip_to_bounds_batch(pop)
        fitness = func(clipped)
        if len(fitness) < len(pop):
            fitness = np.pad(fitness, (0, len(pop) - len(fitness)), 
                           constant_values=np.inf)
        return fitness
    
    def _update_personal_best_batch(self, fitness):
        """Update personal best positions where new fitness is better."""
        improved = fitness < self.personal_best_fit
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fit[improved] = fitness[improved]
        
    def _update_global_best_batch(self):
        """Update global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fit)
        if self.personal_best_fit[best_idx] < self.global_best_fit:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fit = self.personal_best_fit[best_idx]
            self.stagnation_counter[:] = 0
        else:
            self.stagnation_counter += 1
            
    def _compute_progress_ratio(self):
        """Compute progress ratio for adaptive inertia."""
        if len(self.fitness_history) < 2:
            return 0.5
        recent = np.mean(self.fitness_history[-5:])
        older = np.mean(self.fitness_history[-15:-5]) if len(self.fitness_history) > 15 else recent
        if older - recent < 1e-12:
            return 1.0
        ratio = min((older - recent) / (abs(older) + 1e-10), 2.0)
        return ratio
    
    def _compute_adaptive_inertia(self):
        """Compute inertia weight based on progress ratio."""
        progress = self._compute_progress_ratio()
        inertia = self.w_max - (self.w_max - self.w_min) * progress
        return np.clip(inertia, self.w_min, self.w_max)
    
    def _quantum_tunnel_batch(self):
        """Apply quantum tunneling mutation to stagnant particles."""
        stagnant_mask = self.stagnation_counter >= self.stagnation_limit
        if not np.any(stagnant_mask):
            return
        
        n_stagnant = np.sum(stagnant_mask)
        tunnel_distance = (self.upper - self.lower) * self.tunnel_scale * self.step_size
        
        tunnel_pop = np.copy(self.population)
        n_dims_to_modify = max(1, self.dim // 5)
        for i in np.where(stagnant_mask)[0]:
            dims = np.random.choice(self.dim, n_dims_to_modify, replace=False)
            tunnel_pop[i, dims] += np.random.uniform(
                -tunnel_distance, tunnel_distance, size=n_dims_to_modify
            )
        tunnel_pop = self._clip_to_bounds_batch(tunnel_pop)
        
        tunnel_fitness = self.personal_best_fit.copy()
        for i in np.where(stagnant_mask)[0]:
            trial = tunnel_pop[i:i+1]
            fit = self._evaluate_batch(trial, lambda x: self._eval_wrapper(x))
            tunnel_fitness[i] = fit[0]
            
        improved = tunnel_fitness < self.personal_best_fit
        self.population[improved & stagnant_mask] = tunnel_pop[improved & stagnant_mask]
        self.velocity[improved & stagnant_mask] *= 0.5
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fit[improved] = tunnel_fitness[improved]
        self.stagnation_counter[improved & stagnant_mask] = 0
        
    def _eval_wrapper(self, x):
        """Robust wrapper with bounds enforcement and NaN/Inf penalty."""
        x = np.atleast_2d(x)
        x_clipped = np.clip(x, self.lower, self.upper)
        result = self._user_func(x_clipped)
        # Penalty for non-finite values (NaN/Inf) to avoid breaking optimization
        result = np.where(np.isfinite(result), result, 1e30)
        return result
    
    def _update_velocity_batch(self, inertia):
        """Update velocity using cognitive, social, and neighborhood components."""
        r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        
        cognitive = self.c1 * r1 * (self.personal_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)
        
        self.velocity = inertia * self.velocity + cognitive + social + neighborhood
        
        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
        
    def _update_position_batch(self):
        """Update positions by adding velocity."""
        self.population = self.population + self.velocity
        
    def _reinitialize_diverse_particles(self):
        """Reinitialize particles if population diversity is too low."""
        diversity = np.std(self.population, axis=0).mean()
        if diversity < self.diversity_threshold * (self.upper - self.lower):
            n_reinit = max(1, self.pop_size // 5)
            indices = np.random.choice(self.pop_size, n_reinit, replace=False)
            self.population[indices] = np.random.uniform(
                self.lower, self.upper, size=(n_reinit, self.dim)
            )
            self.velocity[indices] = 0.0
            
    def _adapt_step_size(self):
        """Adapt global step size based on convergence behavior."""
        if len(self.fitness_history) < 10:
            return
        recent_std = np.std(self.fitness_history[-10:])
        if recent_std < 1e-8:
            self.step_size = min(self.step_size * 1.1, 2.0)
        else:
            self.step_size = max(self.step_size * 0.95, 0.1)
            
    def _compute_diversity(self):
        """Compute population diversity measure."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def __call__(self, func, stopping_condition):
        """
        Run optimization.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._initialize_population()
        self._user_func = func
        generation = 0
        
        while not stopping_condition():
            fitness = self._evaluate_batch(self.population, func)
            
            if len(fitness) < len(self.population):
                fitness = np.pad(fitness, (0, len(self.population) - len(fitness)),
                               constant_values=np.inf)
                break
                
            self._update_personal_best_batch(fitness)
            self._update_global_best_batch()
            self.fitness_history.append(self.global_best_fit)
            
            self._compute_ring_topology()
            self._quantum_tunnel_batch()
            
            inertia = self._compute_adaptive_inertia()
            self._update_velocity_batch(inertia)
            self._update_position_batch()
            self.population = self._clip_to_bounds_batch(self.population)
            
            self._reinitialize_diverse_particles()
            self._adapt_step_size()
            
            generation += 1
            
            if generation % 50 == 0:
                div = self._compute_diversity()
                
        return self.global_best_fit, self.global_best

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _clip_to_bounds_batch(self, ...):
    ...
```