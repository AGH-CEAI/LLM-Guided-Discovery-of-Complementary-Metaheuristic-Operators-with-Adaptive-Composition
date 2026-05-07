This is iteration 10 of 10. Propose ONE replacement implementation for `_update_velocity_batch`.

Requirements:
- Keep the EXACT function signature: `def _update_velocity_batch(self, inertia):`
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
  Task 12: *** UNSOLVED *** — best=1.511e+03 by variant_07_idea_0.py      (target=1e-08, ~+11.2 decades above target)
  Task  8: *** UNSOLVED *** — best=3.623e+02 by original.py               (target=1e-08, ~+10.6 decades above target)
  Task 16: *** UNSOLVED *** — best=1.540e+02 by variant_06_idea_0.py      (target=1e-08, ~+10.2 decades above target)
  Task 13: *** UNSOLVED *** — best=1.051e+02 by variant_06_idea_0.py      (target=1e-08, ~+10.0 decades above target)
  Task 21: *** UNSOLVED *** — best=5.718e+01 by original.py               (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=3.710e+01 by variant_05_idea_0.py      (target=1e-08, ~+9.6 decades above target)
  Task  2: *** UNSOLVED *** — best=3.423e+01 by original.py               (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=5.688e+00 by variant_08_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 14: *** UNSOLVED *** — best=5.480e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task  3: *** UNSOLVED *** — best=2.741e+00 by variant_08_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 18: *** UNSOLVED *** — best=2.500e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task  5: *** UNSOLVED *** — best=1.473e+00 by variant_07_idea_0.py      (target=1e-08, ~+8.2 decades above target)
  Task 17: *** UNSOLVED *** — best=1.334e+00 by variant_08_idea_0.py      (target=1e-08, ~+8.1 decades above target)
  Task  4: *** UNSOLVED *** — best=1.312e+00 by variant_07_idea_0.py      (target=1e-08, ~+8.1 decades above target)
  Task 19: *** UNSOLVED *** — best=8.373e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task 22: *** UNSOLVED *** — best=8.209e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task 15: *** UNSOLVED *** — best=7.886e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task 11: *** UNSOLVED *** — best=6.484e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.8 decades above target)
  Task 10: *** UNSOLVED *** — best=5.736e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.8 decades above target)
  Task  9: *** UNSOLVED *** — best=5.635e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.8 decades above target)
  Task  1: *** UNSOLVED *** — best=5.307e-01 by variant_07_idea_0.py      (target=1e-08, ~+7.7 decades above target)
  Task  7: *** UNSOLVED *** — best=4.771e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.7 decades above target)
  Task  0: *** UNSOLVED *** — best=4.049e-01 by original.py               (target=1e-08, ~+7.6 decades above target)
  Task  6: *** UNSOLVED *** — best=3.923e-01 by variant_08_idea_0.py      (target=1e-08, ~+7.6 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  variant_07_idea_0.py  variant_08_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    4.0488e-01            -inf                  4.8233e+02            2.3145e+00            -inf                  1.7822e+00            7.5952e-01            1.2762e+02            4.2929e-01            
1    9.6828e-01            -inf                  1.3761e+00            1.0387e+00            -inf                  1.0225e+00            9.8698e-01            5.3068e-01            9.5840e-01            
2    3.4229e+01            -inf                  1.4730e+08            6.6750e+04            -inf                  4.0541e+05            1.4195e+05            1.0801e+07            1.1859e+04            
3    3.2082e+00            -inf                  3.1791e+03            1.2388e+01            -inf                  8.7285e+00            3.7837e+00            1.2207e+02            2.7412e+00            
4    1.8185e+00            -inf                  2.6367e+00            1.9712e+00            -inf                  1.9994e+00            1.8891e+00            1.3122e+00            1.8437e+00            
5    1.9528e+00            -inf                  2.6584e+00            2.0766e+00            -inf                  2.0286e+00            1.9949e+00            1.4731e+00            1.9469e+00            
6    6.8933e-01            -inf                  8.0361e+02            2.4811e+00            -inf                  1.7557e+00            1.1628e+00            4.8834e+03            3.9228e-01            
7    7.1524e-01            -inf                  8.6388e+02            2.7638e+00            -inf                  1.9196e+00            8.3303e-01            5.0759e+03            4.7706e-01            
8    3.6229e+02            -inf                  1.2043e+04            2.1351e+03            -inf                  1.3872e+03            5.2667e+02            8.8474e+02            1.1441e+03            
9    7.9834e-01            -inf                  9.3125e+02            3.1665e+00            -inf                  2.4028e+00            1.0224e+00            5.2148e+03            5.6345e-01            
10   9.8719e-01            -inf                  1.0255e+03            4.3557e+00            -inf                  2.1499e+00            1.4059e+00            7.0499e+03            5.7359e-01            
11   9.1151e-01            -inf                  1.0286e+03            3.6949e+00            -inf                  2.1554e+00            1.0942e+00            7.6243e+03            6.4842e-01            
12   2.1074e+03            -inf                  1.3744e+04            5.7732e+03            -inf                  7.4647e+03            5.4397e+03            1.5110e+03            6.1037e+03            
13   1.5524e+02            -inf                  4.3224e+03            4.6180e+02            -inf                  1.7801e+02            1.0508e+02            1.0740e+04            1.2515e+02            
14   5.4804e+00            -inf                  6.9522e+00            5.9499e+00            -inf                  5.6188e+00            5.5011e+00            5.7090e+00            5.6037e+00            
15   8.4285e-01            -inf                  1.2181e+03            2.5311e+00            -inf                  4.9552e+00            9.2521e-01            3.4873e+01            7.8862e-01            
16   7.1008e+02            -inf                  5.0418e+04            4.4993e+02            -inf                  3.3083e+02            1.5398e+02            7.9551e+02            1.7322e+02            
17   3.4134e+00            -inf                  2.1403e+03            5.6161e+02            -inf                  3.2551e+00            1.4378e+00            8.8838e+03            1.3344e+00            
18   2.4997e+00            -inf                  4.7476e+03            3.5262e+01            -inf                  5.4652e+02            1.3658e+01            8.8698e+03            5.9127e+02            
19   9.7682e-01            -inf                  5.2766e+00            1.4118e+00            -inf                  1.2444e+00            9.9797e-01            9.1493e+00            8.3733e-01            
20   5.7180e+00            -inf                  2.6592e+01            6.6201e+00            -inf                  6.3202e+00            5.8798e+00            2.3243e+01            5.6879e+00            
21   5.7179e+01            -inf                  1.3280e+04            9.6738e+01            -inf                  6.9901e+01            5.8993e+01            4.5159e+04            6.6491e+01            
22   9.0287e-01            -inf                  1.3496e+01            1.4496e+00            -inf                  1.3112e+00            9.9805e-01            2.6203e+01            8.2092e-01            
23   4.0184e+01            -inf                  7.7875e+01            4.4739e+01            -inf                  3.7095e+01            3.7596e+01            1.2238e+02            4.0958e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 12: best error so far = 1.511e+03  (target = 1e-08)
  Task  8: best error so far = 3.623e+02  (target = 1e-08)
  Task 16: best error so far = 1.540e+02  (target = 1e-08)
  Task 13: best error so far = 1.051e+02  (target = 1e-08)
  Task 21: best error so far = 5.718e+01  (target = 1e-08)
  Task 23: best error so far = 3.710e+01  (target = 1e-08)
  Task  2: best error so far = 3.423e+01  (target = 1e-08)
  Task 20: best error so far = 5.688e+00  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

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
        """Reflective boundary handling - invert velocity at bounds to preserve momentum."""
        reflected = pop.copy()
        # Below lower bound: reflect back inward
        below_lower = reflected < self.lower
        reflected[below_lower] = 2.0 * self.lower - reflected[below_lower]
        # Above upper bound: reflect back inward
        above_upper = reflected > self.upper
        reflected[above_upper] = 2.0 * self.upper - reflected[above_upper]
        # Final hard clip for any double-reflections (extreme cases)
        return np.clip(reflected, self.lower, self.upper)
    
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
def _update_velocity_batch(self, ...):
    ...
```