This is iteration 2 of 10. Propose ONE replacement implementation for `_update_velocity_batch`.

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
  Task 12: *** UNSOLVED *** — best=2.107e+03 by original.py               (target=1e-08, ~+11.3 decades above target)
  Task 16: *** UNSOLVED *** — best=7.101e+02 by original.py               (target=1e-08, ~+10.9 decades above target)
  Task  8: *** UNSOLVED *** — best=3.623e+02 by original.py               (target=1e-08, ~+10.6 decades above target)
  Task 13: *** UNSOLVED *** — best=1.552e+02 by original.py               (target=1e-08, ~+10.2 decades above target)
  Task 21: *** UNSOLVED *** — best=5.718e+01 by original.py               (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=4.018e+01 by original.py               (target=1e-08, ~+9.6 decades above target)
  Task  2: *** UNSOLVED *** — best=3.423e+01 by original.py               (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=5.718e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 14: *** UNSOLVED *** — best=5.480e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task 17: *** UNSOLVED *** — best=3.413e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task  3: *** UNSOLVED *** — best=3.208e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.500e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task  5: *** UNSOLVED *** — best=1.953e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task  4: *** UNSOLVED *** — best=1.818e+00 by original.py               (target=1e-08, ~+8.3 decades above target)
  Task 10: *** UNSOLVED *** — best=9.872e-01 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task 19: *** UNSOLVED *** — best=9.768e-01 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task  1: *** UNSOLVED *** — best=9.683e-01 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task 11: *** UNSOLVED *** — best=9.115e-01 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task 22: *** UNSOLVED *** — best=9.029e-01 by original.py               (target=1e-08, ~+8.0 decades above target)
  Task 15: *** UNSOLVED *** — best=8.429e-01 by original.py               (target=1e-08, ~+7.9 decades above target)
  Task  9: *** UNSOLVED *** — best=7.983e-01 by original.py               (target=1e-08, ~+7.9 decades above target)
  Task  7: *** UNSOLVED *** — best=7.152e-01 by original.py               (target=1e-08, ~+7.9 decades above target)
  Task  6: *** UNSOLVED *** — best=6.893e-01 by original.py               (target=1e-08, ~+7.8 decades above target)
  Task  0: *** UNSOLVED *** — best=4.049e-01 by original.py               (target=1e-08, ~+7.6 decades above target)

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  
-------------------------------------------------
0    4.0488e-01            -inf                  
1    9.6828e-01            -inf                  
2    3.4229e+01            -inf                  
3    3.2082e+00            -inf                  
4    1.8185e+00            -inf                  
5    1.9528e+00            -inf                  
6    6.8933e-01            -inf                  
7    7.1524e-01            -inf                  
8    3.6229e+02            -inf                  
9    7.9834e-01            -inf                  
10   9.8719e-01            -inf                  
11   9.1151e-01            -inf                  
12   2.1074e+03            -inf                  
13   1.5524e+02            -inf                  
14   5.4804e+00            -inf                  
15   8.4285e-01            -inf                  
16   7.1008e+02            -inf                  
17   3.4134e+00            -inf                  
18   2.4997e+00            -inf                  
19   9.7682e-01            -inf                  
20   5.7180e+00            -inf                  
21   5.7179e+01            -inf                  
22   9.0287e-01            -inf                  
23   4.0184e+01            -inf                  

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 12: best error so far = 2.107e+03  (target = 1e-08)
  Task 16: best error so far = 7.101e+02  (target = 1e-08)
  Task  8: best error so far = 3.623e+02  (target = 1e-08)
  Task 13: best error so far = 1.552e+02  (target = 1e-08)
  Task 21: best error so far = 5.718e+01  (target = 1e-08)
  Task 23: best error so far = 4.018e+01  (target = 1e-08)
  Task  2: best error so far = 3.423e+01  (target = 1e-08)
  Task 20: best error so far = 5.718e+00  (target = 1e-08)
  ... and 16 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0

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