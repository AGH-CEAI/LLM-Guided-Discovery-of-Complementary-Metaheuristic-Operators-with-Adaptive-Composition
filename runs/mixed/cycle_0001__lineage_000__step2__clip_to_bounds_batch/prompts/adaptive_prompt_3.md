Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_clip_to_bounds_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.064143e-01            4.817751e-01            1.023258e+00            8.990243e-02            5.127558e-01            5.512179e-01            9.794817e-02            5.458411e-01            4.048779e-01            4.681595e-01            5.442819e-01            
1      9.711021e-01            9.704660e-01            1.001997e+00            8.887857e-01            9.721873e-01            9.686182e-01            8.847170e-01            9.732214e-01            9.682844e-01            9.724469e-01            9.739160e-01            
2      2.108573e+04            6.599520e+01            2.934536e+04            7.950062e+00            4.565889e+01            1.747311e+01            1.067517e+01            1.338150e+03            3.422938e+01            1.750796e+04            1.658687e+03            
3      2.955297e+00            3.016640e+00            6.709484e+00            9.635242e-01            2.321492e+00            3.137343e+00            1.028630e+00            2.977254e+00            3.208152e+00            2.820888e+00            2.916515e+00            
4      1.818109e+00            1.812034e+00            1.887604e+00            1.719008e+00            1.795802e+00            1.832065e+00            1.739922e+00            1.796556e+00            1.818475e+00            1.828703e+00            1.794048e+00            
5      1.932659e+00            1.954271e+00            1.986235e+00            1.894599e+00            1.935616e+00            1.944748e+00            1.899569e+00            1.931983e+00            1.952838e+00            1.959458e+00            1.976221e+00            
6      7.367680e-01            5.804176e-01            1.556965e+00            1.856395e-01            6.315515e-01            6.548785e-01            1.665381e-01            6.274392e-01            6.893326e-01            6.457878e-01            6.370692e-01            
7      7.991788e-01            7.203704e-01            1.373428e+00            2.315998e-01            6.244660e-01            7.977595e-01            2.013815e-01            6.583236e-01            7.152444e-01            8.139403e-01            8.288412e-01            
8      6.077746e+02            4.596882e+02            7.245895e+02            4.306013e+02            3.477996e+02            4.065920e+02            3.827123e+02            4.188533e+02            3.622913e+02            5.430377e+02            4.496008e+02            
9      1.109315e+00            6.496791e-01            6.214404e+02            5.301205e-01            9.433912e-01            8.319912e-01            6.376506e-01            9.283920e-01            7.983386e-01            8.945797e-01            7.768965e-01            
10     1.184327e+01            8.051175e-01            5.452926e+03            7.460649e-01            8.990173e-01            8.171944e-01            1.219082e+01            1.979180e+00            9.871874e-01            1.302040e+02            1.023116e+00            
11     2.948709e+00            8.060435e-01            4.640226e+03            9.986822e+00            8.698071e-01            9.352140e-01            9.751843e-01            1.204599e+00            9.115073e-01            1.688499e+00            1.030805e+00            
12     2.976695e+03            2.263169e+03            9.585677e+03            2.388238e+03            1.919613e+03            2.386274e+03            2.405042e+03            2.222519e+03            2.107386e+03            1.871693e+03            2.255344e+03            
13     2.124182e+03            1.967871e+02            8.315054e+03            1.525857e+03            1.244624e+02            1.523433e+02            1.944626e+03            4.859919e+02            1.552435e+02            1.732143e+03            3.990625e+02            
14     5.634599e+00            5.436418e+00            5.755390e+00            5.628986e+00            5.448467e+00            5.415178e+00            5.576632e+00            5.771671e+00            5.480418e+00            5.557531e+00            5.608533e+00            
15     1.413984e+00            7.632014e-01            7.565836e+00            1.160685e-01            2.099646e+00            3.776470e+00            5.253094e-02            6.919848e-01            8.428538e-01            1.806054e+00            5.432898e-01            
16     7.352282e+02            7.249183e+02            2.229480e+02            3.003881e+01            1.312158e+02            5.642733e+01            4.210982e+01            7.526845e+02            7.100790e+02            1.058296e+02            7.646818e+01            
17     3.411521e+02            4.926160e+00            5.930757e+03            6.225450e+02            1.249845e+01            1.838093e+00            7.732241e+02            2.553285e+00            3.413406e+00            6.561954e+02            1.736237e+01            
18     6.141119e+02            7.256231e+00            3.558949e+03            4.377559e+02            2.687181e+00            3.685395e+02            9.968361e+02            8.588065e+00            2.499659e+00            2.622617e+02            2.760076e+01            
19     4.571131e+00            9.904289e-01            1.296630e+01            3.240711e+00            9.996334e-01            1.047708e+00            2.986352e+00            1.026383e+00            9.768249e-01            5.150205e+00            2.130036e+00            
20     5.746785e+00            5.731814e+00            5.684003e+00            5.772836e+00            5.692563e+00            5.724249e+00            5.714756e+00            5.730555e+00            5.718049e+00            5.792637e+00            5.757365e+00            
21     1.741782e+03            6.141239e+01            1.260400e+04            3.631125e+04            6.937374e+01            5.664486e+01            4.055056e+04            6.352201e+01            5.717854e+01            1.559760e+03            1.630879e+03            
22     9.864549e-01            9.146407e-01            2.755765e+01            9.577661e-01            9.629584e-01            8.853783e-01            1.787637e+00            1.021511e+00            9.028710e-01            1.146805e+00            9.499544e-01            
23     6.169560e+01            3.750776e+01            9.657378e+01            6.103527e+01            3.207622e+01            3.733709e+01            5.945012e+01            4.757564e+01            4.018412e+01            7.368232e+01            4.653113e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_idea_0.py  (error=8.990243e-02)
Task  1: variant_06_idea_0.py  (error=8.847170e-01)
Task  2: variant_03_idea_0.py  (error=7.950062e+00)
Task  3: variant_03_idea_0.py  (error=9.635242e-01)
Task  4: variant_03_idea_0.py  (error=1.719008e+00)
Task  5: variant_03_idea_0.py  (error=1.894599e+00)
Task  6: variant_06_idea_0.py  (error=1.665381e-01)
Task  7: variant_06_idea_0.py  (error=2.013815e-01)
Task  8: variant_04_idea_0.py  (error=3.477996e+02)
Task  9: variant_03_idea_0.py  (error=5.301205e-01)
Task 10: variant_03_idea_0.py  (error=7.460649e-01)
Task 11: variant_01_idea_0.py  (error=8.060435e-01)
Task 12: variant_09_idea_0.py  (error=1.871693e+03)
Task 13: variant_04_idea_0.py  (error=1.244624e+02)
Task 14: variant_05_idea_0.py  (error=5.415178e+00)
Task 15: variant_06_idea_0.py  (error=5.253094e-02)
Task 16: variant_03_idea_0.py  (error=3.003881e+01)
Task 17: variant_05_idea_0.py  (error=1.838093e+00)
Task 18: variant_08_idea_0.py  (error=2.499659e+00)
Task 19: variant_08_idea_0.py  (error=9.768249e-01)
Task 20: variant_02_idea_0.py  (error=5.684003e+00)
Task 21: variant_05_idea_0.py  (error=5.664486e+01)
Task 22: variant_05_idea_0.py  (error=8.853783e-01)
Task 23: variant_04_idea_0.py  (error=3.207622e+01)

WIN COUNTS:
  variant_03_idea_0.py: 8 wins
  variant_06_idea_0.py: 4 wins
  variant_05_idea_0.py: 4 wins
  variant_04_idea_0.py: 3 wins
  variant_08_idea_0.py: 2 wins
  variant_01_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins
  variant_02_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Reflect particles back from bounds to maintain exploration momentum."""
        reflected = pop.copy()

        # Reflect values below lower bound
        below_mask = reflected < self.lower
        if np.any(below_mask):
            excess = self.lower - reflected[below_mask]
            reflected[below_mask] = self.lower + excess

        # Reflect values above upper bound
        above_mask = reflected > self.upper
        if np.any(above_mask):
            excess = reflected[above_mask] - self.upper
            reflected[above_mask] = self.upper - excess

        # Final safety clip for any remaining edge cases
        return np.clip(reflected, self.lower, self.upper)
```

# --- From variant_02_idea_0.py (1 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Reflect particles at boundaries to preserve velocity and exploration."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        range_arr = upper - lower

        # First, clip to prevent extreme values
        pop = np.clip(pop, lower - range_arr * 0.5, upper + range_arr * 0.5)

        # Reflect particles that went below lower bound
        below_mask = pop < lower
        if np.any(below_mask):
            offset = lower - pop
            reflections = np.floor(offset / range_arr) + 1
            pop = lower + (offset - reflections * range_arr)
            pop = np.where(np.abs(offset - reflections * range_arr) < 
                           np.abs(offset - (reflections - 1) * range_arr),
                           pop, lower - (offset - (reflections - 1) * range_arr))

        # Reflect particles that went above upper bound
        above_mask = pop > upper
        if np.any(above_mask):
            offset = pop - upper
            reflections = np.floor(offset / range_arr) + 1
            pop = upper - (offset - reflections * range_arr)
            pop = np.where(np.abs(offset - reflections * range_arr) < 
                           np.abs(offset - (reflections - 1) * range_arr),
                           pop, upper + (offset - (reflections - 1) * range_arr))

        return np.clip(pop, lower, upper)
```

# --- From variant_03_idea_0.py (8 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Reflective boundary handling that preserves particle momentum."""
        if not hasattr(self, '_boundary_mode'):
            self._boundary_mode = 'reflect'

        pop = np.asarray(pop, dtype=np.float64)
        range_val = self.upper - self.lower

        if self._boundary_mode == 'wrap':
            # Periodic boundary
            return self.lower + (pop - self.lower) % range_val

        # Reflective boundary with momentum preservation
        result = pop.copy()

        # Reflect upper bounds
        upper_mask = pop > self.upper
        if np.any(upper_mask):
            overshoot = pop[upper_mask] - self.upper
            result[upper_mask] = self.upper - (overshoot % (2 * range_val))
            result[upper_mask] = np.abs(result[upper_mask] - self.upper) + self.lower

        # Reflect lower bounds  
        lower_mask = pop < self.lower
        if np.any(lower_mask):
            overshoot = self.lower - pop[lower_mask]
            result[lower_mask] = self.lower + (overshoot % (2 * range_val))
            result[lower_mask] = self.upper - np.abs(result[lower_mask] - self.lower)

        # Final safety clip
        np.clip(result, self.lower, self.upper, out=result)
        return result
```

# --- From variant_04_idea_0.py (3 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Reflective boundary handling - particles bounce off bounds instead of hard clipping."""
        range_size = self.upper - self.lower
        result = np.copy(pop)

        # Reflect above upper bound
        above_mask = result > self.upper
        if np.any(above_mask):
            excess = result[above_mask] - self.upper
            reflected = self.upper - (excess % (2 * range_size))
            reflected = np.where(reflected < self.lower, 
                                2 * self.lower - reflected, reflected)
            result[above_mask] = np.clip(reflected, self.lower, self.upper)

        # Reflect below lower bound
        below_mask = result < self.lower
        if np.any(below_mask):
            excess = self.lower - result[below_mask]
            reflected = self.lower + (excess % (2 * range_size))
            reflected = np.where(reflected > self.upper, 
                                2 * self.upper - reflected, reflected)
            result[below_mask] = np.clip(reflected, self.lower, self.upper)

        return result
```

# --- From variant_05_idea_0.py (4 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Reflective boundary handling: bounce particles back with velocity reversal."""
        # Calculate overflow past upper bound and reflect
        upper_overflow = pop - self.upper
        upper_mask = upper_overflow > 0
        pop = np.where(upper_mask, 2 * self.upper - pop, pop)

        # Calculate underflow past lower bound and reflect  
        lower_underflow = self.lower - pop
        lower_mask = lower_underflow > 0
        pop = np.where(lower_mask, 2 * self.lower - pop, pop)

        # Final safety clip for any remaining out-of-bounds values
        return np.clip(pop, self.lower, self.upper)
```

# --- From variant_06_idea_0.py (4 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Wrap population toroidally within search bounds using periodic boundary conditions."""
        range_size = self.upper - self.lower
        # Wrap each dimension using modulo arithmetic
        wrapped = (pop - self.lower) % range_size + self.lower
        return wrapped
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
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
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _clip_to_bounds_batch(self, pop):
        """Reflective boundary handling with velocity damping for better exploration."""
        result = np.clip(pop, self.lower, self.upper)
        # Identify particles that hit boundaries (need reflection)
        below = pop < self.lower
        above = pop > self.upper
        # Apply reflection: particles that went out bounce back with damped velocity
        # This preserves momentum and encourages boundary region exploration
        if np.any(below) or np.any(above):
            reflected = result.copy()
            # Reflect with 0.7 damping factor to gradually reduce boundary exploration
            reflected[below] = self.lower + 0.7 * (result[below] - self.lower + 1e-10)
            reflected[above] = self.upper - 0.7 * (self.upper - result[above] + 1e-10)
            result = reflected
        return result
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