Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_velocity_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      4.048779e-01            -inf                    4.823302e+02            2.314459e+00            -inf                    1.782217e+00            7.595249e-01            1.276189e+02            4.292865e-01            -inf                    
1      9.682844e-01            -inf                    1.376095e+00            1.038720e+00            -inf                    1.022534e+00            9.869835e-01            5.306832e-01            9.584002e-01            -inf                    
2      3.422938e+01            -inf                    1.473021e+08            6.674990e+04            -inf                    4.054101e+05            1.419534e+05            1.080145e+07            1.185925e+04            -inf                    
3      3.208152e+00            -inf                    3.179062e+03            1.238771e+01            -inf                    8.728469e+00            3.783682e+00            1.220706e+02            2.741234e+00            -inf                    
4      1.818475e+00            -inf                    2.636730e+00            1.971237e+00            -inf                    1.999423e+00            1.889146e+00            1.312207e+00            1.843725e+00            -inf                    
5      1.952838e+00            -inf                    2.658440e+00            2.076571e+00            -inf                    2.028558e+00            1.994880e+00            1.473100e+00            1.946931e+00            -inf                    
6      6.893326e-01            -inf                    8.036105e+02            2.481106e+00            -inf                    1.755651e+00            1.162844e+00            4.883400e+03            3.922769e-01            -inf                    
7      7.152444e-01            -inf                    8.638768e+02            2.763763e+00            -inf                    1.919603e+00            8.330257e-01            5.075879e+03            4.770647e-01            -inf                    
8      3.622913e+02            -inf                    1.204305e+04            2.135130e+03            -inf                    1.387232e+03            5.266682e+02            8.847360e+02            1.144059e+03            -inf                    
9      7.983386e-01            -inf                    9.312497e+02            3.166535e+00            -inf                    2.402832e+00            1.022400e+00            5.214842e+03            5.634512e-01            -inf                    
10     9.871874e-01            -inf                    1.025543e+03            4.355696e+00            -inf                    2.149899e+00            1.405892e+00            7.049893e+03            5.735906e-01            -inf                    
11     9.115073e-01            -inf                    1.028624e+03            3.694901e+00            -inf                    2.155364e+00            1.094177e+00            7.624267e+03            6.484166e-01            -inf                    
12     2.107386e+03            -inf                    1.374434e+04            5.773155e+03            -inf                    7.464747e+03            5.439685e+03            1.510999e+03            6.103692e+03            -inf                    
13     1.552435e+02            -inf                    4.322426e+03            4.618031e+02            -inf                    1.780148e+02            1.050822e+02            1.073972e+04            1.251535e+02            -inf                    
14     5.480418e+00            -inf                    6.952176e+00            5.949934e+00            -inf                    5.618841e+00            5.501060e+00            5.709038e+00            5.603677e+00            -inf                    
15     8.428538e-01            -inf                    1.218077e+03            2.531110e+00            -inf                    4.955169e+00            9.252108e-01            3.487289e+01            7.886216e-01            -inf                    
16     7.100790e+02            -inf                    5.041771e+04            4.499292e+02            -inf                    3.308346e+02            1.539809e+02            7.955055e+02            1.732222e+02            -inf                    
17     3.413406e+00            -inf                    2.140273e+03            5.616114e+02            -inf                    3.255053e+00            1.437826e+00            8.883774e+03            1.334420e+00            -inf                    
18     2.499659e+00            -inf                    4.747632e+03            3.526224e+01            -inf                    5.465200e+02            1.365800e+01            8.869793e+03            5.912733e+02            -inf                    
19     9.768249e-01            -inf                    5.276592e+00            1.411805e+00            -inf                    1.244418e+00            9.979716e-01            9.149310e+00            8.373338e-01            -inf                    
20     5.718049e+00            -inf                    2.659174e+01            6.620144e+00            -inf                    6.320190e+00            5.879828e+00            2.324290e+01            5.687944e+00            -inf                    
21     5.717854e+01            -inf                    1.327993e+04            9.673800e+01            -inf                    6.990068e+01            5.899349e+01            4.515907e+04            6.649082e+01            -inf                    
22     9.028710e-01            -inf                    1.349572e+01            1.449637e+00            -inf                    1.311212e+00            9.980542e-01            2.620282e+01            8.209246e-01            -inf                    
23     4.018412e+01            -inf                    7.787512e+01            4.473947e+01            -inf                    3.709500e+01            3.759615e+01            1.223795e+02            4.095761e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=4.048779e-01)
Task  1: variant_07_idea_0.py  (error=5.306832e-01)
Task  2: original.py  (error=3.422938e+01)
Task  3: variant_08_idea_0.py  (error=2.741234e+00)
Task  4: variant_07_idea_0.py  (error=1.312207e+00)
Task  5: variant_07_idea_0.py  (error=1.473100e+00)
Task  6: variant_08_idea_0.py  (error=3.922769e-01)
Task  7: variant_08_idea_0.py  (error=4.770647e-01)
Task  8: original.py  (error=3.622913e+02)
Task  9: variant_08_idea_0.py  (error=5.634512e-01)
Task 10: variant_08_idea_0.py  (error=5.735906e-01)
Task 11: variant_08_idea_0.py  (error=6.484166e-01)
Task 12: variant_07_idea_0.py  (error=1.510999e+03)
Task 13: variant_06_idea_0.py  (error=1.050822e+02)
Task 14: original.py  (error=5.480418e+00)
Task 15: variant_08_idea_0.py  (error=7.886216e-01)
Task 16: variant_06_idea_0.py  (error=1.539809e+02)
Task 17: variant_08_idea_0.py  (error=1.334420e+00)
Task 18: original.py  (error=2.499659e+00)
Task 19: variant_08_idea_0.py  (error=8.373338e-01)
Task 20: variant_08_idea_0.py  (error=5.687944e+00)
Task 21: original.py  (error=5.717854e+01)
Task 22: variant_08_idea_0.py  (error=8.209246e-01)
Task 23: variant_05_idea_0.py  (error=3.709500e+01)

WIN COUNTS:
  variant_08_idea_0.py: 11 wins
  original.py: 6 wins
  variant_07_idea_0.py: 4 wins
  variant_06_idea_0.py: 2 wins
  variant_05_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_05_idea_0.py (1 wins) ---
```python
def _update_velocity_batch(self, inertia):
        """Update velocity using orthogonal learning to escape local optima."""
        pop_size, dim = self.pop_size, self.dim

        # Standard PSO components
        r1 = np.random.uniform(0, 1, size=(pop_size, dim))
        r2 = np.random.uniform(0, 1, size=(pop_size, dim))
        r3 = np.random.uniform(0, 1, size=(pop_size, dim))

        cognitive = self.c1 * r1 * (self.personal_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)

        base_velocity = inertia * self.velocity + cognitive + social + neighborhood

        # Orthogonal learning: construct candidate solutions via dimension grouping
        n_groups = max(2, dim // 10)
        group_size = dim // n_groups

        # Select particles with above-median fitness for orthogonal perturbation
        median_fit = np.median(self.personal_best_fit)
        elite_mask = self.personal_best_fit <= median_fit
        n_elite = np.sum(elite_mask)

        if n_elite > 0 and dim >= 4:
            # Create orthogonal test matrix using Hadamard-like grouping
            ort_velocity = np.zeros_like(base_velocity)

            for i in np.where(elite_mask)[0]:
                # Generate 2 candidate perturbations per group
                for g in range(n_groups):
                    start = g * group_size
                    end = start + group_size if g < n_groups - 1 else dim
                    group_dims = end - start

                    if group_dims < 2:
                        continue

                    # Two orthogonal perturbation directions for this group
                    p1 = np.random.uniform(-1, 1, group_dims)
                    p2 = np.random.uniform(-1, 1, group_dims)
                    # Orthogonalize
                    p2 = p2 - np.dot(p1, p2) / (np.dot(p1, p1) + 1e-10) * p1

                    # Test both and use better direction
                    delta1 = self.tunnel_scale * (self.upper - self.lower) * p1
                    delta2 = self.tunnel_scale * (self.upper - self.lower) * p2

                    cand1 = self.population[i].copy()
                    cand2 = self.population[i].copy()
                    cand1[start:end] += delta1
                    cand2[start:end] += delta2

                    # Quick fitness comparison (clip to bounds)
                    cand1 = np.clip(cand1, self.lower, self.upper)
                    cand2 = np.clip(cand2, self.lower, self.upper)

                    # Use the better direction for velocity contribution
                    dist1 = np.linalg.norm(cand1 - self.global_best)
                    dist2 = np.linalg.norm(cand2 - self.global_best)

                    if dist1 < dist2:
                        ort_velocity[i, start:end] = delta1 * 0.5
                    else:
                        ort_velocity[i, start:end] = delta2 * 0.5

            # Blend base velocity with orthogonal learning
            blend = np.where(elite_mask, 0.3, 0.0)
            blend = blend.reshape(-1, 1)
            self.velocity = (1 - blend) * base_velocity + blend * (base_velocity + ort_velocity)
        else:
            self.velocity = base_velocity

        # Velocity clamping
        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
```

# --- From variant_06_idea_0.py (2 wins) ---
```python
def _update_velocity_batch(self, inertia):
        """Velocity update with orthogonal learning and convergence-driven perturbation."""
        cognitive = self.personal_best - self.population
        social = self.global_best - self.population
        neighborhood = self.neighborhood_best - self.population

        guides = np.stack([cognitive, social, neighborhood], axis=0)

        weights = np.array([self.c1, self.c2, self.c3]).reshape(3, 1, 1)
        weighted_sum = np.sum(guides * weights, axis=0)

        norms = np.linalg.norm(guides, axis=2, keepdims=True) + 1e-10
        normalized = guides / norms
        ortho_component = np.sum(normalized, axis=0) - weighted_sum / (np.linalg.norm(weighted_sum, axis=1, keepdims=True) + 1e-10)

        r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))

        self.velocity = inertia * self.velocity + r1 * weighted_sum + r2 * ortho_component

        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
```

# --- From variant_07_idea_0.py (4 wins) ---
```python
def _update_velocity_batch(self, inertia):
        """Update velocity using success-history adaptive components."""
        # Track position deltas for each component's effectiveness
        if not hasattr(self, '_prev_pos_for_success'):
            self._prev_pos_for_success = self.population.copy()
            self._success_cognitive = np.ones(self.pop_size)
            self._success_social = np.ones(self.pop_size)
            self._success_neighborhood = np.ones(self.pop_size)

        # Compute how much each component actually moved the particle
        delta_pos = self.population - self._prev_pos_for_success
        self._prev_pos_for_success = self.population.copy()

        # Success rate = proportion of dimensions where position changed meaningfully
        pos_mag = np.linalg.norm(delta_pos, axis=1, keepdims=True) + 1e-10
        delta_normalized = np.abs(delta_pos) / (pos_mag + 1e-10)

        # Update success rates with exponential moving average
        alpha = 0.3
        self._success_cognitive = (1 - alpha) * self._success_cognitive + alpha * np.mean(delta_normalized, axis=1)
        self._success_social = (1 - alpha) * self._success_social + alpha * np.mean(delta_normalized, axis=1)
        self._success_neighborhood = (1 - alpha) * self._success_neighborhood + alpha * np.mean(delta_normalized, axis=1)

        # Normalize success rates and compute adaptive weights
        total_success = self._success_cognitive + self._success_social + self._success_neighborhood + 1e-10
        w_c = self._success_cognitive / total_success
        w_s = self._success_social / total_success
        w_n = self._success_neighborhood / total_success

        # Apply minimum exploration weight to prevent premature convergence
        min_weight = 0.05
        w_c = np.clip(w_c, min_weight, None)
        w_s = np.clip(w_s, min_weight, None)
        w_n = np.clip(w_n, min_weight, None)

        # Recompute after clipping
        total = w_c + w_s + w_n
        w_c, w_s, w_n = w_c / total, w_s / total, w_n / total

        # Compute velocity components with success-adaptive coefficients
        cognitive = self.c1 * w_c[:, np.newaxis] * (self.personal_best - self.population)
        social = self.c2 * w_s[:, np.newaxis] * (self.global_best - self.population)
        neighborhood = self.c3 * w_n[:, np.newaxis] * (self.neighborhood_best - self.population)

        self.velocity = inertia * self.velocity + cognitive + social + neighborhood

        # Velocity clamping
        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
```

# --- From variant_08_idea_0.py (11 wins) ---
```python
def _update_velocity_batch(self, inertia):
        """Dual-mode velocity update: stagnation-driven exploration vs exploitation."""
        # Compute population centroid for exploration direction
        centroid = np.mean(self.population, axis=0)

        # Determine stagnant vs improving particles
        stagnant_mask = self.stagnation_counter >= self.stagnation_limit // 2

        # Exploration mode: move toward OPPOSITE of centroid (implicit niching)
        exploration_dir = centroid - self.population  # Direction away from centroid
        exploration_mag = np.linalg.norm(exploration_dir, axis=1, keepdims=True)
        safe_mag = np.where(exploration_mag < 1e-10, 1.0, exploration_mag)
        exploration_dir_normalized = exploration_dir / safe_mag

        # Scale exploration by stagnation level and search space
        stagnation_factor = np.minimum(self.stagnation_counter[:, np.newaxis] / max(1, self.stagnation_limit), 3.0)
        exploration_scale = stagnation_factor * (self.upper - self.lower) * 0.1
        exploration_velocity = exploration_dir_normalized * exploration_scale

        # Exploitation mode: standard PSO velocity components
        r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))

        cognitive = self.c1 * r1 * (self.personal_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)

        # Dual-mode selection: stagnant = exploration, improving = exploitation
        exploitation_velocity = inertia * self.velocity + cognitive + social + neighborhood
        self.velocity = np.where(stagnant_mask[:, np.newaxis], exploration_velocity, exploitation_velocity)

        # Velocity clamping
        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
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