Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_eval_wrapper` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.603348e-01            4.253178e-01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    5.868509e-01            
1      9.710430e-01            9.732682e-01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    9.734031e-01            
2      1.824538e+02            1.692808e+04            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    2.773945e+04            
3      2.850471e+00            2.978292e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    3.144114e+00            
4      1.844645e+00            1.831448e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.844574e+00            
5      1.949502e+00            1.946215e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.956315e+00            
6      8.915770e-01            6.827187e-01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    7.059156e-01            
7      6.682472e-01            7.419620e-01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    6.640566e-01            
8      4.419827e+02            5.985528e+02            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    7.032669e+02            
9      9.744912e-01            9.532716e-01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.016964e+00            
10     1.761190e+02            3.141418e+01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.576539e+02            
11     1.460576e+00            6.748600e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.983747e+00            
12     3.226117e+03            2.627676e+03            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    3.678115e+03            
13     1.080202e+03            1.715788e+03            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.660294e+03            
14     5.730734e+00            5.775388e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    5.753100e+00            
15     4.992817e-01            3.806803e-01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.505270e+00            
16     1.826325e+02            1.856598e+02            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    2.700399e+02            
17     5.437868e+02            8.042782e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.121979e+01            
18     2.142372e+02            1.275477e+01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    5.936732e+02            
19     3.355849e+00            3.608714e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    4.666813e+00            
20     5.785432e+00            5.707779e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    5.838891e+00            
21     1.704798e+03            1.720021e+03            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.360633e+03            
22     9.436343e-01            1.107518e+00            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    1.306174e+00            
23     6.771831e+01            6.538856e+01            -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    -inf                    7.256662e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_idea_0.py  (error=4.253178e-01)
Task  1: original.py  (error=9.710430e-01)
Task  2: original.py  (error=1.824538e+02)
Task  3: original.py  (error=2.850471e+00)
Task  4: variant_01_idea_0.py  (error=1.831448e+00)
Task  5: variant_01_idea_0.py  (error=1.946215e+00)
Task  6: variant_01_idea_0.py  (error=6.827187e-01)
Task  7: variant_10_idea_0.py  (error=6.640566e-01)
Task  8: original.py  (error=4.419827e+02)
Task  9: variant_01_idea_0.py  (error=9.532716e-01)
Task 10: variant_01_idea_0.py  (error=3.141418e+01)
Task 11: original.py  (error=1.460576e+00)
Task 12: variant_01_idea_0.py  (error=2.627676e+03)
Task 13: original.py  (error=1.080202e+03)
Task 14: original.py  (error=5.730734e+00)
Task 15: variant_01_idea_0.py  (error=3.806803e-01)
Task 16: original.py  (error=1.826325e+02)
Task 17: variant_01_idea_0.py  (error=8.042782e+00)
Task 18: variant_01_idea_0.py  (error=1.275477e+01)
Task 19: original.py  (error=3.355849e+00)
Task 20: variant_01_idea_0.py  (error=5.707779e+00)
Task 21: variant_10_idea_0.py  (error=1.360633e+03)
Task 22: original.py  (error=9.436343e-01)
Task 23: variant_01_idea_0.py  (error=6.538856e+01)

WIN COUNTS:
  variant_01_idea_0.py: 12 wins
  original.py: 10 wins
  variant_10_idea_0.py: 2 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (12 wins) ---
```python
def _eval_wrapper(self, x):
        """Robust wrapper with bounds enforcement and NaN/Inf penalty."""
        x = np.atleast_2d(x)
        x_clipped = np.clip(x, self.lower, self.upper)
        result = self._user_func(x_clipped)
        # Penalty for non-finite values (NaN/Inf) to avoid breaking optimization
        result = np.where(np.isfinite(result), result, 1e30)
        return result
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _eval_wrapper(self, x):
        """Wrapper with gradient approximation and random restart to escape local optima."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        eps = 1e-5
        f0 = self._user_func(x)

        if np.isfinite(f0) and f0 < 1e6:
            return f0

        grad = np.zeros_like(x)
        for d in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[:, d] += eps
            f_plus = self._user_func(x_plus)
            grad[:, d] = (f_plus - f0) / (eps + 1e-12)

        grad_norm = np.linalg.norm(grad) + 1e-12
        if grad_norm > 1e-6:
            step_size = min(1.0, grad_norm * 0.1)
            x_new = x - grad / grad_norm * step_size
            x_new = np.clip(x_new, self.lower, self.upper)
            f_new = self._user_func(x_new)
            if np.isfinite(f_new) and f_new < f0:
                return f_new

        if np.random.random() < 0.3:
            x_random = np.random.uniform(self.lower, self.upper, size=x.shape)
            f_random = self._user_func(x_random)
            if np.isfinite(f_random) and f_random < f0:
                return f_random

        x_clipped = np.clip(x, self.lower, self.upper)
        return self._user_func(x_clipped)
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
        """Wrapper for single evaluation during tunnel."""
        return self._user_func(x)
    
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