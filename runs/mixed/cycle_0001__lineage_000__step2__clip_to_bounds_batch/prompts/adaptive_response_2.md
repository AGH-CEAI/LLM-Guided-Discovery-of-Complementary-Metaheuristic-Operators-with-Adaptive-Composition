```python
import numpy as np


class QuantumRingSwarmOptimizer:
    """
    A PSO-based optimizer with dynamic ring topology and quantum tunneling mutation.
    Features adaptive boundary handling via Thompson Sampling with sliding window credit assignment.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = kwargs.get('pop_size', 6 * dim)
        self.lower = kwargs.get('lower', -100.0)
        self.upper = kwargs.get('upper', 100.0)
        
        self.k_neighbors = kwargs.get('k_neighbors', 3)
        self.w_max = kwargs.get('w_max', 0.9)
        self.w_min = kwargs.get('w_min', 0.4)
        self.c1 = kwargs.get('c1', 1.5)
        self.c2 = kwargs.get('c2', 1.5)
        self.c3 = kwargs.get('c3', 2.0)
        
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
        self.stagnation_counter = None
        self.fitness_history = []
        self.step_size = 1.0
        
        # Adaptive operator selection state
        self._num_strategies = 8
        self._strategy_alpha = np.ones(self._num_strategies)
        self._strategy_beta = np.ones(self._num_strategies)
        self._strategy_rewards = [[] for _ in range(self._num_strategies)]
        self._strategy_selected_count = np.zeros(self._num_strategies, dtype=np.int64)
        self._last_best_fit = None
        self._generation = 0
        self._epsilon = 0.3
        self._sliding_window = 15
        self._current_strategy = None
        
        # Initialize priors based on benchmark win counts
        benchmark_priors = {
            0: 8, 1: 4, 2: 4, 3: 3, 4: 2, 5: 1, 6: 1, 7: 1
        }
        for idx, wins in benchmark_priors.items():
            self._strategy_alpha[idx] = 1.0 + wins * 2.0
            self._strategy_beta[idx] = 1.0 + (24 - wins) * 0.5
    
    # === All 8 boundary handling strategies ===
    
    def _clip_strategy_0(self, pop):
        """Reflect particles back from bounds to maintain exploration momentum."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        reflected = pop.copy()
        
        below_mask = reflected < lower
        if np.any(below_mask):
            excess = lower - reflected[below_mask]
            reflected[below_mask] = lower + excess
        
        above_mask = reflected > upper
        if np.any(above_mask):
            excess = reflected[above_mask] - upper
            reflected[above_mask] = upper - excess
        
        return np.clip(reflected, lower, upper)
    
    def _clip_strategy_1(self, pop):
        """Reflective boundary handling with momentum preservation."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        range_arr = upper - lower
        
        pop = np.clip(pop, lower - range_arr * 0.5, upper + range_arr * 0.5)
        
        below_mask = pop < lower
        if np.any(below_mask):
            offset = lower - pop[below_mask]
            reflections = np.floor(offset / range_arr) + 1
            pop[below_mask] = lower + (offset - reflections * range_arr)
            pop[below_mask] = np.where(
                np.abs(offset - reflections * range_arr) < np.abs(offset - (reflections - 1) * range_arr),
                pop[below_mask], lower - (offset - (reflections - 1) * range_arr)
            )
        
        above_mask = pop > upper
        if np.any(above_mask):
            offset = pop[above_mask] - upper
            reflections = np.floor(offset / range_arr) + 1
            pop[above_mask] = upper - (offset - reflections * range_arr)
            pop[above_mask] = np.where(
                np.abs(offset - reflections * range_arr) < np.abs(offset - (reflections - 1) * range_arr),
                pop[above_mask], upper + (offset - (reflections - 1) * range_arr)
            )
        
        return np.clip(pop, lower, upper)
    
    def _clip_strategy_2(self, pop):
        """Reflective boundary handling that preserves particle momentum."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        range_val = upper - lower
        
        result = pop.copy()
        
        upper_mask = pop > upper
        if np.any(upper_mask):
            overshoot = pop[upper_mask] - upper
            result[upper_mask] = upper - (overshoot % (2 * range_val))
            result[upper_mask] = np.abs(result[upper_mask] - upper) + lower
        
        lower_mask = pop < lower
        if np.any(lower_mask):
            overshoot = lower - pop[lower_mask]
            result[lower_mask] = lower + (overshoot % (2 * range_val))
            result[lower_mask] = upper - np.abs(result[lower_mask] - lower)
        
        np.clip(result, lower, upper, out=result)
        return result
    
    def _clip_strategy_3(self, pop):
        """Reflective boundary handling - particles bounce off bounds."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        range_size = upper - lower
        result = pop.copy()
        
        above_mask = result > upper
        if np.any(above_mask):
            excess = result[above_mask] - upper
            reflected = upper - (excess % (2 * range_size))
            reflected = np.where(reflected < lower, 2 * lower - reflected, reflected)
            result[above_mask] = np.clip(reflected, lower, upper)
        
        below_mask = result < lower
        if np.any(below_mask):
            excess = lower - result[below_mask]
            reflected = lower + (excess % (2 * range_size))
            reflected = np.where(reflected > upper, 2 * upper - reflected, reflected)
            result[below_mask] = np.clip(reflected, lower, upper)
        
        return result
    
    def _clip_strategy_4(self, pop):
        """Reflective boundary handling: bounce particles back with velocity reversal."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        
        upper_overflow = pop - upper
        upper_mask = upper_overflow > 0
        pop = np.where(upper_mask, 2 * upper - pop, pop)
        
        lower_underflow = lower - pop
        lower_mask = lower_underflow > 0
        pop = np.where(lower_mask, 2 * lower - pop, pop)
        
        return np.clip(pop, lower, upper)
    
    def _clip_strategy_5(self, pop):
        """Wrap population toroidally within search bounds using periodic boundary conditions."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        range_size = upper - lower
        wrapped = (pop - lower) % range_size + lower
        return wrapped
    
    def _clip_strategy_6(self, pop):
        """Reflective boundary handling - invert velocity at bounds to preserve momentum."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        reflected = pop.copy()
        
        below_lower = reflected < lower
        reflected[below_lower] = 2.0 * lower - reflected[below_lower]
        
        above_upper = reflected > upper
        reflected[above_upper] = 2.0 * upper - reflected[above_upper]
        
        return np.clip(reflected, lower, upper)
    
    def _clip_strategy_7(self, pop):
        """Reflective boundary handling with velocity damping for better exploration."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        result = np.clip(pop, lower, upper)
        
        below = pop < lower
        above = pop > upper
        if np.any(below) or np.any(above):
            reflected = result.copy()
            reflected[below] = lower + 0.7 * (result[below] - lower + 1e-10)
            reflected[above] = upper - 0.7 * (upper - result[above] + 1e-10)
            result = reflected
        return result
    
    def _get_all_strategies(self):
        """Return list of all clipping strategies."""
        return [
            self._clip_strategy_0,
            self._clip_strategy_1,
            self._clip_strategy_2,
            self._clip_strategy_3,
            self._clip_strategy_4,
            self._clip_strategy_5,
            self._clip_strategy_6,
            self._clip_strategy_7,
        ]
    
    def _select_strategy_thompson(self):
        """Select strategy using Thompson Sampling with epsilon-greedy exploration."""
        strategies = self._get_all_strategies()
        
        if np.random.random() < self._epsilon:
            selected = np.random.randint(self._num_strategies)
        else:
            samples = np.array([
                np.random.beta(max(self._strategy_alpha[i], 0.1), max(self._strategy_beta[i], 0.1))
                for i in range(self._num_strategies)
            ])
            selected = int(np.argmax(samples))
        
        self._current_strategy = selected
        self._strategy_selected_count[selected] += 1
        return strategies[selected]
    
    def _update_strategy_reward(self, reward):
        """Update strategy reward using sliding window credit assignment."""
        if self._current_strategy is None:
            return
        
        strategy_idx = self._current_strategy
        self._strategy_rewards[strategy_idx].append(float(reward))
        
        if len(self._strategy_rewards[strategy_idx]) > self._sliding_window:
            self._strategy_rewards[strategy_idx].pop(0)
        
        rewards = self._strategy_rewards[strategy_idx]
        if len(rewards) >= 3:
            recent_avg = float(np.mean(rewards[-5:]))
            if recent_avg > 0.01:
                self._strategy_alpha[strategy_idx] += 0.3
            elif recent_avg < -0.01:
                self._strategy_beta[strategy_idx] += 0.3
    
    def _clip_to_bounds_batch(self, pop):
        """Adaptive boundary handling - delegates to selected strategy."""
        strategy = self._select_strategy_thompson()
        return strategy(pop)
    
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
        self.stagnation_counter = np.zeros(self.pop_size, dtype=np.int32)
        self.fitness_history = []
        self.step_size = 1.0
        self._last_best_fit = None
        self._generation = 0
    
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
            
            # Compute reward for adaptive strategy selection
            if self._last_best_fit is not None and np.isfinite(self.global_best_fit) and np.isfinite(self._last_best_fit):
                reward = float(self._last_best_fit) - float(self.global_best_fit)
                self._update_strategy_reward(reward)
            self._last_best_fit = self.global_best_fit
            
            self._compute_ring_topology()
            self._quantum_tunnel_batch()
            
            inertia = self._compute_adaptive_inertia()
            self._update_velocity_batch(inertia)
            self._update_position_batch()
            self.population = self._clip_to_bounds_batch(self.population)
            
            self._reinitialize_diverse_particles()
            self._adapt_step_size()
            
            generation += 1
            self._generation = generation
            
            # Decay exploration rate
            if generation % 50 == 0:
                self._epsilon = max(0.05, self._epsilon * 0.98)
            
            if generation % 100 == 0:
                div = self._compute_diversity()
                
        return self.global_best_fit, self.global_best
```