import numpy as np
import collections


class QuantumRingSwarmOptimizer:
    """
    A PSO-based optimizer with dynamic ring topology and quantum tunneling mutation.
    Now includes ADAPTIVE clipping strategy selection using Multi-Armed Bandit with UCB.
    
    Key components:
    - Ring lattice topology with k-nearest neighborhood
    - Quantum tunneling mutation triggered by stagnation
    - Adaptive inertia weight based on progress ratio
    - Diversity-based reinitialization of stagnant particles
    - Adaptive clipping strategy selection (8 variants, UCB-based selection)
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
        
        # Adaptive clipping mechanism
        self._adaptive_initialized = False
        self._n_strategies = 8
        self._strategy_rewards = [collections.deque(maxlen=15) for _ in range(self._n_strategies)]
        self._strategy_counts = np.zeros(self._n_strategies, dtype=np.float64)
        self._ucb_c = 2.5
        self._current_strategy = 0
        self._generation = 0
        
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
        """Adaptive clipping: select best strategy using UCB Multi-Armed Bandit."""
        if not self._adaptive_initialized:
            self._adaptive_initialized = True
        
        # Select strategy using UCB
        strategy = self._select_strategy_ucb()
        self._current_strategy = strategy
        
        # Apply selected clipping strategy
        result = self._clip_strategies[strategy](pop)
        
        # Update reward based on global best fitness improvement
        self._update_strategy_reward(strategy, self.global_best_fit)
        
        self._generation += 1
        
        return result
    
    def _clip_strategy_00(self, pop):
        """Original: Simple hard clipping to bounds."""
        return np.clip(pop, self.lower, self.upper)
    
    def _clip_strategy_01(self, pop):
        """Reflect particles back from bounds to maintain exploration momentum."""
        reflected = pop.copy()
        below_mask = reflected < self.lower
        if np.any(below_mask):
            excess = self.lower - reflected[below_mask]
            reflected[below_mask] = self.lower + excess
        above_mask = reflected > self.upper
        if np.any(above_mask):
            excess = reflected[above_mask] - self.upper
            reflected[above_mask] = self.upper - excess
        return np.clip(reflected, self.lower, self.upper)
    
    def _clip_strategy_02(self, pop):
        """Reflect particles at boundaries with periodic extension."""
        pop = np.asarray(pop, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        range_arr = upper - lower
        pop = np.clip(pop, lower - range_arr * 0.5, upper + range_arr * 0.5)
        below_mask = pop < lower
        if np.any(below_mask):
            offset = lower - pop
            reflections = np.floor(offset / range_arr) + 1
            pop = lower + (offset - reflections * range_arr)
            pop = np.where(np.abs(offset - reflections * range_arr) < 
                           np.abs(offset - (reflections - 1) * range_arr),
                           pop, lower - (offset - (reflections - 1) * range_arr))
        above_mask = pop > upper
        if np.any(above_mask):
            offset = pop - upper
            reflections = np.floor(offset / range_arr) + 1
            pop = upper - (offset - reflections * range_arr)
            pop = np.where(np.abs(offset - reflections * range_arr) < 
                           np.abs(offset - (reflections - 1) * range_arr),
                           pop, upper + (offset - (reflections - 1) * range_arr))
        return np.clip(pop, lower, upper)
    
    def _clip_strategy_03(self, pop):
        """Reflective boundary handling that preserves particle momentum (BEST OVERALL)."""
        pop = np.asarray(pop, dtype=np.float64)
        range_val = self.upper - self.lower
        result = pop.copy()
        upper_mask = pop > self.upper
        if np.any(upper_mask):
            overshoot = pop[upper_mask] - self.upper
            result[upper_mask] = self.upper - (overshoot % (2 * range_val))
            result[upper_mask] = np.abs(result[upper_mask] - self.upper) + self.lower
        lower_mask = pop < self.lower
        if np.any(lower_mask):
            overshoot = self.lower - pop[lower_mask]
            result[lower_mask] = self.lower + (overshoot % (2 * range_val))
            result[lower_mask] = self.upper - np.abs(result[lower_mask] - self.lower)
        np.clip(result, self.lower, self.upper, out=result)
        return result
    
    def _clip_strategy_04(self, pop):
        """Reflective boundary handling - particles bounce off bounds."""
        range_size = self.upper - self.lower
        result = np.copy(pop)
        above_mask = result > self.upper
        if np.any(above_mask):
            excess = result[above_mask] - self.upper
            reflected = self.upper - (excess % (2 * range_size))
            reflected = np.where(reflected < self.lower, 
                                2 * self.lower - reflected, reflected)
            result[above_mask] = np.clip(reflected, self.lower, self.upper)
        below_mask = result < self.lower
        if np.any(below_mask):
            excess = self.lower - result[below_mask]
            reflected = self.lower + (excess % (2 * range_size))
            reflected = np.where(reflected > self.upper, 
                                2 * self.upper - reflected, reflected)
            result[below_mask] = np.clip(reflected, self.lower, self.upper)
        return result
    
    def _clip_strategy_05(self, pop):
        """Reflective boundary handling: bounce particles back with velocity reversal."""
        pop = np.asarray(pop, dtype=np.float64)
        upper_overflow = pop - self.upper
        upper_mask = upper_overflow > 0
        pop = np.where(upper_mask, 2 * self.upper - pop, pop)
        lower_underflow = self.lower - pop
        lower_mask = lower_underflow > 0
        pop = np.where(lower_mask, 2 * self.lower - pop, pop)
        return np.clip(pop, self.lower, self.upper)
    
    def _clip_strategy_06(self, pop):
        """Wrap population toroidally within search bounds using periodic boundary."""
        range_size = self.upper - self.lower
        wrapped = (pop - self.lower) % range_size + self.lower
        return wrapped
    
    def _clip_strategy_07(self, pop):
        """Reflective boundary handling - invert at bounds with damped velocity."""
        result = np.clip(pop, self.lower, self.upper)
        below = pop < self.lower
        above = pop > self.upper
        if np.any(below) or np.any(above):
            reflected = result.copy()
            reflected[below] = self.lower + 0.7 * (result[below] - self.lower + 1e-10)
            reflected[above] = self.upper - 0.7 * (self.upper - result[above] + 1e-10)
            result = reflected
        return result
    
    @property
    def _clip_strategies(self):
        """List of all clipping strategies."""
        return [
            self._clip_strategy_00,
            self._clip_strategy_01,
            self._clip_strategy_02,
            self._clip_strategy_03,
            self._clip_strategy_04,
            self._clip_strategy_05,
            self._clip_strategy_06,
            self._clip_strategy_07,
        ]
    
    def _select_strategy_ucb(self):
        """Select strategy using UCB1 algorithm with sliding window rewards."""
        # Exploration phase: try all strategies at least 3 times
        if self._generation < self._n_strategies * 3:
            return int(self._generation % self._n_strategies)
        
        # Check for untried strategies
        zero_count_mask = self._strategy_counts < 1.0
        if np.any(zero_count_mask):
            candidates = np.where(zero_count_mask)[0]
            return int(np.random.choice(candidates))
        
        # Compute UCB values
        ucb_values = np.zeros(self._n_strategies)
        total_counts = np.sum(self._strategy_counts)
        
        for i in range(self._n_strategies):
            window = self._strategy_rewards[i]
            if len(window) == 0:
                avg_reward = 0.0
            else:
                avg_reward = float(np.mean(window))
            
            exploration = self._ucb_c * np.sqrt(np.log(total_counts + 1) / (self._strategy_counts[i] + 1e-10))
            ucb_values[i] = avg_reward + exploration
        
        best_idx = int(np.argmax(ucb_values))
        
        # Penalize strategies that consistently underperform
        current_best = self.global_best_fit
        if np.isfinite(current_best):
            for i in range(self._n_strategies):
                window = self._strategy_rewards[i]
                if len(window) >= 5:
                    avg_reward = float(np.mean(window))
                    if avg_reward < -0.01:
                        for _ in range(3):
                            if len(window) > 0:
                                window.pop()
        
        return best_idx
    
    def _update_strategy_reward(self, strategy_idx, current_best_fit):
        """Update reward for a strategy based on global best fitness."""
        strategy_idx = int(strategy_idx)
        
        if current_best_fit is None or not np.isfinite(current_best_fit) or np.isinf(current_best_fit):
            reward = float(0.0)
        else:
            current_best_fit = float(current_best_fit)
            reward = 1.0 / (current_best_fit + 1e-10)
            reward = float(np.clip(reward, -10.0, 10.0))
        
        self._strategy_rewards[strategy_idx].append(reward)
        self._strategy_counts[strategy_idx] += 1.0
        
        # Keep window bounded
        while len(self._strategy_rewards[strategy_idx]) > 15:
            self._strategy_rewards[strategy_idx].popleft()
    
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
            self.global_best_fit = float(self.personal_best_fit[best_idx])
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
