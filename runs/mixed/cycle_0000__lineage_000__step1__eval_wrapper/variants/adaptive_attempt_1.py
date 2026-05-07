import numpy as np


class QuantumRingSwarmOptimizer:
    """
    A PSO-based optimizer with dynamic ring topology and quantum tunneling mutation.
    
    Key components:
    - Ring lattice topology with k-nearest neighborhood
    - Quantum tunneling mutation triggered by stagnation
    - Adaptive inertia weight based on progress ratio
    - Diversity-based reinitialization of stagnant particles
    - Adaptive _eval_wrapper selection using Thompson Sampling Multi-Armed Bandit
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
        
        # Adaptive operator selection using Thompson Sampling
        self.n_operators = 3
        self.operator_rewards = np.zeros(self.n_operators, dtype=np.float64)
        self.operator_counts = np.zeros(self.n_operators, dtype=np.int32)
        
        self._operators = [
            self._eval_wrapper_operator_0,
            self._eval_wrapper_operator_1,
            self._eval_wrapper_operator_2
        ]
        
        self.last_operator = -1
        
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
        
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling with beta distributions."""
        if np.any(self.operator_counts == 0):
            return int(np.argmin(self.operator_counts))
        
        samples = np.array([
            float(np.random.beta(
                max(float(self.operator_rewards[i]), 0.01) * max(int(self.operator_counts[i]), 1) + 1.0,
                max(int(self.operator_counts[i]), 1)
            ))
            for i in range(self.n_operators)
        ], dtype=np.float64)
        
        return int(np.argmax(samples))
    
    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator using exponential weighted average."""
        operator_idx = int(operator_idx)
        reward = float(reward)
        alpha = 0.3
        self.operator_rewards[operator_idx] = alpha * reward + (1.0 - alpha) * float(self.operator_rewards[operator_idx])
        self.operator_counts[operator_idx] += 1
    
    def _eval_wrapper_operator_0(self, x):
        """Original wrapper - direct evaluation."""
        x = np.atleast_2d(x)
        x_clipped = np.clip(x, self.lower, self.upper)
        result = self._user_func(x_clipped)
        return result
    
    def _eval_wrapper_operator_1(self, x):
        """Robust wrapper with bounds enforcement and NaN/Inf penalty."""
        x = np.atleast_2d(x)
        x_clipped = np.clip(x, self.lower, self.upper)
        result = self._user_func(x_clipped)
        result = np.where(np.isfinite(result), result, 1e30)
        return result
    
    def _eval_wrapper_operator_2(self, x):
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
    
    def _eval_wrapper(self, x):
        """Adaptive wrapper that selects the best operator using Thompson Sampling."""
        if int(self.operator_counts.sum()) < 5:
            selected = 0
        else:
            selected = self._select_operator_thompson()
        
        self.last_operator = selected
        return self._operators[selected](x)
    
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
    
    def _compute_reward(self, prev_best, curr_best):
        """Compute reward for operator selection based on improvement."""
        if curr_best < prev_best:
            reward = float(prev_best - curr_best)
        else:
            reward = -0.01
        return float(np.clip(reward, -1e30, 1e30))
    
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
        last_best_fitness = np.inf
        
        while not stopping_condition():
            fitness = self._evaluate_batch(self.population, func)
            
            if len(fitness) < len(self.population):
                fitness = np.pad(fitness, (0, len(self.population) - len(fitness)),
                               constant_values=np.inf)
                break
            
            prev_best = float(self.global_best_fit) if np.isfinite(self.global_best_fit) else np.inf
            
            self._update_personal_best_batch(fitness)
            self._update_global_best_batch()
            self.fitness_history.append(self.global_best_fit)
            
            curr_best = float(self.global_best_fit) if np.isfinite(self.global_best_fit) else np.inf
            
            if self.last_operator >= 0:
                reward = self._compute_reward(prev_best, curr_best)
                self._update_operator_rewards(self.last_operator, reward)
                self.last_operator = -1
            
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
