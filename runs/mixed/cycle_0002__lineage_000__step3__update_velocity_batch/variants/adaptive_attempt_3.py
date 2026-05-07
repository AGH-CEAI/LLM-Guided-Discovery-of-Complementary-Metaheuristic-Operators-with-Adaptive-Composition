import numpy as np


class QuantumRingSwarmOptimizer:
    """
    An Adaptive PSO-based optimizer with dynamic ring topology, quantum tunneling mutation,
    and Thompson Sampling-based adaptive operator selection for velocity updates.
    
    Key components:
    - Ring lattice topology with k-nearest neighborhood
    - Quantum tunneling mutation triggered by stagnation
    - Adaptive inertia weight based on progress ratio
    - Diversity-based reinitialization of stagnant particles
    - Thompson Sampling for adaptive selection of velocity update strategies
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
        self.stagnation_counter = np.zeros(self.pop_size, dtype=np.int32)
        self.fitness_history = []
        self.step_size = 1.0
        
        # Adaptive operator selection via Thompson Sampling
        self.num_operators = 5
        self.operator_names = ['original', 'variant_05', 'variant_06', 'variant_07', 'variant_08']
        self.operator_alpha = np.ones(self.num_operators, dtype=np.float64)
        self.operator_beta = np.ones(self.num_operators, dtype=np.float64)
        self.reward_history = [[] for _ in range(self.num_operators)]
        self.reward_window_size = 15
        self.current_operator = 0
        self.selection_counts = np.zeros(self.num_operators, dtype=np.int32)
        self._prev_fitness_for_op = None
        
    def _initialize_population(self):
        """Initialize population uniformly in search space with zero velocity."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.pop_size, self.dim)
        )
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fit = np.full(self.pop_size, np.inf, dtype=np.float64)
        self.neighborhood_best = np.zeros((self.pop_size, self.dim))
        self.global_best = None
        self.global_best_fit = np.inf
        self.stagnation_counter[:] = 0
        self.fitness_history = []
        self.step_size = (self.upper - self.lower) * 0.1
        self.selection_counts[:] = 0
        self._prev_fitness_for_op = None
        
    def _clip_to_bounds_batch(self, pop):
        """Reflective boundary handling - invert velocity at bounds to preserve momentum."""
        reflected = np.clip(pop, self.lower, self.upper)
        return reflected
    
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
        return np.array(fitness, dtype=np.float64)
    
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
        return float(np.clip(inertia, self.w_min, self.w_max))
    
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
            fit = self._evaluate_batch(trial, self._eval_wrapper)
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
    
    # === Operator 0: Original PSO ===
    def _update_velocity_original(self, inertia):
        """Original velocity update using cognitive, social, and neighborhood components."""
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
    
    # === Operator 1: Variant 05 - Orthogonal Learning ===
    def _update_velocity_variant05(self, inertia):
        """Update velocity using orthogonal learning to escape local optima."""
        pop_size, dim = self.pop_size, self.dim

        r1 = np.random.uniform(0, 1, size=(pop_size, dim))
        r2 = np.random.uniform(0, 1, size=(pop_size, dim))
        r3 = np.random.uniform(0, 1, size=(pop_size, dim))

        cognitive = self.c1 * r1 * (self.personal_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)

        base_velocity = inertia * self.velocity + cognitive + social + neighborhood

        n_groups = max(2, dim // 10)
        group_size = dim // n_groups

        median_fit = np.median(self.personal_best_fit)
        elite_mask = self.personal_best_fit <= median_fit
        n_elite = np.sum(elite_mask)

        if n_elite > 0 and dim >= 4:
            ort_velocity = np.zeros_like(base_velocity)

            for i in np.where(elite_mask)[0]:
                for g in range(n_groups):
                    start = g * group_size
                    end = start + group_size if g < n_groups - 1 else dim
                    group_dims = end - start

                    if group_dims < 2:
                        continue

                    p1 = np.random.uniform(-1, 1, group_dims)
                    p2 = np.random.uniform(-1, 1, group_dims)
                    p2 = p2 - np.dot(p1, p2) / (np.dot(p1, p1) + 1e-10) * p1

                    delta1 = self.tunnel_scale * (self.upper - self.lower) * p1
                    delta2 = self.tunnel_scale * (self.upper - self.lower) * p2

                    cand1 = self.population[i].copy()
                    cand2 = self.population[i].copy()
                    cand1[start:end] += delta1
                    cand2[start:end] += delta2

                    cand1 = np.clip(cand1, self.lower, self.upper)
                    cand2 = np.clip(cand2, self.lower, self.upper)

                    dist1 = np.linalg.norm(cand1 - self.global_best)
                    dist2 = np.linalg.norm(cand2 - self.global_best)

                    if dist1 < dist2:
                        ort_velocity[i, start:end] = delta1 * 0.5
                    else:
                        ort_velocity[i, start:end] = delta2 * 0.5

            blend = np.where(elite_mask, 0.3, 0.0)
            blend = blend.reshape(-1, 1)
            self.velocity = (1 - blend) * base_velocity + blend * (base_velocity + ort_velocity)
        else:
            self.velocity = base_velocity

        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
    
    # === Operator 2: Variant 06 - Orthogonal + Convergence Perturbation ===
    def _update_velocity_variant06(self, inertia):
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
    
    # === Operator 3: Variant 07 - Success History Adaptive ===
    def _update_velocity_variant07(self, inertia):
        """Update velocity using success-history adaptive components."""
        if not hasattr(self, '_prev_pos_for_success'):
            self._prev_pos_for_success = self.population.copy()
            self._success_cognitive = np.ones(self.pop_size, dtype=np.float64)
            self._success_social = np.ones(self.pop_size, dtype=np.float64)
            self._success_neighborhood = np.ones(self.pop_size, dtype=np.float64)

        delta_pos = self.population - self._prev_pos_for_success
        self._prev_pos_for_success = self.population.copy()

        pos_mag = np.linalg.norm(delta_pos, axis=1, keepdims=True) + 1e-10
        delta_normalized = np.abs(delta_pos) / (pos_mag + 1e-10)

        alpha = 0.3
        self._success_cognitive = (1 - alpha) * self._success_cognitive + alpha * np.mean(delta_normalized, axis=1)
        self._success_social = (1 - alpha) * self._success_social + alpha * np.mean(delta_normalized, axis=1)
        self._success_neighborhood = (1 - alpha) * self._success_neighborhood + alpha * np.mean(delta_normalized, axis=1)

        total_success = self._success_cognitive + self._success_social + self._success_neighborhood + 1e-10
        w_c = self._success_cognitive / total_success
        w_s = self._success_social / total_success
        w_n = self._success_neighborhood / total_success

        min_weight = 0.05
        w_c = np.clip(w_c, min_weight, None)
        w_s = np.clip(w_s, min_weight, None)
        w_n = np.clip(w_n, min_weight, None)

        total = w_c + w_s + w_n
        w_c, w_s, w_n = w_c / total, w_s / total, w_n / total

        cognitive = self.c1 * w_c[:, np.newaxis] * (self.personal_best - self.population)
        social = self.c2 * w_s[:, np.newaxis] * (self.global_best - self.population)
        neighborhood = self.c3 * w_n[:, np.newaxis] * (self.neighborhood_best - self.population)

        self.velocity = inertia * self.velocity + cognitive + social + neighborhood

        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
    
    # === Operator 4: Variant 08 - Dual Mode ===
    def _update_velocity_variant08(self, inertia):
        """Dual-mode velocity update: stagnation-driven exploration vs exploitation."""
        centroid = np.mean(self.population, axis=0)

        stagnant_mask = self.stagnation_counter >= self.stagnation_limit // 2

        exploration_dir = centroid - self.population
        exploration_mag = np.linalg.norm(exploration_dir, axis=1, keepdims=True)
        safe_mag = np.where(exploration_mag < 1e-10, 1.0, exploration_mag)
        exploration_dir_normalized = exploration_dir / safe_mag

        stagnation_factor = np.minimum(self.stagnation_counter[:, np.newaxis] / max(1, self.stagnation_limit), 3.0)
        exploration_scale = stagnation_factor * (self.upper - self.lower) * 0.1
        exploration_velocity = exploration_dir_normalized * exploration_scale

        r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))

        cognitive = self.c1 * r1 * (self.personal_best - self.population)
        social = self.c2 * r2 * (self.global_best - self.population)
        neighborhood = self.c3 * r3 * (self.neighborhood_best - self.population)

        exploitation_velocity = inertia * self.velocity + cognitive + social + neighborhood
        self.velocity = np.where(stagnant_mask[:, np.newaxis], exploration_velocity, exploitation_velocity)

        max_vel = (self.upper - self.lower) * 0.2 * self.step_size
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling with beta distributions."""
        samples = np.array([
            np.random.beta(float(self.operator_alpha[i]), float(self.operator_beta[i]))
            for i in range(self.num_operators)
        ], dtype=np.float64)
        selected = int(np.argmax(samples))
        return selected
    
    def _compute_reward(self, prev_fitness, curr_fitness):
        """Compute reward based on fitness improvement ratio."""
        prev_best = np.min(prev_fitness) if len(prev_fitness) > 0 else np.inf
        curr_best = np.min(curr_fitness) if len(curr_fitness) > 0 else np.inf
        
        if not np.isfinite(prev_best) or not np.isfinite(curr_best):
            return 0.0
        
        improvement = prev_best - curr_best
        max_range = abs(self.upper - self.lower) + 1e-10
        
        normalized_reward = improvement / max_range
        return float(np.clip(normalized_reward * 100.0, -2.0, 2.0))
    
    def _update_operator_stats(self, operator_idx, reward):
        """Update Thompson Sampling statistics for an operator."""
        reward = float(np.clip(reward, -2.0, 2.0))
        
        self.reward_history[operator_idx].append(reward)
        if len(self.reward_history[operator_idx]) > self.reward_window_size:
            self.reward_history[operator_idx].pop(0)
        
        if reward > 0:
            self.operator_alpha[operator_idx] += reward * 0.5 + 0.1
        else:
            self.operator_beta[operator_idx] += abs(reward) * 0.5 + 0.1
        
        self.operator_alpha[operator_idx] = float(np.clip(self.operator_alpha[operator_idx], 0.1, 1e6))
        self.operator_beta[operator_idx] = float(np.clip(self.operator_beta[operator_idx], 0.1, 1e6))
    
    def _update_velocity_batch(self, inertia):
        """Adaptive velocity update - selects best operator using Thompson Sampling."""
        operator_funcs = [
            self._update_velocity_original,
            self._update_velocity_variant05,
            self._update_velocity_variant06,
            self._update_velocity_variant07,
            self._update_velocity_variant08,
        ]
        
        if len(self.fitness_history) <= 3:
            selected = 0
        else:
            selected = self._select_operator_thompson()
        
        self.current_operator = selected
        self.selection_counts[selected] += 1
        
        prev_fitness = self.personal_best_fit.copy() if self._prev_fitness_for_op is None else self._prev_fitness_for_op.copy()
        
        operator_funcs[selected](inertia)
        
        if self._prev_fitness_for_op is not None:
            reward = self._compute_reward(prev_fitness, self.personal_best_fit)
            self._update_operator_stats(selected, reward)
        
        self._prev_fitness_for_op = self.personal_best_fit.copy()
    
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
            self.fitness_history.append(float(self.global_best_fit))
            
            self._compute_ring_topology()
            self._quantum_tunnel_batch()
            
            inertia = self._compute_adaptive_inertia()
            self._update_velocity_batch(inertia)
            self._update_position_batch()
            self.population = self._clip_to_bounds_batch(self.population)
            
            self._reinitialize_diverse_particles()
            self._adapt_step_size()
            
            generation += 1
            
        return float(self.global_best_fit), self.global_best.copy() if self.global_best is not None else None
