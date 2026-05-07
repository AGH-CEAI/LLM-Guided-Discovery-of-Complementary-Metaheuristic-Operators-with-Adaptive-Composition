import numpy as np


class TopologyAdaptivePSO:
    """
    Particle Swarm Optimization with adaptive topology, local search,
    stagnation detection, and restart mechanisms.
    
    Key features:
    - Ring topology with adaptive neighborhood size
    - Velocity clamping with adaptive inertia
    - Comprehensive Learning PSO (CLPSO) inspired mutation
    - Stagnation detection with partial/full restarts
    - Opposition-based learning for restarts
    - Local refinement around global best
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = min(max(6 * dim, 40), 300)
        self.w_max = 0.9
        self.w_min = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_max = 0.5 * (self.ub - self.lb)
        self.neighborhood_size = 3
        self.max_neighborhood = max(self.pop_size // 2, 5)
        self.stagnation_limit = 15
        self.stagnation_counter = 0
        self.generation = 0
        self.max_restarts = 50
        self.restart_count = 0
        self.learning_probs = np.full(self.pop_size, 0.05)

    def __call__(self, func, stopping_condition):
        self._init_tracking()
        positions, velocities = self._initialize_population()
        fitness = func(positions)
        if stopping_condition():
            return self._finalize(positions, fitness)
        self._update_tracking(positions, fitness)
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        self._update_global_best(positions, fitness)

        while not stopping_condition():
            self.generation += 1
            w = self._compute_inertia()
            nbest_pos = self._get_neighborhood_best(pbest_pos, pbest_fit)
            
            # Generate parameters for the batch
            r1, r2, use_clpso = self._generate_parameters_batch()
            
            # Standard PSO velocity update
            velocities = self._update_velocities_batch(
                velocities, positions, pbest_pos, nbest_pos, w, r1, r2
            )
            
            # CLPSO-style comprehensive learning for selected particles
            velocities = self._apply_clpso_learning_batch(
                velocities, positions, pbest_pos, pbest_fit, w, use_clpso
            )
            
            velocities = self._clamp_velocities(velocities)
            positions = self._update_positions_batch(positions, velocities)
            positions = self._clip_to_bounds(positions)
            
            fitness = func(positions)
            if stopping_condition():
                break
            if len(fitness) < len(positions):
                positions = positions[:len(fitness)]
                velocities = velocities[:len(fitness)]
                pbest_pos = pbest_pos[:len(fitness)]
                pbest_fit = pbest_fit[:len(fitness)]
            
            self._update_tracking(positions, fitness)
            pbest_pos, pbest_fit = self._update_personal_best(
                positions, fitness, pbest_pos, pbest_fit
            )
            self._update_global_best(positions, fitness)
            
            # Stagnation check and possible restart
            stagnant = self._detect_stagnation()
            if stagnant:
                self._adapt_neighborhood_size()
                self._adapt_learning_probs(pbest_fit)
                
                if self.stagnation_counter >= self.stagnation_limit * 2:
                    if stopping_condition():
                        break
                    positions, velocities, pbest_pos, pbest_fit = self._restart(
                        positions, velocities, pbest_pos, pbest_fit, func, stopping_condition
                    )
                    if stopping_condition():
                        break
            
            # Local search around global best periodically
            if self.generation % 5 == 0 and not stopping_condition():
                self._local_search_batch(func, stopping_condition)
                if stopping_condition():
                    break

        return self.best_fit, self.best_pos.copy()

    def _init_tracking(self):
        self.best_fit = np.inf
        self.best_pos = None
        self.prev_best_fit = np.inf
        self.stagnation_counter = 0
        self.generation = 0
        self.restart_count = 0
        self.neighborhood_size = 3
        self.fit_history = []

    def _initialize_population(self):
        """Initialize positions uniformly and velocities to small random values."""
        positions = np.random.uniform(
            self.lb, self.ub, size=(self.pop_size, self.dim)
        )
        vel_range = 0.1 * (self.ub - self.lb)
        velocities = np.random.uniform(
            -vel_range, vel_range, size=(self.pop_size, self.dim)
        )
        return positions, velocities

    def _generate_parameters_batch(self):
        """Generate random parameters for the entire population at once."""
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        use_clpso = np.random.rand(self.pop_size) < self.learning_probs[:self.pop_size]
        return r1, r2, use_clpso

    def _compute_inertia(self):
        """Compute adaptive inertia weight based on stagnation."""
        # Base inertia decreases over generations
        progress = min(self.generation / max(1, 500), 1.0)
        w = self.w_max - (self.w_max - self.w_min) * progress
        # Increase inertia when stagnating to encourage exploration
        if self.stagnation_counter > 5:
            w = min(w + 0.15, self.w_max)
        return w

    def _get_neighborhood_best(self, pbest_pos, pbest_fit):
        """Get the best position in each particle's ring neighborhood."""
        n = len(pbest_pos)
        k = min(self.neighborhood_size, n - 1)
        nbest_pos = np.empty_like(pbest_pos)
        indices = np.arange(n)
        
        for i in range(n):
            # Ring topology neighbors
            neighbor_ids = [(i + j) % n for j in range(-k, k + 1)]
            neighbor_fits = pbest_fit[neighbor_ids]
            best_neighbor = neighbor_ids[np.argmin(neighbor_fits)]
            nbest_pos[i] = pbest_pos[best_neighbor]
        
        return nbest_pos

    def _update_velocities_batch(self, velocities, positions, pbest_pos, nbest_pos, w, r1, r2):
        """Standard PSO velocity update for the whole population."""
        cognitive = self.c1 * r1 * (pbest_pos - positions)
        social = self.c2 * r2 * (nbest_pos - positions)
        new_velocities = w * velocities + cognitive + social
        return new_velocities

    def _apply_clpso_learning_batch(self, velocities, positions, pbest_pos, pbest_fit, w, use_clpso):
        """
        Comprehensive Learning PSO: for selected particles, each dimension
        learns from a different particle's personal best.
        """
        n = len(positions)
        if not np.any(use_clpso):
            return velocities
        
        clpso_mask = use_clpso
        num_clpso = np.sum(clpso_mask)
        
        # For each dimension, pick a random exemplar via tournament selection
        exemplar_indices = np.zeros((num_clpso, self.dim), dtype=int)
        for d in range(self.dim):
            # Tournament of size 2
            a = np.random.randint(0, n, size=num_clpso)
            b = np.random.randint(0, n, size=num_clpso)
            better = np.where(pbest_fit[a] < pbest_fit[b], a, b)
            exemplar_indices[:, d] = better
        
        # Gather exemplar positions
        exemplar_pos = np.zeros((num_clpso, self.dim))
        for d in range(self.dim):
            exemplar_pos[:, d] = pbest_pos[exemplar_indices[:, d], d]
        
        r = np.random.rand(num_clpso, self.dim)
        clpso_vel = w * velocities[clpso_mask] + self.c1 * r * (exemplar_pos - positions[clpso_mask])
        velocities[clpso_mask] = clpso_vel
        
        return velocities

    def _clamp_velocities(self, velocities):
        """Clamp velocities to maximum allowed."""
        return np.clip(velocities, -self.v_max, self.v_max)

    def _update_positions_batch(self, positions, velocities):
        """Update positions by adding velocities."""
        return positions + velocities

    def _clip_to_bounds(self, positions):
        """Clip positions to search bounds."""
        return np.clip(positions, self.lb, self.ub)

    def _update_personal_best(self, positions, fitness, pbest_pos, pbest_fit):
        """Update personal best positions where fitness improved."""
        n = min(len(fitness), len(pbest_fit))
        improved = np.zeros(n, dtype=bool)
        valid = ~np.isnan(fitness[:n])
        improved[valid] = fitness[:n][valid] < pbest_fit[:n][valid]
        pbest_pos[:n][improved] = positions[:n][improved]
        pbest_fit[:n][improved] = fitness[:n][improved]
        return pbest_pos, pbest_fit

    def _update_global_best(self, positions, fitness):
        """Update the global best solution found so far."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fitness = fitness[valid]
        valid_positions = positions[valid]
        best_idx = np.argmin(valid_fitness)
        if valid_fitness[best_idx] < self.best_fit:
            self.best_fit = valid_fitness[best_idx]
            self.best_pos = valid_positions[best_idx].copy()

    def _update_tracking(self, positions, fitness):
        """Track fitness history for stagnation detection."""
        valid = ~np.isnan(fitness)
        if np.any(valid):
            current_best = np.min(fitness[valid])
            self.fit_history.append(current_best)

    def _detect_stagnation(self):
        """Detect if the swarm is stagnating (no improvement)."""
        if len(self.fit_history) < 2:
            self.stagnation_counter = 0
            return False
        
        current_best = self.fit_history[-1]
        rel_improvement = abs(self.prev_best_fit - current_best) / (abs(self.prev_best_fit) + 1e-30)
        
        if rel_improvement < 1e-10:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        self.prev_best_fit = current_best
        return self.stagnation_counter >= self.stagnation_limit

    def _adapt_neighborhood_size(self):
        """Increase neighborhood size when stagnating to speed up information flow."""
        self.neighborhood_size = min(
            self.neighborhood_size + 2,
            self.max_neighborhood
        )

    def _adapt_learning_probs(self, pbest_fit):
        """Adapt CLPSO learning probabilities based on particle quality."""
        n = len(pbest_fit)
        if n == 0:
            return
        # Worse particles should explore more (higher CLPSO probability)
        ranks = np.argsort(np.argsort(pbest_fit[:n])).astype(float)
        self.learning_probs[:n] = 0.05 + 0.45 * ranks / max(n - 1, 1)

    def _restart(self, positions, velocities, pbest_pos, pbest_fit, func, stopping_condition):
        """
        Restart strategy: reinitialize worst particles using opposition-based learning,
        keep the best particles intact.
        """
        self.restart_count += 1
        self.stagnation_counter = 0
        self.neighborhood_size = 3  # Reset neighborhood
        
        n = len(positions)
        num_keep = max(n // 5, 2)  # Keep top 20%
        
        sort_idx = np.argsort(pbest_fit[:n])
        keep_idx = sort_idx[:num_keep]
        replace_idx = sort_idx[num_keep:]
        num_replace = len(replace_idx)
        
        if num_replace == 0:
            return positions, velocities, pbest_pos, pbest_fit
        
        # Opposition-based reinitialization
        new_positions = self._opposition_based_init(
            positions[keep_idx], num_replace
        )
        positions[replace_idx] = new_positions
        
        vel_range = 0.1 * (self.ub - self.lb)
        velocities[replace_idx] = np.random.uniform(
            -vel_range, vel_range, size=(num_replace, self.dim)
        )
        
        positions = self._clip_to_bounds(positions)
        
        if not stopping_condition():
            new_fitness = func(positions[replace_idx])
            if stopping_condition():
                # Just update what we can
                valid_len = min(len(new_fitness), num_replace)
                for i in range(valid_len):
                    if not np.isnan(new_fitness[i]):
                        idx = replace_idx[i]
                        pbest_pos[idx] = positions[idx].copy()
                        pbest_fit[idx] = new_fitness[i]
                        if new_fitness[i] < self.best_fit:
                            self.best_fit = new_fitness[i]
                            self.best_pos = positions[idx].copy()
            else:
                valid_len = min(len(new_fitness), num_replace)
                for i in range(valid_len):
                    if not np.isnan(new_fitness[i]):
                        idx = replace_idx[i]
                        pbest_pos[idx] = positions[idx].copy()
                        pbest_fit[idx] = new_fitness[i]
                        if new_fitness[i] < self.best_fit:
                            self.best_fit = new_fitness[i]
                            self.best_pos = positions[idx].copy()
        
        return positions, velocities, pbest_pos, pbest_fit

    def _opposition_based_init(self, elite_positions, num_new):
        """Generate new positions using opposition-based learning from elites."""
        center = np.mean(elite_positions, axis=0)
        
        # Mix of opposition and random
        new_pos = np.empty((num_new, self.dim))
        num_opposition = num_new // 2
        num_random = num_new - num_opposition
        
        if num_opposition > 0:
            # Opposition of center with perturbation
            base = self.lb + self.ub - center
            perturbation = np.random.normal(0, 20.0, size=(num_opposition, self.dim))
            new_pos[:num_opposition] = base + perturbation
        
        if num_random > 0:
            # Random in full space, biased away from center
            random_pos = np.random.uniform(self.lb, self.ub, size=(num_random, self.dim))
            new_pos[num_opposition:] = random_pos
        
        return np.clip(new_pos, self.lb, self.ub)

    def _local_search_batch(self, func, stopping_condition):
        """
        Perform a local search around the global best using a batch of
        perturbations with adaptive step size.
        """
        if self.best_pos is None or stopping_condition():
            return
        
        num_trials = min(self.pop_size // 2, 20)
        
        # Adaptive step size based on stagnation
        if self.stagnation_counter > 10:
            scale = 0.5
        elif self.stagnation_counter > 5:
            scale = 1.0
        else:
            scale = 5.0
        
        # Cauchy distribution for heavy tails (better exploration)
        perturbations = np.random.standard_cauchy(size=(num_trials, self.dim)) * scale
        trials = self.best_pos + perturbations
        trials = self._clip_to_bounds(trials)
        
        trial_fitness = func(trials)
        if stopping_condition():
            # Still try to extract improvements
            pass
        
        valid = ~np.isnan(trial_fitness)
        if np.any(valid):
            valid_fit = trial_fitness[valid]
            valid_pos = trials[valid]
            best_idx = np.argmin(valid_fit)
            if valid_fit[best_idx] < self.best_fit:
                self.best_fit = valid_fit[best_idx]
                self.best_pos = valid_pos[best_idx].copy()
                self.stagnation_counter = 0

    def _compute_diversity(self, positions):
        """Compute population diversity as mean distance from centroid."""
        centroid = np.mean(positions, axis=0)
        distances = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
        return np.mean(distances)

    def _finalize(self, positions, fitness):
        """Finalize and return the best solution found."""
        valid = ~np.isnan(fitness)
        if np.any(valid):
            best_idx = np.argmin(fitness[valid])
            valid_positions = positions[valid]
            if fitness[valid][best_idx] < (self.best_fit if self.best_pos is not None else np.inf):
                self.best_fit = fitness[valid][best_idx]
                self.best_pos = valid_positions[best_idx].copy()
        
        if self.best_pos is None:
            self.best_pos = positions[0].copy()
            self.best_fit = float('inf')
        
        return self.best_fit, self.best_pos.copy()
