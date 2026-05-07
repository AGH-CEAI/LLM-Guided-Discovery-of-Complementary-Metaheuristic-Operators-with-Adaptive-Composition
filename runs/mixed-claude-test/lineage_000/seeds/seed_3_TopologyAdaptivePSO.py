import numpy as np


class TopologyAdaptivePSO:
    """
    Particle Swarm Optimization with adaptive ring/star topology switching,
    velocity clamping, inertia weight adaptation based on swarm diversity,
    and periodic stochastic reinitialization of stagnant particles.
    
    Key ideas:
    - Ring topology for exploration (each particle influenced by k nearest neighbors)
    - Star topology for exploitation (global best influences all)
    - Topology mixing ratio adapts based on improvement rate
    - Cauchy mutation on personal bests to escape local optima
    - Opposition-based learning for reinitialization of stagnant particles
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np = min(max(6 * dim, 40), 300)
        self.k_neighbors = max(3, self.np // 10)
        
        # PSO parameters
        self.w_max = 0.9
        self.w_min = 0.3
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_max_ratio = 0.15
        
        # Topology mixing: 0 = full ring, 1 = full star
        self.topology_ratio = 0.5
        self.topology_adapt_rate = 0.05
        
        # Stagnation detection
        self.stagnation_limit = 15
        self.cauchy_scale = 5.0

    def _initialize_population(self):
        positions = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        v_range = (self.ub - self.lb) * self.v_max_ratio
        velocities = np.random.uniform(-v_range, v_range, (self.np, self.dim))
        return positions, velocities

    def _initialize_personal_bests(self, positions, fitness):
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        return pbest_pos, pbest_fit

    def _find_global_best(self, pbest_pos, pbest_fit):
        valid = np.isfinite(pbest_fit)
        if not np.any(valid):
            return pbest_pos[0].copy(), pbest_fit[0]
        best_idx = np.argmin(pbest_fit[valid])
        actual_idx = np.where(valid)[0][best_idx]
        return pbest_pos[actual_idx].copy(), pbest_fit[actual_idx]

    def _compute_ring_neighbors(self, pbest_pos, pbest_fit):
        """For each particle, find the best among its k ring neighbors."""
        n = len(pbest_pos)
        k = self.k_neighbors
        half_k = k // 2
        nbest_pos = np.empty_like(pbest_pos)
        
        for i in range(n):
            neighbor_indices = [(i + j) % n for j in range(-half_k, half_k + 1)]
            neighbor_fits = pbest_fit[neighbor_indices]
            valid = np.isfinite(neighbor_fits)
            if np.any(valid):
                local_best_idx = neighbor_indices[np.argmin(neighbor_fits)]
            else:
                local_best_idx = i
            nbest_pos[i] = pbest_pos[local_best_idx]
        
        return nbest_pos

    def _compute_social_component(self, positions, pbest_pos, pbest_fit, gbest_pos):
        """Blend ring topology and star topology based on topology_ratio."""
        ring_best = self._compute_ring_neighbors(pbest_pos, pbest_fit)
        
        # Blend: social target = (1 - ratio) * ring_best + ratio * gbest
        ratio = self.topology_ratio
        social_target = (1.0 - ratio) * ring_best + ratio * gbest_pos[np.newaxis, :]
        return social_target

    def _compute_inertia_weight(self, positions):
        """Adapt inertia weight based on population diversity."""
        centroid = np.mean(positions, axis=0)
        distances = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
        avg_dist = np.mean(distances)
        max_dist = np.sqrt(self.dim) * (self.ub - self.lb)
        
        # Normalize diversity to [0, 1]
        diversity = min(avg_dist / (max_dist + 1e-30), 1.0)
        
        # High diversity -> lower inertia (exploit); low diversity -> higher inertia (explore)
        w = self.w_min + (self.w_max - self.w_min) * (1.0 - diversity)
        return np.clip(w, self.w_min, self.w_max)

    def _update_velocities_batch(self, velocities, positions, pbest_pos, social_target, w):
        """Update all velocities in batch using cognitive + social components."""
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.c1 * r1 * (pbest_pos - positions)
        social = self.c2 * r2 * (social_target - positions)
        
        new_velocities = w * velocities + cognitive + social
        return new_velocities

    def _clamp_velocities_batch(self, velocities):
        """Clamp velocities to prevent explosion."""
        v_max = (self.ub - self.lb) * self.v_max_ratio
        return np.clip(velocities, -v_max, v_max)

    def _update_positions_batch(self, positions, velocities):
        """Move all particles according to their velocities."""
        new_positions = positions + velocities
        return np.clip(new_positions, self.lb, self.ub)

    def _reflect_velocities_at_bounds(self, positions, velocities):
        """Reverse velocity component when particle hits a boundary."""
        at_lower = positions <= self.lb
        at_upper = positions >= self.ub
        velocities[at_lower | at_upper] *= -0.5
        return velocities

    def _update_personal_bests_batch(self, positions, fitness, pbest_pos, pbest_fit):
        """Update personal bests for all particles where fitness improved."""
        valid = np.isfinite(fitness)
        improved = valid & (fitness < pbest_fit)
        
        pbest_pos[improved] = positions[improved].copy()
        pbest_fit[improved] = fitness[improved]
        
        return pbest_pos, pbest_fit, improved

    def _adapt_topology_ratio(self, improved_mask, prev_gbest_fit, gbest_fit):
        """Shift topology toward star if global best improved, toward ring otherwise."""
        improvement_count = np.sum(improved_mask)
        improvement_rate = improvement_count / max(len(improved_mask), 1)
        
        if gbest_fit < prev_gbest_fit:
            # Global improvement: lean toward current topology slightly more
            self.topology_ratio = min(1.0, self.topology_ratio + self.topology_adapt_rate)
        elif improvement_rate < 0.05:
            # Very few personal improvements: switch toward ring (exploration)
            self.topology_ratio = max(0.0, self.topology_ratio - self.topology_adapt_rate * 2)
        else:
            # Some personal improvements but no global: slight shift to ring
            self.topology_ratio = max(0.0, self.topology_ratio - self.topology_adapt_rate * 0.5)

    def _detect_stagnant_particles(self, stagnation_counters):
        """Return boolean mask of particles that have been stagnant too long."""
        return stagnation_counters >= self.stagnation_limit

    def _update_stagnation_counters(self, counters, improved_mask):
        """Increment stagnation for non-improved, reset for improved."""
        counters[improved_mask] = 0
        counters[~improved_mask] += 1
        return counters

    def _cauchy_mutate_stagnant_batch(self, positions, velocities, pbest_pos, 
                                       gbest_pos, stagnant_mask):
        """Apply Cauchy perturbation to stagnant particles' personal bests."""
        n_stagnant = np.sum(stagnant_mask)
        if n_stagnant == 0:
            return positions, velocities
        
        # Strategy 1: Cauchy mutation around global best for half
        # Strategy 2: Opposition-based reinitialization for other half
        stagnant_indices = np.where(stagnant_mask)[0]
        np.random.shuffle(stagnant_indices)
        split = len(stagnant_indices) // 2
        
        cauchy_indices = stagnant_indices[:max(split, 1)]
        opposition_indices = stagnant_indices[max(split, 1):]
        
        # Cauchy mutation around gbest
        scale = self.cauchy_scale * np.random.uniform(0.5, 1.5)
        cauchy_perturbation = np.random.standard_cauchy((len(cauchy_indices), self.dim)) * scale
        positions[cauchy_indices] = gbest_pos + cauchy_perturbation
        velocities[cauchy_indices] *= 0.1
        
        # Opposition-based learning
        if len(opposition_indices) > 0:
            positions[opposition_indices] = (self.lb + self.ub) - positions[opposition_indices]
            # Add small random noise
            noise = np.random.uniform(-5, 5, (len(opposition_indices), self.dim))
            positions[opposition_indices] += noise
            velocities[opposition_indices] = np.random.uniform(
                -5, 5, (len(opposition_indices), self.dim))
        
        positions = np.clip(positions, self.lb, self.ub)
        return positions, velocities

    def _adaptive_cauchy_scale(self, generation, max_gen_estimate):
        """Reduce Cauchy mutation scale over time."""
        progress = min(generation / max(max_gen_estimate, 1), 1.0)
        self.cauchy_scale = 5.0 * (1.0 - 0.8 * progress)

    def _shrink_coefficients_late_stage(self, generation, max_gen_estimate):
        """Reduce c1, c2 and topology ratio in late optimization stages."""
        progress = min(generation / max(max_gen_estimate, 1), 1.0)
        if progress > 0.7:
            late_factor = 1.0 - 0.3 * ((progress - 0.7) / 0.3)
            self.c1 = 2.0 * late_factor + 0.5 * (1.0 - late_factor)
            self.c2 = 2.0 * late_factor + 2.5 * (1.0 - late_factor)
            self.topology_ratio = min(self.topology_ratio + 0.01, 1.0)

    def _safe_handle_partial_eval(self, fitness, n_expected):
        """Handle case where func returns fewer values than expected."""
        if len(fitness) < n_expected:
            padded = np.full(n_expected, np.nan)
            padded[:len(fitness)] = fitness[:len(fitness)]
            return padded, True
        return fitness, False

    def _track_best(self, gbest_pos, gbest_fit, pbest_pos, pbest_fit):
        """Update the global best from personal bests."""
        valid = np.isfinite(pbest_fit)
        if not np.any(valid):
            return gbest_pos, gbest_fit
        best_idx = np.argmin(pbest_fit[valid])
        actual_idx = np.where(valid)[0][best_idx]
        if np.isfinite(pbest_fit[actual_idx]) and pbest_fit[actual_idx] < gbest_fit:
            return pbest_pos[actual_idx].copy(), pbest_fit[actual_idx]
        return gbest_pos, gbest_fit

    def __call__(self, func, stopping_condition):
        # Initialize swarm
        positions, velocities = self._initialize_population()
        
        # Evaluate initial population
        fitness = func(positions)
        fitness, truncated = self._safe_handle_partial_eval(fitness, self.np)
        if truncated or stopping_condition():
            valid = np.isfinite(fitness)
            if np.any(valid):
                best_idx = np.argmin(fitness[valid])
                actual_idx = np.where(valid)[0][best_idx]
                return float(fitness[actual_idx]), positions[actual_idx].copy()
            return float('inf'), positions[0].copy()
        
        # Initialize personal and global bests
        pbest_pos, pbest_fit = self._initialize_personal_bests(positions, fitness)
        gbest_pos, gbest_fit = self._find_global_best(pbest_pos, pbest_fit)
        
        # Stagnation counters
        stagnation_counters = np.zeros(self.np, dtype=int)
        
        # Estimate max generations for schedule
        budget_estimate = self.np * 200
        max_gen_estimate = budget_estimate // self.np
        generation = 0
        
        # Reset PSO coefficients
        self.c1 = 2.0
        self.c2 = 2.0
        self.topology_ratio = 0.5
        self.cauchy_scale = 5.0
        
        while not stopping_condition():
            generation += 1
            prev_gbest_fit = gbest_fit
            
            # Adapt parameters
            w = self._compute_inertia_weight(positions)
            self._adaptive_cauchy_scale(generation, max_gen_estimate)
            self._shrink_coefficients_late_stage(generation, max_gen_estimate)
            
            # Compute social influence (blended topology)
            social_target = self._compute_social_component(
                positions, pbest_pos, pbest_fit, gbest_pos)
            
            # Update velocities and positions
            velocities = self._update_velocities_batch(
                velocities, positions, pbest_pos, social_target, w)
            velocities = self._clamp_velocities_batch(velocities)
            positions = self._update_positions_batch(positions, velocities)
            velocities = self._reflect_velocities_at_bounds(positions, velocities)
            positions = np.clip(positions, self.lb, self.ub)
            
            # Evaluate
            fitness = func(positions)
            fitness, truncated = self._safe_handle_partial_eval(fitness, self.np)
            
            if stopping_condition():
                gbest_pos, gbest_fit = self._track_best(
                    gbest_pos, gbest_fit, pbest_pos, pbest_fit)
                break
            
            if truncated:
                # Update what we can and break
                valid_count = np.sum(np.isfinite(fitness))
                if valid_count > 0:
                    pbest_pos, pbest_fit, _ = self._update_personal_bests_batch(
                        positions, fitness, pbest_pos, pbest_fit)
                    gbest_pos, gbest_fit = self._track_best(
                        gbest_pos, gbest_fit, pbest_pos, pbest_fit)
                break
            
            # Update personal bests
            pbest_pos, pbest_fit, improved = self._update_personal_bests_batch(
                positions, fitness, pbest_pos, pbest_fit)
            
            # Update global best
            gbest_pos, gbest_fit = self._track_best(
                gbest_pos, gbest_fit, pbest_pos, pbest_fit)
            
            # Adapt topology
            self._adapt_topology_ratio(improved, prev_gbest_fit, gbest_fit)
            
            # Handle stagnation
            stagnation_counters = self._update_stagnation_counters(
                stagnation_counters, improved)
            stagnant_mask = self._detect_stagnant_particles(stagnation_counters)
            
            if np.any(stagnant_mask):
                positions, velocities = self._cauchy_mutate_stagnant_batch(
                    positions, velocities, pbest_pos, gbest_pos, stagnant_mask)
                stagnation_counters[stagnant_mask] = 0
        
        return float(gbest_fit), gbest_pos.copy()
