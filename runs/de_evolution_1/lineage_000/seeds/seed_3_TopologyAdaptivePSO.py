import numpy as np


class TopologyAdaptivePSO:
    """
    Particle Swarm Optimization with adaptive ring/star topology switching,
    velocity clamping, inertia weight adaptation based on swarm diversity,
    and periodic stochastic reset of stagnant particles.
    
    Key ideas:
    - Ring topology for exploration, star topology for exploitation
    - Topology mixing ratio adapts based on improvement rate
    - Cauchy mutation on global best for escaping local optima
    - Diversity-based inertia weight control
    - Stagnant particle reinitialization with Levy-flight-like jumps
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 40), 300)
        self.w_min = 0.3
        self.w_max = 0.9
        self.c1 = 1.8
        self.c2 = 1.8
        self.v_max = 0.3 * (self.ub - self.lb)
        self.ring_size = 3
        self.topology_ratio = 0.5  # 0 = full ring, 1 = full star
        self.stagnation_limit = 15
        self.cauchy_scale = 1.0
        self.improvement_history = []

    def _initialize_population(self):
        positions = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        velocities = np.random.uniform(
            -0.1 * (self.ub - self.lb),
            0.1 * (self.ub - self.lb),
            (self.np_size, self.dim)
        )
        return positions, velocities

    def _initialize_personal_bests(self, positions, fitness):
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        return pbest_pos, pbest_fit

    def _find_global_best(self, pbest_pos, pbest_fit):
        valid = ~np.isnan(pbest_fit)
        if not np.any(valid):
            return pbest_pos[0].copy(), np.inf
        best_idx = np.nanargmin(pbest_fit)
        return pbest_pos[best_idx].copy(), pbest_fit[best_idx]

    def _compute_ring_neighbors_best(self, pbest_pos, pbest_fit):
        """For each particle, find the best in its ring neighborhood."""
        n = self.np_size
        nbest_pos = np.empty_like(pbest_pos)
        nbest_fit = np.full(n, np.inf)
        half_k = self.ring_size // 2
        for offset in range(-half_k, half_k + 1):
            neighbor_idx = (np.arange(n) + offset) % n
            better = pbest_fit[neighbor_idx] < nbest_fit
            nbest_fit[better] = pbest_fit[neighbor_idx[better]]
            nbest_pos[better] = pbest_pos[neighbor_idx[better]]
        return nbest_pos, nbest_fit

    def _compute_topology_attractors(self, pbest_pos, pbest_fit, gbest_pos):
        """Blend ring-neighborhood best and global best based on topology_ratio."""
        ring_pos, ring_fit = self._compute_ring_neighbors_best(pbest_pos, pbest_fit)
        ratio = self.topology_ratio
        # Each particle's social attractor is a weighted blend
        attractors = (1.0 - ratio) * ring_pos + ratio * gbest_pos[np.newaxis, :]
        return attractors

    def _compute_diversity(self, positions):
        """Compute normalized swarm diversity as mean distance to centroid."""
        centroid = np.mean(positions, axis=0)
        diffs = positions - centroid[np.newaxis, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        max_diag = np.sqrt(self.dim) * (self.ub - self.lb)
        diversity = np.mean(distances) / (max_diag + 1e-30)
        return diversity

    def _adapt_inertia_weight(self, diversity):
        """Higher diversity -> lower inertia (let swarm converge),
        lower diversity -> higher inertia (encourage exploration)."""
        # Map diversity (roughly 0-0.5) to inertia weight
        # Low diversity (<0.01) => high w, high diversity (>0.2) => low w
        norm_div = np.clip(diversity / 0.25, 0.0, 1.0)
        w = self.w_max - (self.w_max - self.w_min) * norm_div
        return w

    def _adapt_topology_ratio(self):
        """Adapt topology mixing based on recent improvement rate.
        More improvement -> favor star (exploitation).
        Less improvement -> favor ring (exploration)."""
        if len(self.improvement_history) < 5:
            return
        recent = self.improvement_history[-10:]
        improvement_rate = np.mean(recent)
        if improvement_rate > 0.1:
            self.topology_ratio = min(1.0, self.topology_ratio + 0.05)
        elif improvement_rate < 0.02:
            self.topology_ratio = max(0.0, self.topology_ratio - 0.05)

    def _update_velocities_batch(self, velocities, positions, pbest_pos, attractors, w):
        """Standard PSO velocity update with topology-blended social component."""
        r1 = np.random.rand(self.np_size, self.dim)
        r2 = np.random.rand(self.np_size, self.dim)
        cognitive = self.c1 * r1 * (pbest_pos - positions)
        social = self.c2 * r2 * (attractors - positions)
        new_vel = w * velocities + cognitive + social
        # Clamp velocities
        new_vel = np.clip(new_vel, -self.v_max, self.v_max)
        return new_vel

    def _update_positions_batch(self, positions, velocities):
        """Move particles and clip to bounds."""
        new_pos = positions + velocities
        new_pos = np.clip(new_pos, self.lb, self.ub)
        return new_pos

    def _update_personal_bests_batch(self, positions, fitness, pbest_pos, pbest_fit):
        """Update personal bests where improvement is found."""
        valid = ~np.isnan(fitness)
        improved = valid & (fitness < pbest_fit)
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]
        num_improved = np.sum(improved)
        return pbest_pos, pbest_fit, num_improved

    def _detect_stagnant_particles(self, stagnation_counters):
        """Identify particles that haven't improved in many generations."""
        return stagnation_counters >= self.stagnation_limit

    def _reinitialize_stagnant_batch(self, positions, velocities, stagnation_counters, gbest_pos):
        """Reset stagnant particles using Levy-flight-like jumps from global best."""
        stagnant = self._detect_stagnant_particles(stagnation_counters)
        n_stagnant = np.sum(stagnant)
        if n_stagnant == 0:
            return positions, velocities, stagnation_counters
        # Levy-flight-like: some near gbest, some random
        near_fraction = 0.5
        n_near = int(n_stagnant * near_fraction)
        n_far = n_stagnant - n_near
        stagnant_idx = np.where(stagnant)[0]
        np.random.shuffle(stagnant_idx)
        # Near gbest with Cauchy perturbation
        if n_near > 0:
            near_idx = stagnant_idx[:n_near]
            perturbation = np.random.standard_cauchy((n_near, self.dim)) * self.cauchy_scale * 10.0
            positions[near_idx] = gbest_pos[np.newaxis, :] + perturbation
        # Far: random reinitialization
        if n_far > 0:
            far_idx = stagnant_idx[n_near:]
            positions[far_idx] = np.random.uniform(self.lb, self.ub, (n_far, self.dim))
        positions[stagnant] = np.clip(positions[stagnant], self.lb, self.ub)
        velocities[stagnant] = np.random.uniform(
            -0.05 * (self.ub - self.lb), 0.05 * (self.ub - self.lb),
            (n_stagnant, self.dim)
        )
        stagnation_counters[stagnant] = 0
        return positions, velocities, stagnation_counters

    def _cauchy_mutation_on_gbest(self, gbest_pos, gbest_fit, gen):
        """Generate a batch of Cauchy-mutated variants around global best."""
        n_mutants = max(5, self.np_size // 10)
        scale = self.cauchy_scale * max(1.0, 50.0 / (1.0 + gen))
        perturbations = np.random.standard_cauchy((n_mutants, self.dim)) * scale
        mutants = gbest_pos[np.newaxis, :] + perturbations
        mutants = np.clip(mutants, self.lb, self.ub)
        return mutants

    def _adapt_cauchy_scale(self, diversity):
        """Scale Cauchy mutations based on current diversity."""
        self.cauchy_scale = np.clip(diversity * 200.0, 0.1, 20.0)

    def _handle_truncated_fitness(self, trial_batch, trial_fit):
        """Handle case where func returns fewer values than expected."""
        if len(trial_fit) < len(trial_batch):
            n = len(trial_fit)
            return trial_batch[:n], trial_fit[:n], True
        # Also handle NaN at the end
        valid_count = np.sum(~np.isnan(trial_fit))
        if valid_count < len(trial_fit):
            # Keep only valid
            mask = ~np.isnan(trial_fit)
            return trial_batch[mask], trial_fit[mask], False
        return trial_batch, trial_fit, False

    def _shuffle_ring_topology(self):
        """Periodically shuffle particle indices to refresh ring neighborhoods."""
        return np.random.permutation(self.np_size)

    def __call__(self, func, stopping_condition):
        # Initialize
        positions, velocities = self._initialize_population()
        fitness = func(positions)
        if stopping_condition():
            best_idx = np.nanargmin(fitness)
            return float(fitness[best_idx]), positions[best_idx].copy()

        # Handle truncation
        if len(fitness) < self.np_size:
            n = len(fitness)
            positions = positions[:n]
            velocities = velocities[:n]
            self.np_size = n

        pbest_pos, pbest_fit = self._initialize_personal_bests(positions, fitness)
        gbest_pos, gbest_fit = self._find_global_best(pbest_pos, pbest_fit)
        stagnation_counters = np.zeros(self.np_size, dtype=int)
        
        # Ring index permutation
        ring_perm = np.arange(self.np_size)
        gen = 0

        while not stopping_condition():
            # Compute diversity and adapt parameters
            diversity = self._compute_diversity(positions)
            w = self._adapt_inertia_weight(diversity)
            self._adapt_topology_ratio()
            self._adapt_cauchy_scale(diversity)

            # Periodically shuffle ring topology
            if gen > 0 and gen % 20 == 0:
                ring_perm = self._shuffle_ring_topology()

            # Compute topology-aware attractors using permuted order
            perm_pbest_pos = pbest_pos[ring_perm]
            perm_pbest_fit = pbest_fit[ring_perm]
            perm_attractors = self._compute_topology_attractors(
                perm_pbest_pos, perm_pbest_fit, gbest_pos
            )
            # Invert permutation to get attractors in original order
            inv_perm = np.argsort(ring_perm)
            attractors = perm_attractors[inv_perm]

            # Update velocities and positions
            velocities = self._update_velocities_batch(
                velocities, positions, pbest_pos, attractors, w
            )
            positions = self._update_positions_batch(positions, velocities)

            # Evaluate
            fitness = func(positions)
            if stopping_condition():
                break
            if len(fitness) < self.np_size:
                fitness = np.concatenate([fitness, np.full(self.np_size - len(fitness), np.nan)])

            # Update personal and global bests
            pbest_pos, pbest_fit, num_improved = self._update_personal_bests_batch(
                positions, fitness, pbest_pos, pbest_fit
            )
            improvement_rate = num_improved / self.np_size
            self.improvement_history.append(improvement_rate)

            new_gbest_pos, new_gbest_fit = self._find_global_best(pbest_pos, pbest_fit)
            if new_gbest_fit < gbest_fit:
                gbest_pos, gbest_fit = new_gbest_pos, new_gbest_fit

            # Update stagnation counters
            valid_fitness = ~np.isnan(fitness)
            improved_mask = valid_fitness & (fitness <= pbest_fit + 1e-15)
            stagnation_counters[~improved_mask] += 1
            stagnation_counters[improved_mask & valid_fitness] = 0

            # Reinitialize stagnant particles
            positions, velocities, stagnation_counters = self._reinitialize_stagnant_batch(
                positions, velocities, stagnation_counters, gbest_pos
            )

            # Cauchy mutation on global best
            if gen % 3 == 0 and not stopping_condition():
                mutants = self._cauchy_mutation_on_gbest(gbest_pos, gbest_fit, gen)
                mut_fit = func(mutants)
                if stopping_condition():
                    # Still check for improvements
                    valid_mut = ~np.isnan(mut_fit[:len(mut_fit)])
                    if np.any(valid_mut):
                        best_mut_idx = np.nanargmin(mut_fit)
                        if mut_fit[best_mut_idx] < gbest_fit:
                            gbest_fit = float(mut_fit[best_mut_idx])
                            gbest_pos = mutants[best_mut_idx].copy()
                    break
                if len(mut_fit) > 0:
                    valid_mut = ~np.isnan(mut_fit)
                    if np.any(valid_mut):
                        best_mut_idx = np.nanargmin(mut_fit)
                        if mut_fit[best_mut_idx] < gbest_fit:
                            gbest_fit = float(mut_fit[best_mut_idx])
                            gbest_pos = mutants[best_mut_idx].copy()

            gen += 1

        # Final check across all personal bests
        final_gbest_pos, final_gbest_fit = self._find_global_best(pbest_pos, pbest_fit)
        if final_gbest_fit < gbest_fit:
            gbest_fit = final_gbest_fit
            gbest_pos = final_gbest_pos

        return float(gbest_fit), gbest_pos.copy()
