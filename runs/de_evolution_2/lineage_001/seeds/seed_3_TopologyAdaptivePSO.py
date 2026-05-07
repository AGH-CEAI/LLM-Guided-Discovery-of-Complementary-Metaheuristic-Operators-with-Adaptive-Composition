import numpy as np


class TopologyAdaptivePSO:
    """
    Particle Swarm Optimization with adaptive topology neighborhoods,
    velocity clamping, inertia weight adaptation, and stagnation-based
    topology restructuring.
    
    Uses a dynamic ring/random topology that rewires when the swarm
    stagnates, combined with a local-best model and opposition-based
    reinitialization for escaped particles.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np = min(max(6 * dim, 40), 300)
        self.k_neighbors = max(3, self.np // 10)  # neighborhood size
        self.w_init = 0.9
        self.w_min = 0.2
        self.c1 = 2.05
        self.c2 = 2.05
        self.v_max_ratio = 0.3
        self.stagnation_limit = 15
        self.topology_type = "ring"  # starts with ring, can switch

    def _initialize_population(self):
        """Create initial positions, velocities, and personal bests."""
        positions = np.random.uniform(self.lb, self.ub, size=(self.np, self.dim))
        v_max = self.v_max_ratio * (self.ub - self.lb)
        velocities = np.random.uniform(-v_max, v_max, size=(self.np, self.dim))
        return positions, velocities

    def _initialize_personal_bests(self, positions, fitness):
        """Set personal best positions and fitness values."""
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        return pbest_pos, pbest_fit

    def _build_ring_topology(self):
        """Build a ring topology where each particle connects to k nearest indices."""
        neighbors = np.zeros((self.np, self.k_neighbors), dtype=int)
        half_k = self.k_neighbors // 2
        for i in range(self.np):
            offsets = np.arange(-half_k, half_k + 1)
            offsets = offsets[offsets != 0]
            if len(offsets) > self.k_neighbors:
                offsets = offsets[:self.k_neighbors]
            elif len(offsets) < self.k_neighbors:
                extra = np.arange(half_k + 1, half_k + 1 + self.k_neighbors - len(offsets))
                offsets = np.concatenate([offsets, extra])
            neighbors[i] = (i + offsets[:self.k_neighbors]) % self.np
        return neighbors

    def _build_random_topology(self):
        """Build a random topology where each particle has k random neighbors."""
        neighbors = np.zeros((self.np, self.k_neighbors), dtype=int)
        for i in range(self.np):
            candidates = np.delete(np.arange(self.np), i)
            chosen = np.random.choice(candidates, size=self.k_neighbors, replace=False)
            neighbors[i] = chosen
        return neighbors

    def _build_topology(self):
        """Dispatch to the appropriate topology builder."""
        if self.topology_type == "ring":
            return self._build_ring_topology()
        else:
            return self._build_random_topology()

    def _compute_neighborhood_bests(self, neighbors, pbest_pos, pbest_fit):
        """For each particle, find the best position among its neighbors."""
        nbest_pos = np.zeros_like(pbest_pos)
        nbest_fit = np.full(self.np, np.inf)
        for i in range(self.np):
            nb_indices = neighbors[i]
            # Include self in neighborhood
            all_indices = np.concatenate([[i], nb_indices])
            nb_fits = pbest_fit[all_indices]
            best_local = all_indices[np.argmin(nb_fits)]
            nbest_pos[i] = pbest_pos[best_local]
            nbest_fit[i] = pbest_fit[best_local]
        return nbest_pos, nbest_fit

    def _adapt_inertia_weight(self, generation, max_approx_generations):
        """Linearly decay inertia weight from w_init to w_min."""
        progress = min(generation / max(max_approx_generations, 1), 1.0)
        w = self.w_init - (self.w_init - self.w_min) * progress
        return w

    def _compute_constriction_factor(self):
        """Compute Clerc's constriction factor for velocity update stability."""
        phi = self.c1 + self.c2
        if phi > 4.0:
            chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
        else:
            chi = 1.0
        return chi

    def _update_velocities_batch(self, velocities, positions, pbest_pos, nbest_pos, w):
        """Update all velocities in a single vectorized operation."""
        chi = self._compute_constriction_factor()
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        cognitive = self.c1 * r1 * (pbest_pos - positions)
        social = self.c2 * r2 * (nbest_pos - positions)
        new_velocities = chi * (w * velocities + cognitive + social)
        return new_velocities

    def _clamp_velocities_batch(self, velocities):
        """Clamp velocities to prevent explosion."""
        v_max = self.v_max_ratio * (self.ub - self.lb)
        return np.clip(velocities, -v_max, v_max)

    def _update_positions_batch(self, positions, velocities):
        """Move all particles by their velocities and clip to bounds."""
        new_positions = positions + velocities
        new_positions = np.clip(new_positions, self.lb, self.ub)
        return new_positions

    def _reflect_out_of_bounds_batch(self, positions, velocities):
        """Reflect velocities for particles that hit bounds and clip positions."""
        below = positions < self.lb
        above = positions > self.ub
        velocities[below | above] *= -0.5
        positions = np.clip(positions, self.lb, self.ub)
        return positions, velocities

    def _update_personal_bests_batch(self, positions, fitness, pbest_pos, pbest_fit):
        """Update personal bests where current fitness improves upon stored best."""
        valid = np.isfinite(fitness)
        improved = valid & (fitness < pbest_fit)
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]
        return pbest_pos, pbest_fit

    def _update_global_best(self, pbest_pos, pbest_fit, gbest_pos, gbest_fit):
        """Track the overall best solution found so far."""
        valid_mask = np.isfinite(pbest_fit)
        if not np.any(valid_mask):
            return gbest_pos, gbest_fit
        best_idx = np.argmin(np.where(valid_mask, pbest_fit, np.inf))
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()
        return gbest_pos, gbest_fit

    def _detect_stagnation(self, gbest_fit, gbest_fit_history, stagnation_counter):
        """Check if the global best has not improved for several generations."""
        if len(gbest_fit_history) > 0 and gbest_fit >= gbest_fit_history[-1] - 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        gbest_fit_history.append(gbest_fit)
        return stagnation_counter, gbest_fit_history

    def _restructure_topology_on_stagnation(self, stagnation_counter):
        """Switch topology type and rebuild when stagnation is detected."""
        if stagnation_counter >= self.stagnation_limit:
            if self.topology_type == "ring":
                self.topology_type = "random"
            else:
                self.topology_type = "ring"
            # Also increase neighborhood size slightly
            self.k_neighbors = min(self.k_neighbors + 1, self.np - 1)
            neighbors = self._build_topology()
            return neighbors, 0  # reset stagnation counter
        return None, stagnation_counter

    def _opposition_reinitialize_worst_batch(self, positions, fitness, fraction=0.1):
        """Reinitialize a fraction of the worst particles using opposition-based learning."""
        n_reinit = max(1, int(self.np * fraction))
        valid_fit = np.where(np.isfinite(fitness), fitness, np.inf)
        worst_indices = np.argsort(valid_fit)[-n_reinit:]
        # Opposition-based: reflect through the center of the search space
        center = (self.lb + self.ub) / 2.0
        for idx in worst_indices:
            opposition = 2.0 * center - positions[idx]
            # Add some noise
            noise = np.random.normal(0, 0.1 * (self.ub - self.lb), size=self.dim)
            positions[idx] = np.clip(opposition + noise, self.lb, self.ub)
        return positions, worst_indices

    def _compute_diversity(self, positions):
        """Compute population diversity as average distance from centroid."""
        centroid = np.mean(positions, axis=0)
        distances = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
        return np.mean(distances)

    def _adaptive_velocity_max(self, diversity):
        """Adjust velocity clamping based on swarm diversity."""
        range_size = self.ub - self.lb
        max_diversity = np.sqrt(self.dim) * range_size / 2.0
        diversity_ratio = diversity / max(max_diversity, 1e-10)
        # When diversity is low, allow larger velocities to explore
        if diversity_ratio < 0.01:
            self.v_max_ratio = min(0.6, self.v_max_ratio * 1.2)
        elif diversity_ratio > 0.3:
            self.v_max_ratio = max(0.1, self.v_max_ratio * 0.9)
        return self.v_max_ratio

    def _cauchy_perturbation_batch(self, gbest_pos, n_trials=5):
        """Generate Cauchy-perturbed candidates around the global best for local search."""
        perturbations = np.random.standard_cauchy(size=(n_trials, self.dim))
        scale = 0.01 * (self.ub - self.lb)
        candidates = gbest_pos[np.newaxis, :] + scale * perturbations
        candidates = np.clip(candidates, self.lb, self.ub)
        return candidates

    def _safe_evaluate(self, func, candidates, stopping_condition):
        """Evaluate candidates with proper truncation and stopping checks."""
        if stopping_condition():
            return None
        fitness = func(candidates)
        n_valid = len(fitness)
        if n_valid < len(candidates):
            candidates = candidates[:n_valid]
            fitness = fitness[:n_valid]
        # Replace any NaN with inf
        fitness = np.where(np.isfinite(fitness), fitness, np.inf)
        return fitness

    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating all components."""
        # Initialize
        positions, velocities = self._initialize_population()
        fitness = self._safe_evaluate(func, positions, stopping_condition)
        if fitness is None:
            return 0.0, np.zeros(self.dim)

        pbest_pos, pbest_fit = self._initialize_personal_bests(positions, fitness)
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        neighbors = self._build_topology()
        stagnation_counter = 0
        gbest_fit_history = []
        generation = 0
        approx_max_gen = 500  # rough estimate for inertia schedule

        while not stopping_condition():
            generation += 1

            # Adapt inertia weight
            w = self._adapt_inertia_weight(generation, approx_max_gen)

            # Compute neighborhood bests
            nbest_pos, nbest_fit = self._compute_neighborhood_bests(
                neighbors, pbest_pos, pbest_fit
            )

            # Update velocities and positions
            velocities = self._update_velocities_batch(
                velocities, positions, pbest_pos, nbest_pos, w
            )
            velocities = self._clamp_velocities_batch(velocities)
            positions = self._update_positions_batch(positions, velocities)
            positions, velocities = self._reflect_out_of_bounds_batch(positions, velocities)

            # Evaluate new positions
            fitness = self._safe_evaluate(func, positions, stopping_condition)
            if fitness is None:
                break
            if len(fitness) < self.np:
                # Partial evaluation — update what we can and break
                pbest_pos[:len(fitness)], pbest_fit[:len(fitness)] = \
                    self._update_personal_bests_batch(
                        positions[:len(fitness)], fitness,
                        pbest_pos[:len(fitness)], pbest_fit[:len(fitness)]
                    )
                gbest_pos, gbest_fit = self._update_global_best(
                    pbest_pos, pbest_fit, gbest_pos, gbest_fit
                )
                break

            if stopping_condition():
                # Still update bests with what we got
                pbest_pos, pbest_fit = self._update_personal_bests_batch(
                    positions, fitness, pbest_pos, pbest_fit
                )
                gbest_pos, gbest_fit = self._update_global_best(
                    pbest_pos, pbest_fit, gbest_pos, gbest_fit
                )
                break

            # Update personal and global bests
            pbest_pos, pbest_fit = self._update_personal_bests_batch(
                positions, fitness, pbest_pos, pbest_fit
            )
            gbest_pos, gbest_fit = self._update_global_best(
                pbest_pos, pbest_fit, gbest_pos, gbest_fit
            )

            # Stagnation detection and topology restructuring
            stagnation_counter, gbest_fit_history = self._detect_stagnation(
                gbest_fit, gbest_fit_history, stagnation_counter
            )
            new_neighbors, stagnation_counter = self._restructure_topology_on_stagnation(
                stagnation_counter
            )
            if new_neighbors is not None:
                neighbors = new_neighbors
                # Also do opposition reinitialization of worst particles
                positions, worst_idx = self._opposition_reinitialize_worst_batch(
                    positions, fitness, fraction=0.15
                )
                # Reset velocities of reinitialized particles
                v_max = self.v_max_ratio * (self.ub - self.lb)
                velocities[worst_idx] = np.random.uniform(
                    -v_max * 0.5, v_max * 0.5, size=(len(worst_idx), self.dim)
                )

            # Adaptive velocity clamping based on diversity
            diversity = self._compute_diversity(positions)
            self._adaptive_velocity_max(diversity)

            # Local search around global best every few generations
            if generation % 10 == 0 and not stopping_condition():
                cauchy_candidates = self._cauchy_perturbation_batch(gbest_pos, n_trials=5)
                cauchy_fit = self._safe_evaluate(func, cauchy_candidates, stopping_condition)
                if cauchy_fit is not None and len(cauchy_fit) > 0:
                    valid_mask = np.isfinite(cauchy_fit)
                    if np.any(valid_mask):
                        best_cauchy = np.argmin(
                            np.where(valid_mask, cauchy_fit, np.inf)
                        )
                        if cauchy_fit[best_cauchy] < gbest_fit:
                            gbest_fit = cauchy_fit[best_cauchy]
                            gbest_pos = cauchy_candidates[best_cauchy].copy()

        return gbest_fit, gbest_pos
