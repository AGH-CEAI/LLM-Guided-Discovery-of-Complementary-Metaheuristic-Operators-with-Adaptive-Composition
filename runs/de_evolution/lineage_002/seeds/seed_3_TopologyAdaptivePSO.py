import numpy as np


class TopologyAdaptivePSO:
    """Particle Swarm Optimization with adaptive topology, local search,
    stagnation detection, and restart mechanisms."""

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np = min(max(8 * dim, 80), 400)
        self.w_init = 0.9
        self.w_min = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.neighborhood_size = max(3, self.np // 10)
        self.stagnation_threshold = 15
        self.velocity_clamp = 0.3 * (self.ub - self.lb)
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func, stopping_condition):
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)

        positions, velocities = self._initialize_population()
        fitness = self._evaluate_batch(func, positions)
        if fitness is None or stopping_condition():
            return self.best_f, self.best_x

        pbest_pos, pbest_fit = self._init_personal_best(positions, fitness)
        topology = self._build_ring_topology()
        self._update_global_best(positions, fitness)

        generation = 0
        stagnation_counter = 0
        prev_best = self.best_f
        w = self.w_init

        while not stopping_condition():
            generation += 1
            w = self._adapt_inertia(generation, w)

            lbest_pos = self._compute_local_best_batch(pbest_pos, pbest_fit, topology)

            velocities = self._update_velocities_batch(
                positions, velocities, pbest_pos, lbest_pos, w
            )
            positions = self._update_positions_batch(positions, velocities)
            positions = self._clip_bounds_batch(positions)

            fitness = self._evaluate_batch(func, positions)
            if fitness is None or stopping_condition():
                break

            pbest_pos, pbest_fit = self._update_personal_best_batch(
                positions, fitness, pbest_pos, pbest_fit
            )
            self._update_global_best(positions, fitness)

            stagnation_counter, improved = self._detect_stagnation(
                prev_best, self.best_f, stagnation_counter
            )
            prev_best = self.best_f

            if stagnation_counter >= self.stagnation_threshold:
                positions, velocities, pbest_pos, pbest_fit, topology, stagnation_counter = (
                    self._restart(func, stopping_condition, positions, velocities,
                                  pbest_pos, pbest_fit)
                )
                if stopping_condition():
                    break
                prev_best = self.best_f

            if generation % 5 == 0 and not stopping_condition():
                topology = self._adapt_topology(generation, stagnation_counter)

            if generation % 10 == 0 and not stopping_condition():
                positions, fitness, pbest_pos, pbest_fit = self._local_search_batch(
                    func, positions, fitness, pbest_pos, pbest_fit
                )
                if stopping_condition():
                    break

        return self.best_f, self.best_x

    def _initialize_population(self):
        """Create random positions and velocities within bounds."""
        positions = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        vel_range = 0.1 * (self.ub - self.lb)
        velocities = np.random.uniform(-vel_range, vel_range, (self.np, self.dim))
        return positions, velocities

    def _evaluate_batch(self, func, candidates):
        """Evaluate a batch of candidates, updating global best. Returns fitness or None."""
        candidates = self._clip_bounds_batch(candidates)
        fitness = func(candidates)
        if fitness is None:
            return None
        valid = len(fitness)
        if valid == 0:
            return None
        fitness = np.array(fitness, dtype=np.float64)
        # Handle truncation
        if valid < len(candidates):
            fitness = fitness[:valid]
            candidates = candidates[:valid]
        # Update global best from valid, non-NaN entries
        mask = ~np.isnan(fitness)
        if np.any(mask):
            idx = np.nanargmin(fitness)
            if fitness[idx] < self.best_f:
                self.best_f = fitness[idx]
                self.best_x = candidates[idx].copy()
        # Pad with inf if truncated
        if valid < self.np:
            padded = np.full(self.np, np.inf)
            padded[:valid] = fitness
            return padded
        return fitness

    def _init_personal_best(self, positions, fitness):
        """Initialize personal best positions and fitness."""
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        return pbest_pos, pbest_fit

    def _build_ring_topology(self):
        """Build a ring topology with neighborhood_size neighbors on each side."""
        topology = np.zeros((self.np, self.neighborhood_size * 2 + 1), dtype=int)
        for i in range(self.np):
            neighbors = []
            for k in range(-self.neighborhood_size, self.neighborhood_size + 1):
                neighbors.append((i + k) % self.np)
            topology[i] = neighbors
        return topology

    def _adapt_topology(self, generation, stagnation_counter):
        """Adapt topology based on stagnation: increase neighborhood when stagnant."""
        if stagnation_counter > self.stagnation_threshold // 2:
            # Use larger neighborhoods when stagnating
            k = min(self.np // 2, self.neighborhood_size * 2)
        else:
            k = self.neighborhood_size
        topology = np.zeros((self.np, 2 * k + 1), dtype=int)
        for i in range(self.np):
            neighbors = [(i + j) % self.np for j in range(-k, k + 1)]
            topology[i] = neighbors
        return topology

    def _compute_local_best_batch(self, pbest_pos, pbest_fit, topology):
        """Compute the local (neighborhood) best for each particle."""
        lbest_pos = np.empty_like(pbest_pos)
        for i in range(self.np):
            neighbors = topology[i]
            neighbor_fits = pbest_fit[neighbors]
            # Handle NaN
            valid_mask = ~np.isnan(neighbor_fits)
            if np.any(valid_mask):
                best_local_idx = neighbors[np.nanargmin(neighbor_fits)]
            else:
                best_local_idx = i
            lbest_pos[i] = pbest_pos[best_local_idx]
        return lbest_pos

    def _update_velocities_batch(self, positions, velocities, pbest_pos, lbest_pos, w):
        """Update velocities using PSO equations with constriction-like approach."""
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cognitive = self.c1 * r1 * (pbest_pos - positions)
        social = self.c2 * r2 * (lbest_pos - positions)
        velocities = w * velocities + cognitive + social
        # Clamp velocities
        velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)
        return velocities

    def _update_positions_batch(self, positions, velocities):
        """Update positions by adding velocities."""
        new_positions = positions + velocities
        return new_positions

    def _clip_bounds_batch(self, positions):
        """Clip positions to the search bounds."""
        return np.clip(positions, self.lb, self.ub)

    def _update_personal_best_batch(self, positions, fitness, pbest_pos, pbest_fit):
        """Update personal best for each particle if improved."""
        improved = fitness < pbest_fit
        # Also handle NaN in current pbest
        improved = improved | np.isnan(pbest_fit)
        # But only if fitness itself is valid
        improved = improved & ~np.isnan(fitness)
        pbest_pos[improved] = positions[improved].copy()
        pbest_fit[improved] = fitness[improved]
        return pbest_pos, pbest_fit

    def _update_global_best(self, positions, fitness):
        """Update global best from current fitness values."""
        mask = ~np.isnan(fitness)
        if np.any(mask):
            idx = np.nanargmin(fitness)
            if fitness[idx] < self.best_f:
                self.best_f = fitness[idx]
                self.best_x = positions[idx].copy()

    def _adapt_inertia(self, generation, w):
        """Linearly decrease inertia weight over generations."""
        decay_rate = 0.995
        w_new = max(self.w_min, w * decay_rate)
        return w_new

    def _detect_stagnation(self, prev_best, current_best, stagnation_counter):
        """Detect if the algorithm has stagnated (no improvement)."""
        rel_improvement = abs(prev_best - current_best) / (abs(prev_best) + 1e-30)
        if rel_improvement < 1e-12:
            stagnation_counter += 1
            return stagnation_counter, False
        else:
            return 0, True

    def _restart(self, func, stopping_condition, positions, velocities,
                 pbest_pos, pbest_fit):
        """Partial restart: reinitialize worst particles, keep elite."""
        n_elite = max(2, self.np // 5)
        n_random = self.np - n_elite

        # Sort by personal best fitness
        valid_fit = np.where(np.isnan(pbest_fit), np.inf, pbest_fit)
        sorted_idx = np.argsort(valid_fit)
        elite_idx = sorted_idx[:n_elite]

        # Keep elite positions
        new_positions = np.empty((self.np, self.dim))
        new_velocities = np.empty((self.np, self.dim))
        new_positions[:n_elite] = positions[elite_idx]
        new_velocities[:n_elite] = velocities[elite_idx] * 0.5

        # Reinitialize rest with opposition-based or random
        # Use Gaussian around best + some fully random
        n_around_best = n_random // 2
        n_fully_random = n_random - n_around_best

        if self.best_x is not None:
            scale = 30.0  # Moderate exploration radius
            gaussian_pop = self.best_x + scale * np.random.randn(n_around_best, self.dim)
            new_positions[n_elite:n_elite + n_around_best] = gaussian_pop
        else:
            new_positions[n_elite:n_elite + n_around_best] = np.random.uniform(
                self.lb, self.ub, (n_around_best, self.dim))

        new_positions[n_elite + n_around_best:] = np.random.uniform(
            self.lb, self.ub, (n_fully_random, self.dim))

        vel_range = 0.1 * (self.ub - self.lb)
        new_velocities[n_elite:] = np.random.uniform(
            -vel_range, vel_range, (n_random, self.dim))

        new_positions = self._clip_bounds_batch(new_positions)

        if not stopping_condition():
            fitness = self._evaluate_batch(func, new_positions)
            if fitness is not None:
                new_pbest_pos = new_positions.copy()
                new_pbest_fit = fitness.copy()
                # Restore elite personal bests if they were better
                for i, ei in enumerate(elite_idx):
                    if pbest_fit[ei] < new_pbest_fit[i]:
                        new_pbest_pos[i] = pbest_pos[ei].copy()
                        new_pbest_fit[i] = pbest_fit[ei]
            else:
                new_pbest_pos = new_positions.copy()
                new_pbest_fit = np.full(self.np, np.inf)
        else:
            new_pbest_pos = new_positions.copy()
            new_pbest_fit = np.full(self.np, np.inf)

        topology = self._build_ring_topology()
        stagnation_counter = 0
        return new_positions, new_velocities, new_pbest_pos, new_pbest_fit, topology, stagnation_counter

    def _local_search_batch(self, func, positions, fitness, pbest_pos, pbest_fit):
        """Apply a batch local search (perturbation) around the best particles."""
        n_local = max(2, self.np // 5)
        valid_fit = np.where(np.isnan(fitness), np.inf, fitness)
        top_idx = np.argsort(valid_fit)[:n_local]

        # Create perturbations around top particles
        scale = 1.0  # Small perturbation
        perturbations = np.empty((n_local, self.dim))
        for i, idx in enumerate(top_idx):
            perturbations[i] = positions[idx] + scale * np.random.randn(self.dim)
        perturbations = self._clip_bounds_batch(perturbations)

        pert_fitness = self._evaluate_batch(func, perturbations)
        if pert_fitness is None:
            return positions, fitness, pbest_pos, pbest_fit

        # Only take improvements, up to n_local valid results
        valid_count = min(n_local, len(pert_fitness))
        for i in range(valid_count):
            idx = top_idx[i]
            if not np.isnan(pert_fitness[i]) and pert_fitness[i] < fitness[idx]:
                positions[idx] = perturbations[i].copy()
                fitness[idx] = pert_fitness[i]
                if pert_fitness[i] < pbest_fit[idx]:
                    pbest_pos[idx] = perturbations[i].copy()
                    pbest_fit[idx] = pert_fitness[i]

        return positions, fitness, pbest_pos, pbest_fit
