

```python
import numpy as np


class TopologyAdaptivePSO:
    """Particle Swarm Optimization with adaptive topology, local search,
    stagnation detection, and restart mechanisms."""

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np = min(max(6 * dim, 60), 300)
        self.w_init = 0.9
        self.w_min = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.c3 = 0.5  # neighborhood best coefficient
        self.k_neighbors = 3  # ring neighborhood size
        self.velocity_clamp = 0.5 * (self.ub - self.lb)
        self.stagnation_limit = 15
        self.stagnation_counter = 0
        self.restart_count = 0
        self.generation = 0
        self.max_restarts = 50

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        positions, velocities = self._initialize_population()
        fitness = func(positions)
        if stopping_condition():
            self._update_global_best(positions, fitness)
            return self.f_opt, self.x_opt

        fitness = self._handle_truncated(fitness, len(positions))
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        self._update_global_best(positions, fitness)

        self.generation = 0
        prev_best = self.f_opt

        while not stopping_condition():
            self.generation += 1
            w = self._compute_inertia()

            nbest_pos = self._get_neighborhood_best_batch(pbest_pos, pbest_fit)
            velocities = self._update_velocities_batch(
                positions, velocities, pbest_pos, nbest_pos, w
            )
            positions = self._update_positions_batch(positions, velocities)
            positions = self._clip_bounds_batch(positions)

            fitness = func(positions)
            if stopping_condition():
                fitness = self._handle_truncated(fitness, len(positions))
                self._update_personal_best_batch(positions, fitness, pbest_pos, pbest_fit)
                self._update_global_best(pbest_pos, pbest_fit)
                break

            fitness = self._handle_truncated(fitness, len(positions))
            pbest_pos, pbest_fit = self._update_personal_best_batch(
                positions, fitness, pbest_pos, pbest_fit
            )
            self._update_global_best(pbest_pos, pbest_fit)

            # Opposition-based learning for diversity
            if self.generation % 20 == 0 and not stopping_condition():
                opp_positions = self._opposition_batch(positions)
                opp_fitness = func(opp_positions)
                if stopping_condition():
                    opp_fitness = self._handle_truncated(opp_fitness, len(opp_positions))
                    self._update_global_best(opp_positions, opp_fitness)
                    break
                opp_fitness = self._handle_truncated(opp_fitness, len(opp_positions))
                positions, fitness, pbest_pos, pbest_fit = self._merge_opposition_batch(
                    positions, fitness, opp_positions, opp_fitness, pbest_pos, pbest_fit
                )
                self._update_global_best(pbest_pos, pbest_fit)

            # Local search around global best
            if self.generation % 10 == 0 and not stopping_condition():
                local_positions = self._local_search_batch()
                local_fitness = func(local_positions)
                if stopping_condition():
                    local_fitness = self._handle_truncated(local_fitness, len(local_positions))
                    self._update_global_best(local_positions, local_fitness)
                    break
                local_fitness = self._handle_truncated(local_fitness, len(local_positions))
                self._update_global_best(local_positions, local_fitness)
                # Inject best local search results into population
                positions, fitness, pbest_pos, pbest_fit = self._inject_local_results(
                    positions, fitness, local_positions, local_fitness, pbest_pos, pbest_fit
                )

            # Stagnation detection and restart
            stagnated = self._detect_stagnation(prev_best)
            if stagnated and not stopping_condition():
                positions, velocities, fitness, pbest_pos, pbest_fit = self._restart(
                    func, stopping_condition, pbest_pos, pbest_fit
                )
                if stopping_condition():
                    break
            prev_best = self.f_opt

            # Adaptive topology
            self._adapt_topology(fitness)

        return self.f_opt, self.x_opt

    def _initialize_population(self):
        """Initialize positions uniformly and velocities to small random values."""
        positions = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        max_vel = 0.1 * (self.ub - self.lb)
        velocities = np.random.uniform(-max_vel, max_vel, (self.np, self.dim))
        return positions, velocities

    def _compute_inertia(self):
        """Linearly decrease inertia weight over generations, with floor."""
        decay_rate = 0.995
        w = self.w_init * (decay_rate ** self.generation)
        return max(w, self.w_min)

    def _get_neighborhood_best_batch(self, pbest_pos, pbest_fit):
        """For each particle, find the best in its ring neighborhood."""
        n = len(pbest_pos)
        k = min(self.k_neighbors, n - 1)
        nbest_pos = np.empty_like(pbest_pos)
        indices = np.arange(n)

        for i in range(n):
            # Ring topology: neighbors are the k nearest indices (wrapping)
            neighbor_idx = [(i + j) % n for j in range(-k // 2, k // 2 + 1)]
            neighbor_fits = pbest_fit[neighbor_idx]
            valid_mask = np.isfinite(neighbor_fits)
            if np.any(valid_mask):
                best_local = neighbor_idx[np.argmin(neighbor_fits[valid_mask])]
                nbest_pos[i] = pbest_pos[best_local]
            else:
                nbest_pos[i] = pbest_pos[i]

        return nbest_pos

    def _update_velocities_batch(self, positions, velocities, pbest_pos, nbest_pos, w):
        """Update all velocities using cognitive, social, and neighborhood components."""
        n = len(positions)
        r1 = np.random.uniform(0, 1, (n, self.dim))
        r2 = np.random.uniform(0, 1, (n, self.dim))
        r3 = np.random.uniform(0, 1, (n, self.dim))

        cognitive = self.c1 * r1 * (pbest_pos - positions)
        social = self.c2 * r2 * (self.x_opt[np.newaxis, :] - positions)
        neighborhood = self.c3 * r3 * (nbest_pos - positions)

        new_vel = w * velocities + cognitive + social + neighborhood
        new_vel = np.clip(new_vel, -self.velocity_clamp, self.velocity_clamp)
        return new_vel

    def _update_positions_batch(self, positions, velocities):
        """Move all particles by their velocities."""
        return positions + velocities

    def _clip_bounds_batch(self, positions):
        """Clip all positions to the search bounds."""
        return np.clip(positions, self.lb, self.ub)

    def _update_personal_best_batch(self, positions, fitness, pbest_pos, pbest_fit):
        """Update personal bests where current fitness is better."""
        valid = np.isfinite(fitness)
        improved = valid & (fitness < pbest_fit)
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]
        return pbest_pos, pbest_fit

    def _update_global_best(self, positions, fitness):
        """Update the global best solution found so far."""
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return
        valid_idx = np.where(valid)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.f_opt:
            self.f_opt = fitness[best_idx]
            self.x_opt = positions[best_idx].copy()

    def _handle_truncated(self, fitness, expected_len):
        """Handle cases where func returns fewer values than expected."""
        if len(fitness) < expected_len:
            padded = np.full(expected_len, np.nan)
            padded[:len(fitness)] = fitness
            return padded
        return fitness

    def _opposition_batch(self, positions):
        """Generate opposition-based positions for the entire population."""
        opp = self.lb + self.ub - positions
        # Add small perturbation
        noise = np.random.normal(0, 0.05 * (self.ub - self.lb), positions.shape)
        opp = opp + noise
        return np.clip(opp, self.lb, self.ub)

    def _merge_opposition_batch(self, positions, fitness, opp_positions, opp_fitness,
                                 pbest_pos, pbest_fit):
        """Replace particles with opposition particles where opposition is better."""
        valid_opp = np.isfinite(opp_fitness)
        valid_orig = np.isfinite(fitness)
        n = min(len(positions), len(opp_positions))

        for i in range(n):
            if valid_opp[i] and (not valid_orig[i] or opp_fitness[i] < fitness[i]):
                positions[i] = opp_positions[i]
                fitness[i] = opp_fitness[i]
                if opp_fitness[i] < pbest_fit[i]:
                    pbest_pos[i] = opp_positions[i]
                    pbest_fit[i] = opp_fitness[i]

        return positions, fitness, pbest_pos, pbest_fit

    def _local_search_batch(self):
        """Generate a batch of candidates near the global best for local refinement."""
        n_local = min(self.dim, 30)
        # Adaptive radius based on generation
        radius = max(0.01, 10.0 * (0.98 ** self.generation))
        perturbations = np.random.normal(0, radius, (n_local, self.dim))
        candidates = self.x_opt[np.newaxis, :] + perturbations
        return np.clip(candidates, self.lb, self.ub)

    def _inject_local_results(self, positions, fitness, local_pos, local_fit,
                               pbest_pos, pbest_fit):
        """Inject the best local search results into the worst particles."""
        valid_local = np.isfinite(local_fit)
        if not np.any(valid_local):
            return positions, fitness, pbest_pos, pbest_fit

        valid_idx = np.where(valid_local)[0]
        sorted_local = valid_idx[np.argsort(local_fit[valid_idx])]

        # Find worst particles in population
        valid_pop = np.isfinite(fitness)
        if not np.any(valid_pop):
            return positions, fitness, pbest_pos, pbest_fit

        pop_order = np.argsort(-fitness)  # worst first (but handle NaN)
        n_inject = min(len(sorted_local), max(1, self.np // 10))

        for i in range(n_inject):
            li = sorted_local[i]
            pi = pop_order[i]
            if local_fit[li] < fitness[pi] or not np.isfinite(fitness[pi]):
                positions[pi] = local_pos[li]
                fitness[pi] = local_fit[li]
                if local_fit[li] < pbest_fit[pi]:
                    pbest_pos[pi] = local_pos[li]
                    pbest_fit[pi] = local_fit[li]

        return positions, fitness, pbest_pos, pbest_fit

    def _detect_stagnation(self, prev_best):
        """Detect if the optimizer has stagnated (no improvement for many generations)."""
        if self.f_opt < prev_best - 1e-12 * max(1.0, abs(prev_best)):
            self.stagnation_counter = 0
            return False
        else:
            self.stagnation_counter += 1
            return self.stagnation_counter >= self.stagnation_limit

    def _restart(self, func, stopping_condition, pbest_pos, pbest_fit):
        """Perform a partial restart: reinitialize worst particles, keep best ones."""
        self.restart_count += 1
        self.stagnation_counter = 0

        n_keep = max(2, self.np // 5)
        valid = np.isfinite(pbest_fit)

        if np.any(valid):
            valid_idx = np.where(valid)[0]
            sorted_idx = valid_idx[np.argsort(pbest_fit[valid_idx])]
            keep_idx = sorted_idx[:n_keep]
        else:
            keep_idx = np.arange(n_keep)

        # Create new population
        positions = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        # Keep elite particles
        for i, ki in enumerate(keep_idx):
            if i < self.np:
                positions[i] = pbest_pos[ki]

        # Reinitialize with some particles near the global best
        n_near = min(self.np // 4, self.np - n_keep)
        if n_near > 0:
            radius = max(1.0, 50.0 * (0.9 ** self.restart_count))
            for i in range(n_near):
                idx = n_keep + i
                if idx < self.np:
                    positions[idx] = self.x_opt + np.random.normal(0, radius, self.dim)

        positions = np.clip(positions, self.lb, self.ub)

        max_vel = 0.1 * (self.ub - self.lb)
        velocities = np.random.uniform(-max_vel, max_vel, (self.np, self.dim))

        if stopping_condition():
            fitness = np.full(self.np, np.nan)
            return positions, velocities, fitness, positions.copy(), fitness.copy()

        fitness = func(positions)
        fitness = self._handle_truncated(fitness, self.np)

        new_pbest_pos = positions.copy()
        new_pbest_fit = fitness.copy()

        self._update_global_best(positions, fitness)

        # Increase neighborhood size after restart to improve exploration
        self.k_neighbors = min(self.k_neighbors + 2, self.np - 1)

        return positions, velocities, fitness, new_pbest_pos, new_pbest_fit

    def _adapt_topology(self, fitness):
        """Adapt the neighborhood size based on population diversity."""
        diversity = self._compute_diversity(fitness)
        if diversity < 1e-6:
            # Very low diversity -> increase neighborhood for exploration pressure
            self.k_neighbors = min(self.k_neighbors + 1, self.np - 1)
        elif diversity > 50.0:
            # High diversity -> smaller neighborhoods for exploitation
            self.k_neighbors = max(3, self.k_neighbors - 1)

    def _compute_diversity(self, fitness):
        """Compute fitness diversity as standard deviation of valid fitness values."""
        valid = np.isfinite(fitness)
        if np.sum(valid) < 2:
            return 0.0
        return float(np.std(fitness[valid]))
```