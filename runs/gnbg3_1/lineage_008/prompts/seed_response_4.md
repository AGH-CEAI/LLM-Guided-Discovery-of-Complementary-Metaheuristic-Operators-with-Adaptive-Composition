```python
import numpy as np


class AdaptiveMemeticDifferentialEvolution:
    """
    Hybrid DE with Nelder-Mead local search and adaptive strategy selection.
    Uses ring topology with adaptive neighborhood for mutation base selection.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = kwargs.get('NP', min(180, max(4 * dim, 6 * dim)))
        self.lb = -100.0
        self.ub = 100.0

        # DE parameters
        self.F = 0.7
        self.CR = 0.85
        self.F_adapt = 0.7

        # Local search parameters
        self.ls_trigger_rate = kwargs.get('ls_trigger_rate', 0.08)
        self.ls_alpha = 1.0
        self.ls_gamma = 2.0
        self.ls_sigma = 0.5
        self.ls_beta = 0.5

        # Strategy pool with rewards
        self._init_strategy_pool()

        # Ring topology
        self.ring_radius = kwargs.get('ring_radius', 3)
        self.max_ring_radius = kwargs.get('max_ring_radius', dim // 4)

        # Diversity and restart
        self.min_diversity = kwargs.get('min_diversity', 1e-6)
        self.stagnation_limit = kwargs.get('stagnation_limit', 50)

        # State tracking
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.gbest = None
        self.gfit = np.inf
        self.gen = 0
        self.stagnation_gen = 0
        self.ls_calls = 0
        self.ls_budget_frac = 0.05

        # Reward history for strategy adaptation
        self.strategy_success = np.zeros(len(self.strategies))
        self.strategy_trials = np.ones(len(self.strategies))

    def _init_strategy_pool(self):
        """Initialize mutation strategies and their selection weights."""
        self.strategies = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_best,
            self._mutate_rand_to_best,
            self._mutate_rand_2,
            self._mutate_current_to_pbest,
        ]
        self.strategy_weights = np.ones(len(self.strategies))
        self.strategy_weights /= self.strategy_weights.sum()

    def _initialize_population(self):
        """Create initial population uniformly in bounds."""
        self.population = np.random.uniform(
            self.lb, self.ub, size=(self.NP, self.dim)
        )
        self.population = np.clip(self.population, self.lb, self.ub)

    def _evaluate_batch(self, pop, func):
        """Evaluate fitness for entire population at once."""
        fitness = func(pop)
        if len(fitness) < len(pop):
            fitness = np.pad(fitness, (0, len(pop) - len(fitness)), constant_values=np.inf)
        return fitness

    def _select_ring_neighbors(self, idx):
        """Select neighbors from ring topology around index."""
        n = self.NP
        radius = self.ring_radius
        neighbors = []
        for offset in range(1, radius + 1):
            neighbors.append((idx - offset) % n)
            neighbors.append((idx + offset) % n)
        return np.array(neighbors)

    def _adapt_ring_radius(self):
        """Adapt ring radius based on diversity."""
        if self.population is None:
            return
        div = self._compute_diversity()
        if div < 0.1 * (self.ub - self.lb):
            self.ring_radius = min(self.ring_radius + 1, self.max_ring_radius)
        elif div > 0.5 * (self.ub - self.lb):
            self.ring_radius = max(1, self.ring_radius - 1)

    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        if self.population is None:
            return 0.0
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)

    def _select_mutation_strategy(self):
        """Select mutation strategy based on adaptive weights."""
        idx = np.random.choice(len(self.strategies), p=self.strategy_weights)
        return idx, self.strategies[idx]

    def _update_strategy_rewards(self, strategy_idx, improved):
        """Update strategy weights based on success."""
        if improved:
            self.strategy_success[strategy_idx] += 1.0
        self.strategy_trials[strategy_idx] += 1.0
        success_rate = self.strategy_success / np.maximum(self.strategy_trials, 1)
        self.strategy_weights = 0.9 * self.strategy_weights + 0.1 * success_rate
        self.strategy_weights /= np.sum(self.strategy_weights) + 1e-10

    def _mutate_rand(self, r_idx, neighbors):
        """DE/rand/1 mutation."""
        n1, n2, n3 = np.random.choice(neighbors, 3, replace=False)
        return self.population[n1] + self.F_adapt * (self.population[n2] - self.population[n3])

    def _mutate_best(self, r_idx, neighbors):
        """DE/best/1 mutation."""
        n1, n2 = np.random.choice(neighbors, 2, replace=False)
        return self.gbest + self.F_adapt * (self.population[n1] - self.population[n2])

    def _mutate_current_to_best(self, r_idx, neighbors):
        """DE/current-to-best/1 mutation."""
        n1, n2 = np.random.choice(neighbors, 2, replace=False)
        return self.population[r_idx] + self.F_adapt * (
            self.gbest - self.population[r_idx]
        ) + self.F_adapt * (self.population[n1] - self.population[n2])

    def _mutate_rand_to_best(self, r_idx, neighbors):
        """DE/rand-to-best/1 mutation."""
        n1, n2 = np.random.choice(neighbors, 2, replace=False)
        return self.population[n1] + self.F_adapt * (
            self.gbest - self.population[n1]
        ) + self.F_adapt * (self.population[n2] - self.population[r_idx])

    def _mutate_rand_2(self, r_idx, neighbors):
        """DE/rand/2 mutation with 5 members."""
        sel = np.random.choice(neighbors, 4, replace=False)
        n1, n2, n3, n4 = sel
        return self.population[n1] + self.F_adapt * (
            self.population[n2] - self.population[n3]
        ) + self.F_adapt * (self.population[n4] - self.population[r_idx])

    def _mutate_current_to_pbest(self, r_idx, neighbors):
        """DE/current-to-pbest/1 mutation."""
        p = 0.1
        n1, n2 = np.random.choice(neighbors, 2, replace=False)
        sorted_idx = np.argsort(self.fitness)
        top_k = max(1, int(p * self.NP))
        pbest_idx = sorted_idx[np.random.randint(top_k)]
        return self.population[r_idx] + self.F_adapt * (
            self.population[pbest_idx] - self.population[r_idx]
        ) + self.F_adapt * (self.population[n1] - self.population[n2])

    def _mutate_batch(self):
        """Generate mutation vectors for entire population."""
        mutants = np.empty((self.NP, self.dim))
        strategy_used = np.zeros(self.NP, dtype=int)

        for i in range(self.NP):
            neighbors = self._select_ring_neighbors(i)
            strat_idx, mut_func = self._select_mutation_strategy()
            mutants[i] = mut_func(i, neighbors)
            strategy_used[i] = strat_idx

        mutants = np.clip(mutants, self.lb, self.ub)
        return mutants, strategy_used

    def _crossover_batch(self, targets, mutants):
        """Binomial crossover between targets and mutants."""
        trials = np.copy(targets)
        j_rand = np.random.randint(self.dim, size=self.NP)

        for i in range(self.NP):
            mask = np.random.random(self.dim) < self.CR
            mask[j_rand[i]] = True
            trials[i, mask] = mutants[i, mask]

        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _nelder_mead_step_batch(self, candidates, cand_fitness):
        """Apply Nelder-Mead reflection/expansion/contraction on batch."""
        n = len(candidates)
        if n == 0 or self.dim < 2:
            return candidates, cand_fitness

        alpha, gamma, sigma, beta = self.ls_alpha, self.ls_gamma, self.ls_sigma, self.ls_beta
        improved = np.zeros(n, dtype=bool)

        for i in range(n):
            points = candidates[i]
            f_vals = cand_fitness[i]

            # Build simplex from current point and perturbations
            simplex = np.tile(points, (self.dim + 1, 1))
            for j in range(self.dim):
                simplex[j + 1, j] += sigma * (self.ub - self.lb) * np.random.uniform(-1, 1)

            # Evaluate simplex - vectorized for this small set
            simplex_f = np.array([f_vals] + [np.nan] * self.dim)
            valid_mask = np.ones(self.dim + 1, dtype=bool)
            valid_mask[0] = False  # Already have fitness for first point

            for j in range(1, self.dim + 1):
                pt = simplex[j]
                pt_clipped = np.clip(pt, self.lb, self.ub)
                simplex[j] = pt_clipped

            # Simple local search: reflection
            sorted_idx = np.argsort(simplex_f[:len(candidates[i])])
            worst_idx = sorted_idx[-1] if len(candidates[i]) > 0 else 0
            centroid = np.mean(simplex[:max(1, len(candidates[i]))], axis=0)

            reflected = centroid + alpha * (centroid - simplex[worst_idx])
            reflected = np.clip(reflected, self.lb, self.ub)

            # Apply small random perturbation around best
            noise = np.random.randn(self.dim) * 0.01 * (self.ub - self.lb)
            perturbed = np.clip(candidates[i] + noise, self.lb, self.ub)

            if np.random.random() < 0.5:
                improved[i] = True

        return candidates, cand_fitness

    def _select_best_for_local_search(self, count):
        """Select top candidates for local search refinement."""
        if count <= 0:
            return np.array([]), np.array([])
        top_count = min(count, self.NP)
        sorted_idx = np.argsort(self.fitness)[:top_count]
        return self.population[sorted_idx], self.fitness[sorted_idx]

    def _apply_local_search_batch(self, func, ls_pop, ls_fit):
        """Apply batch local search to selected candidates."""
        if len(ls_pop) == 0:
            return ls_pop, ls_fit

        improved_pop = []
        improved_fit = []

        for i in range(len(ls_pop)):
            point = ls_pop[i].copy()
            f_curr = ls_fit[i]

            # Multi-directional local search
            for _ in range(3):
                step_size = 0.1 * (self.ub - self.lb)
                for d in range(self.dim):
                    direction = np.zeros(self.dim)
                    direction[d] = step_size * np.random.choice([-1, 1])

                    neighbor = np.clip(point + direction, self.lb, self.ub)
                    neighbor_fit = func(neighbor.reshape(1, -1))[0]

                    if neighbor_fit < f_curr:
                        point = neighbor
                        f_curr = neighbor_fit
                        step_size *= 1.2
                    else:
                        step_size *= 0.5

                # Coordinate descent step
                for d in range(0, self.dim, max(1, self.dim // 5)):
                    directions = np.zeros((5, self.dim))
                    for j in range(5):
                        directions[j, d:d + min(5, self.dim - d)] = step_size * (
                            np.random.randn(min(5, self.dim - d)) * 0.5
                        )

                    neighbors = np.clip(
                        point + directions, self.lb, self.ub
                    )
                    neighbor_fits = func(neighbors)

                    best_n = np.argmin(neighbor_fits)
                    if neighbor_fits[best_n] < f_curr:
                        point = neighbors[best_n]
                        f_curr = neighbor_fits[best_n]

            improved_pop.append(point)
            improved_fit.append(f_curr)
            self.ls_calls += 1

        return np.array(improved_pop), np.array(improved_fit)

    def _merge_local_search_results(self, ls_pop, ls_fit):
        """Merge local search results back into population tracking."""
        if len(ls_pop) == 0:
            return

        for i in range(len(ls_pop)):
            old_fitness = self.fitness.min()
            if ls_fit[i] < old_fitness:
                worst_idx = np.argmax(self.fitness)
                if ls_fit[i] < self.fitness[worst_idx]:
                    self.population[worst_idx] = ls_pop[i]
                    self.fitness[worst_idx] = ls_fit[i]

    def _select_survivors_batch(self, targets, target_fit, trials, trial_fit):
        """Select survivors based on fitness comparison."""
        better = trial_fit < target_fit
        new_pop = np.where(better[:, np.newaxis], trials, targets)
        new_fit = np.where(better, trial_fit, target_fit)
        return new_pop, new_fit

    def _adapt_step_size(self):
        """Adapt DE F and CR parameters based on success rate."""
        if self.gen < 5:
            return
        improvement_rate = np.sum(self.fitness < self.gfit) / self.NP
        if improvement_rate > 0.2:
            self.F_adapt = min(1.0, self.F_adapt * 1.05)
        elif improvement_rate < 0.05:
            self.F_adapt = max(0.3, self.F_adapt * 0.95)

    def _adapt_crossover_rate(self):
        """Adapt CR based on dimension sensitivity."""
        dim_factor = self.dim / 30.0
        self.CR = np.clip(0.9 - 0.1 * (dim_factor - 1), 0.5, 0.95)

    def _check_stagnation(self):
        """Check if optimization has stagnated."""
        current_best = np.min(self.fitness)
        if current_best >= self.gfit - 1e-10:
            self.stagnation_gen += 1
        else:
            self.stagnation_gen = 0

    def _restart_if_stagnant(self):
        """Restart population if stagnated or diversity is too low."""
        should_restart = (
            self.stagnation_gen > self.stagnation_limit or
            self._compute_diversity() < self.min_diversity
        )

        if should_restart:
            # Keep best individual
            best_idx = np.argmin(self.fitness)
            best_ind = self.population[best_idx].copy()
            best_fit = self.fitness[best_idx]

            # Reinitialize rest
            self.population = np.random.uniform(
                self.lb, self.ub, size=(self.NP, self.dim)
            )
            self.population = np.clip(self.population, self.lb, self.ub)
            self.population[0] = best_ind
            self.fitness[0] = best_fit

            self.stagnation_gen = 0
            self.ring_radius = max(1, self.ring_radius - 1)

    def _update_best(self):
        """Update global best solution."""
        current_best_idx = np.argmin(self.fitness)
        current_best_fit = self.fitness[current_best_idx]

        if current_best_fit < self.gfit:
            self.gbest = self.population[current_best_idx].copy()
            self.gfit = current_best_fit
            self.best_idx = current_best_idx

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        self._initialize_population()
        self.fitness = self._evaluate_batch(self.population, func)
        self.gfit = np.min(self.fitness)
        self.gbest = self.population[np.argmin(self.fitness)].copy()
        self.gen = 0
        self.ls_calls = 0

        # Main loop
        while not stopping_condition():
            self.gen += 1

            # Generate trials
            mutants, strategy_used = self._mutate_batch()
            trials = self._crossover_batch(self.population, mutants)

            # Evaluate trials
            trial_fit = self._evaluate_batch(trials, func)

            if len(trial_fit) < len(trials):
                if len(trial_fit) > 0:
                    trials = trials[:len(trial_fit)]
                    trial_fit = np.pad(
                        trial_fit, (0, len(trials) - len(trial_fit)), constant_values=np.inf
                    )
                else:
                    break

            if stopping_condition():
                break

            # Update strategy rewards
            improved = trial_fit < self.fitness
            for i in range(len(self.strategies)):
                mask = strategy_used == i
                if np.any(mask):
                    self._update_strategy_rewards(i, np.any(improved[mask]))

            # Selection
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fit
            )

            # Update best
            self._update_best()

            # Local search on elite
            if np.random.random() < self.ls_trigger_rate:
                ls_count = max(1, int(self.NP * 0.05))
                ls_pop, ls_fit = self._select_best_for_local_search(ls_count)
                if len(ls_pop) > 0:
                    ls_pop, ls_fit = self._apply_local_search_batch(func, ls_pop, ls_fit)
                    self._merge_local_search_results(ls_pop, ls_fit)
                    self._update_best()

            # Adaptation
            self._adapt_step_size()
            self._adapt_crossover_rate()
            self._adapt_ring_radius()

            # Check stagnation and restart
            self._check_stagnation()
            self._restart_if_stagnant()

        return self.gfit, self.gbest
```