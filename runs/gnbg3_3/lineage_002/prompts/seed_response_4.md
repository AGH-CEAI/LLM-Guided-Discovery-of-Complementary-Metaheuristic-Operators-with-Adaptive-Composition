```python
import numpy as np


class CulturalMemeticSwarmOptimizer:
    """
    Hybrid optimizer combining:
    1. Cultural algorithm belief-space guidance
    2. Particle swarm position/velocity updates with adaptive inertia
    3. Subpopulation clustering for diversity
    4. Periodic Nelder-Mead local search on elite individuals
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = max(5 * dim, 80)
        self.bounds = np.array([-100.0, 100.0])
        self._reset_state()

    def _reset_state(self):
        self.population = None
        self.velocity = None
        self.fitness = None
        self.best_x = None
        self.best_f = np.inf
        self.gbest_x = None
        self.gbest_f = np.inf
        self.gen = 0
        self.stagnation_count = 0
        self.local_search_interval = 15
        self.last_local_search_gen = -100

        # Belief space (cultural algorithm component)
        self.belief_mean = None
        self.belief_std = None
        self.belief_best = None

        # Adaptive PSO parameters
        self.inertia = 0.729
        self.cognitive = 1.494
        self.social = 1.494
        self.success_history = []

        # Subpopulation clustering
        self.labels = None
        self.n_clusters = max(3, self.np // 25)

    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        self._initialize_belief_space()
        self._evaluate_and_update_best()

        while not stopping_condition():
            self._update_clusters()
            self._update_velocity_batch()
            self._update_position_batch()
            self._clip_to_bounds_batch()
            self._evaluate_and_update_best()

            if stopping_condition():
                break

            self._select_survivors_batch()
            self._update_belief_space()

            if stopping_condition():
                break

            self._adapt_inertia_weight()
            self._adapt_cognitive_social()
            self._restart_if_stagnant()

            if self.gen - self.last_local_search_gen >= self.local_search_interval:
                self._apply_local_search_batch()
                self.last_local_search_gen = self.gen

            self.gen += 1

        return self.gbest_f, self.gbest_x.copy()

    def _initialize_population(self):
        low, high = self.bounds
        self.population = np.random.uniform(low, high, (self.np, self.dim))
        self.velocity = np.random.uniform(-0.1 * (high - low), 0.1 * (high - low), (self.np, self.dim))
        self.fitness = np.full(self.np, np.inf)

    def _initialize_belief_space(self):
        self.belief_mean = np.zeros(self.dim)
        self.belief_std = np.ones(self.dim) * 50.0
        self.belief_best = self.gbest_x.copy() if self.gbest_x is not None else np.zeros(self.dim)

    def _evaluate_and_update_best(self):
        valid_mask = np.ones(self.np, dtype=bool)
        batch = self.population
        fvals = self.func(batch)
        fvals = np.asarray(fvals)

        if len(fvals) < self.np:
            valid_mask[:len(fvals)] = True
            valid_mask[len(fvals):] = False
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        self.fitness[valid_mask] = fvals[valid_mask[:len(fvals)]]
        self.fitness[~valid_mask] = np.inf

        any_valid = np.any(valid_mask)
        if any_valid:
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.gbest_f:
                self.gbest_f = self.fitness[best_idx]
                self.gbest_x = self.population[best_idx].copy()
                self.best_f = self.fbest_f = self.gbest_f
                self.best_x = self.gbest_x.copy()

    def _update_clusters(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            self.labels = np.arange(self.np)
            return

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=3)
        self.labels = kmeans.fit_predict(self.population)

    def _update_velocity_batch(self):
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        cognitive = self.cognitive * r1 * (self.best_x - self.population)
        social = self.social * r2 * (self.gbest_x - self.population)

        cultural_bias = 0.15 * r1 * (self.belief_best - self.population)

        self.velocity = self.inertia * self.velocity + cognitive + social + cultural_bias

        max_vel = 0.2 * (self.bounds[1] - self.bounds[0])
        vel_mag = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_mag / max_vel, 1.0, None)
        self.velocity /= scale

    def _update_position_batch(self):
        self.population = self.population + self.velocity

    def _clip_to_bounds_batch(self):
        np.clip(self.population, self.bounds[0], self.bounds[1], out=self.population)

    def _select_survivors_batch(self):
        improved = self.fitness < self.best_f
        self.best_f = np.where(improved, self.fitness, self.best_f)
        self.best_x = np.where(improved[:, np.newaxis], self.population, self.best_x)

        self.success_history.append(np.mean(improved))
        if len(self.success_history) > 10:
            self.success_history.pop(0)

    def _update_belief_space(self):
        alpha = 0.1
        self.belief_mean = (1 - alpha) * self.belief_mean + alpha * np.mean(self.population, axis=0)
        self.belief_std = np.sqrt((1 - alpha) * self.belief_std ** 2 + alpha * np.var(self.population, axis=0) + 1e-8)

        if self.gbest_f < self.func(self.belief_best.reshape(1, -1))[0]:
            self.belief_best = self.gbest_x.copy()

    def _adapt_inertia_weight(self):
        if len(self.success_history) < 3:
            return

        recent_success = np.mean(self.success_history[-3:])
        if recent_success > 0.2:
            self.inertia = min(0.95, self.inertia * 1.05)
        elif recent_success < 0.05:
            self.inertia = max(0.3, self.inertia * 0.9)

    def _adapt_cognitive_social(self):
        if len(self.success_history) < 5:
            return

        diversity = self._compute_diversity()
        if diversity < 0.1:
            self.cognitive = min(2.5, self.cognitive * 1.1)
            self.social = min(2.5, self.social * 1.1)
        else:
            self.cognitive = max(0.5, self.cognitive * 0.95)
            self.social = max(0.5, self.social * 0.95)

    def _compute_diversity(self):
        if self.population is None or len(self.population) < 2:
            return 1.0

        centroid = np.mean(self.population, axis=0)
        avg_dist = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        spread = (self.bounds[1] - self.bounds[0]) / 2.0
        return np.clip(avg_dist / (spread + 1e-8), 0.0, 1.0)

    def _restart_if_stagnant(self):
        threshold = max(50, self.dim * 2)
        if self.stagnation_count > threshold:
            n_replace = self.np // 4
            replace_idx = np.random.choice(self.np, n_replace, replace=False)

            for idx in replace_idx:
                self.population[idx] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                self.velocity[idx] = np.random.uniform(-20, 20, self.dim)

            self.stagnation_count = 0
            self.inertia = 0.729
            self.cognitive = 1.494
            self.social = 1.494

    def _apply_local_search_batch(self):
        elite_count = max(2, self.np // 20)
        sorted_idx = np.argsort(self.fitness)[:elite_count]

        improved_positions = []
        improved_fitness = []

        for idx in sorted_idx:
            x = self.population[idx].copy()
            simplex = self._build_simplex(x)
            simplex_fit = self.func(simplex)
            best_local, best_local_fit = self._nelder_mead_iterate(simplex, simplex_fit)

            if best_local_fit < self.fitness[idx]:
                improved_positions.append(best_local)
                improved_fitness.append(best_local_fit)

        if improved_positions:
            n_improve = len(improved_positions)
            for i, idx in enumerate(sorted_idx[:n_improve]):
                self.population[idx] = improved_positions[i]
                self.fitness[idx] = improved_fitness[i]

            valid_improved = [f for f in improved_fitness if np.isfinite(f)]
            if valid_improved:
                best_new = min(valid_improved)
                if best_new < self.gbest_f:
                    best_new_idx = np.argmin(improved_fitness)
                    self.gbest_f = improved_fitness[best_new_idx]
                    self.gbest_x = improved_positions[best_new_idx].copy()

    def _build_simplex(self, center):
        low, high = self.bounds
        scale = 0.5
        simplex = [center.copy()]

        for _ in range(self.dim):
            direction = np.random.uniform(-1, 1, self.dim)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            point = center + scale * (high - low) * direction
            np.clip(point, low, high, out=point)
            simplex.append(point)

        return np.array(simplex)

    def _nelder_mead_iterate(self, simplex, simplex_fit, max_iter=30):
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        for _ in range(max_iter):
            sorted_idx = np.argsort(simplex_fit)
            simplex = simplex[sorted_idx]
            simplex_fit = simplex_fit[sorted_idx]

            centroid = np.mean(simplex[:-1], axis=0)

            worst = simplex[-1]
            reflected = centroid + alpha * (centroid - worst)
            np.clip(reflected, self.bounds[0], self.bounds[1], out=reflected)
            reflected_fit = self.func(reflected.reshape(1, -1))[0]

            if simplex_fit[0] <= reflected_fit < simplex_fit[-2]:
                simplex[-1] = reflected
                simplex_fit[-1] = reflected_fit
            elif reflected_fit < simplex_fit[0]:
                expanded = centroid + gamma * (reflected - centroid)
                np.clip(expanded, self.bounds[0], self.bounds[1], out=expanded)
                expanded_fit = self.func(expanded.reshape(1, -1))[0]

                if expanded_fit < reflected_fit:
                    simplex[-1] = expanded
                    simplex_fit[-1] = expanded_fit
                else:
                    simplex[-1] = reflected
                    simplex_fit[-1] = reflected_fit
            else:
                contracted = centroid + rho * (worst - centroid)
                np.clip(contracted, self.bounds[0], self.bounds[1], out=contracted)
                contracted_fit = self.func(contracted.reshape(1, -1))[0]

                if contracted_fit < simplex_fit[-1]:
                    simplex[-1] = contracted
                    simplex_fit[-1] = contracted_fit
                else:
                    shrunk = simplex[0] + sigma * (simplex[1:] - simplex[0])
                    for j in range(1, len(simplex)):
                        np.clip(shrunk[j - 1], self.bounds[0], self.bounds[1], out=shrunk[j - 1])
                    simplex[1:] = shrunk
                    simplex_fit[1:] = self.func(shrunk)

            if np.std(simplex_fit) < 1e-8:
                break

        best_idx = np.argmin(simplex_fit)
        return simplex[best_idx], simplex_fit[best_idx]
```