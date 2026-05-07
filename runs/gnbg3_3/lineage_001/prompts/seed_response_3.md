```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Particle Swarm Optimizer with adaptive neighborhood topology and
    multi-role sub-swarm cooperation. Uses ring topology with dynamic
    rewiring based on swarm diversity.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 10 * dim)
        self.bounds = np.array([-100.0, 100.0])
        self.max_velocity = 0.2 * (self.bounds[1] - self.bounds[0])

        self.w_max = 0.9
        self.w_min = 0.1
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.chi = 0.7298

        self.diversity_trigger = 0.1
        self.topology_refresh_interval = 50
        self.scout_fraction = 0.1

        self._topology = None
        self._generation = 0
        self._stagnation_counter = 0
        self._best_history = []

    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        personal_best_positions, personal_best_fitness = self._initialize_personal_best(population, func)
        global_best_idx, global_best_pos, global_best_fit = self._compute_global_best(
            personal_best_positions, personal_best_fitness
        )

        while not stopping_condition():
            self._generation += 1

            inertia_weight = self._adapt_inertia_weight(global_best_fit, personal_best_fitness)

            self._update_topology_if_needed(population)

            velocity = self._compute_velocity_batch(population, personal_best_positions,
                                                     global_best_pos, inertia_weight)

            velocity = self._clip_velocity_batch(velocity)

            population = self._update_position_batch(population, velocity)

            population = self._clip_population_batch(population)

            fitness = func(population)

            if len(fitness) < self.np:
                population = population[:len(fitness)]
                velocity = velocity[:len(fitness)]
                personal_best_positions = personal_best_positions[:len(fitness)]
                personal_best_fitness = personal_best_fitness[:len(fitness)]

            if stopping_condition():
                break

            personal_best_positions, personal_best_fitness = self._update_personal_best_batch(
                population, fitness, personal_best_positions, personal_best_fitness
            )

            g_idx, g_pos, g_fit = self._compute_global_best(personal_best_positions, personal_best_fitness)

            if g_fit < global_best_fit:
                global_best_idx, global_best_pos, global_best_fit = g_idx, g_pos, g_fit
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1

            self._inject_scout_particles(population, fitness, personal_best_positions)

            self._best_history.append(global_best_fit)

        return global_best_fit, global_best_pos

    def _initialize_population(self):
        lower, upper = self.bounds[0], self.bounds[1]
        population = np.random.uniform(lower, upper, (self.np, self.dim))
        return population

    def _initialize_personal_best(self, population, func):
        fitness = func(population)
        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()
        return personal_best_positions, personal_best_fitness

    def _initialize_topology(self):
        np_local = self.np
        neighborhood_size = 3
        topology = np.zeros((np_local, neighborhood_size), dtype=np.intp)
        for i in range(np_local):
            indices = [(i - 1) % np_local, i, (i + 1) % np_local]
            topology[i] = indices
        return topology

    def _compute_global_best(self, pbest_positions, pbest_fitness):
        best_idx = np.argmin(pbest_fitness)
        return best_idx, pbest_positions[best_idx], pbest_fitness[best_idx]

    def _adapt_inertia_weight(self, global_best, personal_best_fitness):
        current_best = np.min(personal_best_fitness)
        improvement_ratio = (global_best - current_best) / (abs(global_best) + 1e-10)

        if improvement_ratio > 1e-6:
            self._stagnation_counter = 0
            w = self.w_max - (self.w_max - self.w_min) * (self._generation / 1000)
        else:
            w = self.w_min + 0.2 * np.exp(-self._stagnation_counter / 20)

        diversity = self._compute_diversity(personal_best_fitness)
        if diversity < self.diversity_trigger:
            w = min(w * 1.2, self.w_max)

        return np.clip(w, self.w_min, self.w_max)

    def _compute_diversity(self, fitness):
        if len(fitness) < 2:
            return 1.0
        normalized = (fitness - np.min(fitness)) / (np.ptp(fitness) + 1e-10)
        return 1.0 - np.mean(np.abs(np.sort(normalized) - np.linspace(0, 1, len(normalized))))

    def _update_topology_if_needed(self, population):
        if self._topology is None:
            self._topology = self._initialize_topology()
            return

        if self._generation % self.topology_refresh_interval == 0:
            self._topology = self._adapt_topology_batch(population)

    def _adapt_topology_batch(self, population):
        np_local = self.np
        neighborhood_size = 3
        topology = np.zeros((np_local, neighborhood_size), dtype=np.intp)

        distances = np.linalg.norm(population[:, np.newaxis, :] - population[np.newaxis, :, :], axis=2)
        sorted_indices = np.argsort(distances, axis=1)

        for i in range(np_local):
            neighbors = sorted_indices[i, 1:neighborhood_size + 1]
            if len(neighbors) < neighborhood_size:
                neighbors = np.append(neighbors, [(i + j) % np_local for j in range(1, neighborhood_size - len(neighbors) + 1)])
            topology[i] = neighbors[:neighborhood_size]

        return topology

    def _compute_velocity_batch(self, population, pbest, gbest, inertia_weight):
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))

        cognitive = self.c1 * r1 * (pbest - population)
        social = self.c2 * r2 * (gbest - population)

        velocity = inertia_weight * self.chi * population.shape[0] * 0.01 + cognitive + social

        return velocity

    def _clip_velocity_batch(self, velocity):
        return np.clip(velocity, -self.max_velocity, self.max_velocity)

    def _update_position_batch(self, population, velocity):
        return population + velocity

    def _clip_population_batch(self, population):
        return np.clip(population, self.bounds[0], self.bounds[1])

    def _update_personal_best_batch(self, population, fitness, pbest_pos, pbest_fit):
        improved_mask = fitness < pbest_fit
        pbest_fit = np.where(improved_mask, fitness, pbest_fit)
        pbest_pos = np.where(improved_mask[:, np.newaxis], population, pbest_pos)
        return pbest_pos, pbest_fit

    def _inject_scout_particles(self, population, fitness, pbest_positions):
        n_scouts = max(1, int(self.scout_fraction * self.np))
        worst_indices = np.argsort(fitness)[-n_scouts:]

        for idx in worst_indices:
            if self._stagnation_counter > 10:
                population[idx] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                pbest_positions[idx] = population[idx].copy()
            else:
                mutation_strength = 0.1 * (self.bounds[1] - self.bounds[0])
                mutation = np.random.normal(0, mutation_strength, self.dim)
                trial = population[idx] + mutation
                trial = np.clip(trial, self.bounds[0], self.bounds[1])
                population[idx] = trial
```