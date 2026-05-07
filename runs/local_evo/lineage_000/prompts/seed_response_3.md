```python
import numpy as np


class TopologyAdaptiveParticleSwarmOptimizer:
    """
    A PSO variant with ring topology, adaptive inertia, diversity-guided restart,
    and levy-perturbed velocity for exploration.
    """

    def __init__(self, dim=30, pop_size=40, bounds=(-100.0, 100.0)):
        self.dim = dim
        self.pop_size = max(4, int(pop_size))
        self.bounds = bounds
        self.lb = np.full(self.dim, bounds[0])
        self.ub = np.full(self.dim, bounds[1])

        # Swarm state
        self.population = None
        self.velocity = None
        self.fitness = None
        self.personal_best_pos = None
        self.personal_best_val = None
        self.local_best_pos = None
        self.local_best_val = None
        self.global_best_pos = None
        self.global_best_val = np.inf

        # Adaptive parameters
        self.inertia = 0.729
        self.inertia_min = 0.3
        self.inertia_max = 0.9
        self.cognitive = 1.49445
        self.social = 1.49445

        # Topology
        self.neighbor_size = 3
        self.topology_indices = None

        # Diversity tracking
        self.diversity_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 50

        # Archive
        self.archive = None
        self.archive_size = self.pop_size * 2

    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self._initialize_topology()
        self._initialize_archives()

        while not stopping_condition():
            self._clip_population()
            fitness = self._evaluate(func)

            self._update_personal_best(fitness)
            self._update_local_best()
            self._update_global_best(fitness)

            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._compute_and_store_diversity()

            self._update_velocity()
            self._update_positions()

            self._manage_archive(fitness)
            self._restart_if_stagnant(func, stopping_condition)

        return self.global_best_val, self.global_best_pos.copy()

    # === Initialization Methods ===

    def _initialize_population(self):
        self.population = np.random.uniform(
            self.lb, self.ub, size=(self.pop_size, self.dim)
        )
        self.velocity = np.random.uniform(
            -np.abs(self.ub - self.lb) * 0.1,
            np.abs(self.ub - self.lb) * 0.1,
            size=(self.pop_size, self.dim)
        )
        self.personal_best_pos = self.population.copy()
        self.personal_best_val = np.full(self.pop_size, np.inf)

    def _initialize_topology(self):
        self._build_ring_topology()

    def _build_ring_topology(self):
        indices = []
        half = self.neighbor_size // 2
        for i in range(self.pop_size):
            neighbors = []
            for offset in range(-half, half + 1):
                idx = (i + offset) % self.pop_size
                if idx != i:
                    neighbors.append(idx)
            indices.append(neighbors)
        self.topology_indices = indices

    def _initialize_archives(self):
        self.archive = np.zeros((0, self.dim))
        self.global_best_val = np.inf
        self.global_best_pos = np.zeros(self.dim)

    # === Evaluation & Clipping ===

    def _clip_population(self):
        self.population = np.clip(self.population, self.lb, self.ub)

    def _evaluate(self, func):
        self.fitness = func(self.population)
        self.fitness = np.where(np.isfinite(self.fitness), self.fitness, np.inf)
        return self.fitness

    # === Best Updates ===

    def _update_personal_best(self, fitness):
        improved = fitness < self.personal_best_val
        self.personal_best_val[improved] = fitness[improved]
        self.personal_best_pos[improved] = self.population[improved].copy()

    def _update_local_best(self):
        self.local_best_pos = np.zeros((self.pop_size, self.dim))
        self.local_best_val = np.full(self.pop_size, np.inf)

        for i in range(self.pop_size):
            neighbors = self.topology_indices[i]
            neighbor_fitness = self.personal_best_val[neighbors]
            best_neighbor_idx = neighbors[np.argmin(neighbor_fitness)]
            self.local_best_pos[i] = self.personal_best_pos[best_neighbor_idx].copy()
            self.local_best_val[i] = self.personal_best_val[best_neighbor_idx]

    def _update_global_best(self, fitness):
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.global_best_val:
            self.global_best_val = fitness[best_idx]
            self.global_best_pos = self.population[best_idx].copy()

    # === Adaptive Parameters ===

    def _adapt_inertia_weight(self):
        if len(self.diversity_history) < 5:
            return
        recent_div = np.mean(self.diversity_history[-5:])
        div_normalized = (recent_div - self._compute_diversity()) / (recent_div + 1e-10)
        self.inertia = np.clip(
            self.inertia + 0.1 * div_normalized,
            self.inertia_min,
            self.inertia_max
        )

    def _adapt_neighborhood_size(self):
        current_div = self._compute_diversity()
        if current_div < 0.1 * np.sqrt(self.dim):
            self.neighbor_size = min(self.neighbor_size + 1, self.pop_size - 1)
            self._build_ring_topology()
        elif current_div > 1.0 * np.sqrt(self.dim) and self.neighbor_size > 2:
            self.neighbor_size = max(self.neighbor_size - 1, 2)
            self._build_ring_topology()

    # === Diversity & Restart ===

    def _compute_diversity(self):
        if self.population is None or len(self.population) < 2:
            return 0.0
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)

    def _compute_and_store_diversity(self):
        div = self._compute_diversity()
        self.diversity_history.append(div)
        if len(self.diversity_history) > 100:
            self.diversity_history.pop(0)

    def _restart_if_stagnant(self, func, stopping_condition):
        if len(self.diversity_history) < 10:
            return

        recent_best_improved = self.global_best_val < np.inf
        if recent_best_improved:
            self.stagnation_counter = 0
            return

        self.stagnation_counter += 1

        if self.stagnation_counter >= self.stagnation_threshold:
            self._reinitialize_portion(0.5)
            self.stagnation_counter = 0
            self.diversity_history.clear()

    def _reinitialize_portion(self, fraction):
        count = max(1, int(fraction * self.pop_size))
        indices = np.random.choice(self.pop_size, count, replace=False)
        for idx in indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub)
            self.velocity[idx] = np.random.uniform(
                -np.abs(self.ub - self.lb) * 0.1,
                np.abs(self.ub - self.lb) * 0.1
            )
            self.personal_best_pos[idx] = self.population[idx].copy()
            self.personal_best_val[idx] = np.inf

    # === Archive Management ===

    def _manage_archive(self, fitness):
        good_indices = np.where(fitness < np.percentile(fitness, 50))[0]
        new_solutions = self.population[good_indices]

        self.archive = np.vstack([self.archive, new_solutions])
        if len(self.archive) > self.archive_size:
            keep_indices = np.random.choice(
                len(self.archive), self.archive_size, replace=False
            )
            self.archive = self.archive[keep_indices]

    # === Movement ===

    def _update_velocity(self):
        r1 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))
        r3 = np.random.uniform(0, 1, size=(self.pop_size, self.dim))

        cognitive_component = self.cognitive * r1 * (
            self.personal_best_pos - self.population
        )
        social_component = self.social * r2 * (
            self.local_best_pos - self.population
        )

        levy_perturbation = self._generate_levy_perturbation()

        self.velocity = (
            self.inertia * self.velocity
            + cognitive_component
            + social_component
            + 0.1 * r3 * levy_perturbation
        )

        max_velocity = 0.2 * np.abs(self.ub - self.lb)
        vel_magnitude = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        scale = np.clip(vel_magnitude / (max_velocity + 1e-10), 0, 1)
        self.velocity = self.velocity / (scale + 1e-10) * np.minimum(scale, 1)

    def _generate_levy_perturbation(self):
        beta = 1.5
        sigma = (
            (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2))
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        step = np.random.normal(0, sigma, size=(self.pop_size, self.dim))
        step_magnitude = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / (step_magnitude + 1e-10)

        length = np.random.uniform(0.01, 0.5, size=(self.pop_size, 1))
        levy = step * length

        return levy

    def _update_positions(self):
        self.population = self.population + self.velocity
        self.population = np.clip(self.population, self.lb, self.ub)
```