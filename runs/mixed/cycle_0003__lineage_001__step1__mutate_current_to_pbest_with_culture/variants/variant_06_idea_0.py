import numpy as np


class CulturalMemeticDE:
    """
    Hybrid Differential Evolution with Cultural Memory and Adaptive Local Search.
    
    Key innovations vs previous algorithms:
    - Cultural memory stores successful mutation patterns, guiding search in similar regions
    - Adaptive local search (pattern search) refines best individuals periodically
    - Per-dimension adaptation using fitness landscape statistics
    - Diversity-triggered reinitialization with elitist preservation
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(kwargs.get('NP', 8 * dim), 300)
        self.min_pop_size = 20
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        # Adaptive parameters
        self.F = 0.7
        self.CR = 0.5
        self.p_best_rate = 0.1

        # Cultural memory
        self.cultural_memory_size = min(30, self.NP)
        self.cultural_memory = np.zeros((self.cultural_memory_size, dim))
        self.cultural_weights = np.ones(self.cultural_memory_size)
        self.culture_idx = 0

        # Local search
        self.local_search_interval = kwargs.get('local_search_interval', 7)
        self.local_radius = kwargs.get('local_radius', 2.0)
        self.local_max_iter = 10

        # Diversity and stagnation
        self.stagnation_count = 0
        self.stagnation_limit = kwargs.get('stagnation_limit', 60)
        self.diversity_threshold = 0.1
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        self.improvement_buffer = []

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = population[best_idx].copy()

        while not stopping_condition():
            mutants = self._mutate_current_to_pbest_with_culture(population, fitness)
            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)

            best_idx = np.argmin(fitness)
            current_best = fitness[best_idx]
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

    def _initialize_population_lhs(self):
        samples = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
        for d in range(self.dim):
            edges = np.linspace(self.lower[d], self.upper[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = edges[perm] + np.random.uniform(0, edges[1] - edges[0], self.NP)
        return self._clip_to_bounds_batch(samples)

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lower, self.upper)

    def _eval_wrapper_batch(self, population, func):
        result = func(population)
        if len(result) < len(population):
            result = np.pad(result, (0, len(population) - len(result)), constant_values=np.inf)
        return result

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
            # Detect stagnation and low diversity to trigger exploration mode
            diversity = np.std(population, axis=0).mean()
            is_stagnant = self.stagnation_count > 10
            is_low_diversity = diversity < 5.0

            # Adaptive F: increases with generation for more exploration, boosted further when stuck
            base_F = self.F
            gen_factor = min(self.generation / 200.0, 0.5)
            if is_stagnant or is_low_diversity:
                exploration_F = min(base_F * (2.0 + gen_factor), 2.0)
            else:
                exploration_F = base_F * (1.0 + gen_factor)

            # Select top performers for elite-guided perturbation
            sorted_idx = np.argsort(fitness)
            n_best = max(1, int(self.NP * self.p_best_rate))
            elite_indices = sorted_idx[:n_best]

            # Weighted elite selection (fitter = higher selection probability)
            elite_weights = 1.0 / (fitness[elite_indices] - fitness[elite_indices].min() + 1e-10)
            elite_weights = elite_weights / elite_weights.sum()

            # Choose p_best from weighted elite (not uniform random)
            p_best_idx = elite_indices[np.random.choice(n_best, size=self.NP, p=elite_weights)]
            p_best = population[p_best_idx]

            # Generate diverse random indices (avoid self)
            idx_a = np.random.randint(0, self.NP, (self.NP, 3))
            idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                    (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
            idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                                   (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
            r1, r2 = idx_a[:, 1], idx_a[:, 2]

            # Core DE mutation with adaptive F
            base = population + exploration_F * (p_best - population) + exploration_F * (population[r1] - population[r2])

            # Elite perturbation: add directed perturbation toward best individuals
            if is_stagnant or is_low_diversity:
                # Find global best for directed escape
                global_best_idx = sorted_idx[0]
                global_best = population[global_best_idx]

                # Perturbation toward global best with random component
                perturb_strength = min(0.3 * exploration_F, 0.5)
                direction = global_best - population
                direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
                direction_norm = np.where(direction_norm < 1e-10, 1.0, direction_norm)
                direction_normalized = direction / direction_norm

                # Random jump vectors for escape
                random_vectors = np.random.randn(self.NP, self.dim)
                random_norm = np.linalg.norm(random_vectors, axis=1, keepdims=True)
                random_norm = np.where(random_norm < 1e-10, 1.0, random_norm)
                random_normalized = random_vectors / random_norm

                # Blend: 60% directed toward best, 40% random escape
                escape_vectors = 0.6 * direction_normalized + 0.4 * random_normalized
                escape_vectors = escape_vectors / (np.linalg.norm(escape_vectors, axis=1, keepdims=True) + 1e-10)

                # Apply perturbation scaled by distance to global best
                distances_to_best = np.linalg.norm(population - global_best, axis=1, keepdims=True)
                jump_magnitude = perturb_strength * (1.0 + distances_to_best / (np.linalg.norm(population, axis=1, keepdims=True) + 1e-10))

                elite_perturbation = population + jump_magnitude * escape_vectors
                elite_perturbation = np.clip(elite_perturbation, self.lower, self.upper)

                # Blend base mutation with elite perturbation
                blend_weight = min(0.4 + 0.3 * gen_factor, 0.7)
                donors = (1 - blend_weight) * base + blend_weight * elite_perturbation
            else:
                donors = base

            return donors

    def _crossover_binomial_batch(self, population, donors):
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        trials = np.where(mask, donors, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        new_population = np.copy(population)
        new_population[better] = trials[better]
        new_fitness = np.copy(fitness)
        new_fitness[better] = trial_fitness[better]
        return new_population, new_fitness

    def _update_cultural_memory_on_improvement(self, best_individual, best_mutant):
        if self.stagnation_count == 0:
            pattern = best_mutant - best_individual
            self.cultural_memory[self.culture_idx] = pattern
            self.cultural_weights[self.culture_idx] = 1.0 / (1.0 + self.stagnation_count)
            self.culture_idx = (self.culture_idx + 1) % self.cultural_memory_size
            decay = 0.95
            self.cultural_weights *= decay

    def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:self.NP // 2]
        improvements = fitness[top_half] - fitness[top_half]

        F_candidates = np.random.uniform(0.4, 1.0, size=5)
        self.F = np.mean(F_candidates)
        self.F = np.clip(self.F, 0.3, 1.5)

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = np.mean(CR_candidates)
        self.CR = np.clip(self.CR, 0.1, 0.95)

        dim_adapt = np.random.rand() < 0.2
        if dim_adapt:
            self.p_best_rate = np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3)

    def _apply_adaptive_local_search(self, func):
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        for _ in range(self.local_max_iter):
            if radius < 1e-6:
                break

            directions = np.random.randn(self.dim)
            directions = directions / (np.linalg.norm(directions) + 1e-10)
            candidates = np.clip(center + radius * directions, self.lower, self.upper)
            candidates = np.vstack([candidates, np.clip(center - radius * directions, self.lower, self.upper)])
            candidates = np.vstack([candidates, center])

            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = np.argmin(fit_candidates)
            best_local_fitness = fit_candidates[best_local_idx]

            if best_local_fitness < current_fitness:
                center = candidates[best_local_idx].copy()
                current_fitness = best_local_fitness
                radius = min(radius * 1.5, 10.0)
            else:
                radius *= 0.4

        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = center.copy()
            self.local_radius = min(radius * 1.2, 10.0)
        else:
            self.local_radius = max(radius * 0.8, 0.1)

    def _compute_diversity_batch(self, population):
        centroids = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroids, axis=1)
        return np.mean(distances)

    def _check_diversity_low(self, population):
        diversity = self._compute_diversity_batch(population)
        return (diversity < self.diversity_threshold or 
                self.stagnation_count > self.stagnation_limit)

    def _reinitialize_population(self, population, fitness):
        if self.best_solution is not None:
            best_idx = np.argmin(fitness)
            population[0] = self.best_solution.copy()
            fitness[0] = self.best_fitness

        n_random = max(self.min_pop_size, self.NP // 2)
        new_part = self._initialize_population_lhs()[:n_random]
        population[1:1+n_random] = new_part
        fitness[1:1+n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5
