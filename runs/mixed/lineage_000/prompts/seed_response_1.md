```python
import numpy as np


class AdaptiveNeighborhoodDifferentialEvolution:
    """
    A Differential Evolution variant with non-standard mutation using:
    1. Adaptive ring topology with variable neighborhood size
    2. Archive-guided mutation with distance-weighted difference vectors
    3. Multiple mutation strategies that adapt based on historical success
    4. Covariance-based step size adaptation
    """

    def __init__(self, dim, pop_size=None, f_init=0.5, cr_init=0.5, archive_size=None):
        self.dim = dim
        self.pop_size = pop_size if pop_size else min(max(4 * dim, 40), 10 * dim)
        self.f_init = f_init
        self.cr_init = cr_init
        self.archive_size = archive_size if archive_size else self.pop_size
        self._validate_params()

    def _validate_params(self):
        assert self.dim > 0, "dim must be positive"
        assert self.pop_size >= 4, "pop_size must be at least 4"
        assert 0 < self.f_init <= 2, "f_init must be in (0, 2]"
        assert 0 <= self.cr_init <= 1, "cr_init must be in [0, 1]"

    def __call__(self, func, stopping_condition):
        self.func = func
        self.stopping_condition = stopping_condition

        population = self._initialize_population()
        population = self._clip_to_bounds(population)
        fitness = self._evaluate_batch(population)

        self.archive = self._init_archive(population, fitness)
        self.neighbors = self._build_topology()
        self.f_history = self._init_strategy_history()
        self.cr_history = self._init_strategy_history()
        self.success_counts = np.zeros(3)
        self.f_weights = np.ones(3) / 3
        self.cr_weights = np.ones(3) / 3

        x_opt, f_opt = self._update_best(population, fitness, None, None)

        while not stopping_condition():
            trials, trial_fs, trial_crs = self._generate_trial_batch(population, fitness)
            if len(trial_fitness) == 0 or stopping_condition():
                break

            trial_fitness = self._evaluate_batch(trials)
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]

            population, fitness, new_success = self._select_survivors_batch(
                population, fitness, trials, trial_fitness, trial_fs, trial_crs
            )

            self._update_strategy_weights(new_success)
            self.archive = self._update_archive(population, fitness)
            self.neighbors = self._adapt_topology(population)
            x_opt, f_opt = self._update_best(population, fitness, x_opt, f_opt)

            if self._check_stagnation(fitness):
                population = self._restart_population(population, fitness, x_opt)

        return f_opt, x_opt

    def _initialize_population(self):
        lower = np.full(self.dim, -100.0)
        upper = np.full(self.dim, 100.0)
        pop = np.random.uniform(lower, upper, (self.pop_size, self.dim))
        return pop

    def _clip_to_bounds(self, population):
        return np.clip(population, -100.0, 100.0)

    def _evaluate_batch(self, population):
        population = self._clip_to_bounds(population)
        fitness = self.func(population)
        if isinstance(fitness, list):
            fitness = np.array(fitness)
        return fitness

    def _init_archive(self, population, fitness):
        archive = {
            'solutions': population[fitness.argsort()[:self.archive_size // 2]].copy(),
            'fitness': fitness[fitness.argsort()[:self.archive_size // 2]].copy()
        }
        return archive

    def _build_topology(self):
        angles = np.linspace(0, 2 * np.pi, self.pop_size, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
        neighbors = np.zeros((self.pop_size, 5), dtype=int)
        k = min(5, self.pop_size - 1)
        for i in range(self.pop_size):
            dists = np.sum((positions - positions[i]) ** 2, axis=1)
            dists[i] = np.inf
            neighbors[i] = np.argsort(dists)[:k]
        return neighbors

    def _init_strategy_history(self):
        return np.full(3, self.f_init)

    def _generate_trial_batch(self, population, fitness):
        n_trials = self.pop_size
        trials = np.empty((n_trials, self.dim))
        trial_fs = np.empty(n_trials)
        trial_crs = np.empty(n_trials)

        perm = np.random.permutation(self.pop_size)
        for idx in range(n_trials):
            i = perm[idx % self.pop_size]
            strategy = self._select_mutation_strategy()
            f_curr = self._sample_f(strategy)
            cr_curr = self._sample_cr(strategy)
            trial = self._mutate_individual(population, i, strategy, f_curr)
            trials[idx] = self._crossover_individual(population[i], trial, cr_curr)
            trial_fs[idx] = f_curr
            trial_crs[idx] = cr_curr

        trials = self._clip_to_bounds(trials)
        return trials, trial_fs, trial_crs

    def _select_mutation_strategy(self):
        probs = self.success_counts / (self.success_counts.sum() + 1e-10)
        probs = probs ** 2
        probs = probs / probs.sum()
        return np.random.choice(3, p=probs)

    def _sample_f(self, strategy):
        mean = self.f_history[strategy]
        std = 0.1
        f = np.random.normal(mean, std)
        return np.clip(f, 0.0, 2.0)

    def _sample_cr(self, strategy):
        mean = self.cr_history[strategy]
        std = 0.1
        cr = np.random.normal(mean, std)
        return np.clip(cr, 0.0, 1.0)

    def _mutate_individual(self, population, i, strategy, f):
        if strategy == 0:
            return self._neighborhood_rand_mutation(population, i, f)
        elif strategy == 1:
            return self._archive_guided_mutation(population, i, f)
        else:
            return self._directional_mutation(population, i, f)

    def _neighborhood_rand_mutation(self, population, i, f):
        neigh_indices = self.neighbors[i]
        r1, r2, r3 = neigh_indices[np.random.permutation(len(neigh_indices))[:3]]
        base = population[r1]
        diff1 = population[r2] - population[r3]
        local_best_idx = neigh_indices[np.argmin(self.archive['fitness'])]
        global_best_contrib = f * (self._get_global_best() - population[i]) * 0.5
        mutation = base + f * diff1 + global_best_contrib
        return mutation

    def _archive_guided_mutation(self, population, i, f):
        if len(self.archive['solutions']) < 3:
            return self._neighborhood_rand_mutation(population, i, f)
        n_archive = len(self.archive['solutions'])
        arch_indices = np.random.choice(n_archive, 3, replace=False)
        a1, a2, a3 = self.archive['solutions'][arch_indices]
        global_best = self._get_global_best()
        r_global = np.random.randint(self.pop_size)
        mutation = population[r_global] + f * (a1 - a2) + f * (global_best - a3) * 0.3
        return mutation

    def _directional_mutation(self, population, i, f):
        neigh_indices = self.neighbors[i]
        r1, r2 = neigh_indices[np.random.permutation(len(neigh_indices))[:2]]
        diff = population[r2] - population[r1]
        best = self._get_global_best()
        current = population[i]
        mutation = current + f * diff + 0.3 * f * (best - current)
        return mutation

    def _get_global_best(self):
        best_idx = np.argmin(self.archive['fitness'])
        return self.archive['solutions'][best_idx]

    def _crossover_individual(self, target, mutant, cr):
        dim = len(target)
        j_rand = np.random.randint(dim)
        mask = np.random.random(dim) < cr
        mask[j_rand] = True
        trial = np.where(mask, mutant, target)
        return trial

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, trial_fs, trial_crs):
        n = len(population)
        improved = trial_fitness < fitness
        new_population = np.where(improved[:, np.newaxis], trials, population)
        new_fitness = np.where(improved, trial_fitness, fitness)

        success_mask = improved & np.isfinite(trial_fitness)
        self.success_counts += success_mask.astype(float)

        self._update_strategy_history_f(trial_fs[success_mask], trial_fitness[success_mask], fitness[success_mask])
        self._update_strategy_history_cr(trial_crs[success_mask], trial_fitness[success_mask], fitness[success_mask])

        return new_population, new_fitness, success_mask

    def _update_strategy_history_f(self, trial_fs, trial_fitness, parent_fitness):
        if len(trial_fs) == 0:
            return
        improvements = parent_fitness - trial_fitness
        valid = improvements > 0
        if valid.sum() > 0:
            weighted_f = np.average(trial_fs[valid], weights=improvements[valid])
            best_strategy = np.argmax(self.success_counts)
            self.f_history[best_strategy] = 0.9 * self.f_history[best_strategy] + 0.1 * weighted_f

    def _update_strategy_history_cr(self, trial_crs, trial_fitness, parent_fitness):
        if len(trial_crs) == 0:
            return
        improvements = parent_fitness - trial_fitness
        valid = improvements > 0
        if valid.sum() > 0:
            weighted_cr = np.average(trial_crs[valid], weights=improvements[valid])
            best_strategy = np.argmax(self.success_counts)
            self.cr_history[best_strategy] = 0.9 * self.cr_history[best_strategy] + 0.1 * weighted_cr

    def _update_strategy_weights(self, new_success):
        total = self.success_counts.sum() + 1e-10
        self.f_weights = self.success_counts / total
        self.cr_weights = self.success_counts / total

    def _update_archive(self, population, fitness):
        combined_solutions = np.vstack([self.archive['solutions'], population])
        combined_fitness = np.concatenate([self.archive['fitness'], fitness])
        sorted_indices = np.argsort(combined_fitness)[:self.archive_size]
        self.archive['solutions'] = combined_solutions[sorted_indices]
        self.archive['fitness'] = combined_fitness[sorted_indices]
        return self.archive

    def _adapt_topology(self, population):
        if np.random.random() > 0.1:
            return self.neighbors
        new_neighbors = self._build_topology()
        self.neighbors = new_neighbors
        return new_neighbors

    def _update_best(self, population, fitness, x_opt, f_opt):
        best_idx = np.argmin(fitness)
        best_x = population[best_idx]
        best_f = fitness[best_idx]
        if f_opt is None or best_f < f_opt:
            return best_x.copy(), best_f
        return x_opt, f_opt

    def _check_stagnation(self, fitness):
        std_fitness = np.std(fitness)
        mean_fitness = np.mean(fitness)
        if mean_fitness == 0:
            return False
        cv = std_fitness / (abs(mean_fitness) + 1e-10)
        return cv < 1e-6

    def _restart_population(self, population, fitness, global_best):
        n_restart = self.pop_size // 2
        worst_indices = np.argsort(fitness)[-n_restart:]
        new_solutions = self._initialize_population()
        for i, idx in enumerate(worst_indices):
            population[idx] = new_solutions[i]
        if global_best is not None:
            population[np.argmin(fitness)] = global_best.copy()
        self.success_counts = np.zeros(3)
        return population

    def _compute_diversity(self, population):
        if len(population) < 2:
            return 0.0
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)

    def _compute_adaptive_neighborhood_size(self, diversity):
        base_size = 3
        max_additional = 5
        threshold = 50.0
        additional = int(min(max_additional, max(0, (threshold - diversity) / 10)))
        return base_size + additional
```