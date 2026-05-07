```python
import numpy as np


class AdaptiveMultiStrategyDE:
    """
    Adaptive Multi-Strategy Differential Evolution with ring topology,
    archive-based mutation, and adaptive strategy selection.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(8 * dim, 240)
        self.lb = -100.0
        self.ub = 100.0

        self._init_parameters()
        self._init_strategy_management()
        self._init_tracking_variables()

    def _init_parameters(self):
        self.F_base = 0.6
        self.CR_base = 0.85
        self.F = self.F_base
        self.CR = self.CR_base
        self.p_best_rate = 0.1
        self.archive_size_max = self.NP

    def _init_strategy_management(self):
        self.num_strategies = 6
        self.strategy_names = [
            'rand_ring_pbest', 'best_2dir', 'current_to_pbest_archive',
            'rand_2opt', 'best_ring', 'current_adaptive'
        ]
        self.strategy_scores = np.ones(self.num_strategies)
        self.strategy_success_counts = np.zeros(self.num_strategies)
        self.strategy_attempt_counts = np.zeros(self.num_strategies)

    def _init_tracking_variables(self):
        self.stagnation_counter = 0
        self.max_stagnation = 30
        self.min_improvement = 1e-10
        self.best_fitness_history = []
        self.improvement_history = []
        self.archive = []
        self.generation = 0

    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        best_idx = np.argmin(fitness)
        self.best_fitness_history.append(fitness[best_idx])

        while not stopping_condition():
            if len(fitness) == 0:
                break

            strategy_idx = self._select_strategy_batch()
            trials = self._mutate_batch(population, fitness, strategy_idx)
            trials = self._crossover_batch(population, trials)
            trials = self._clip_to_bounds_batch(trials)

            if stopping_condition():
                break

            trial_fitness = func(trials)

            if len(trial_fitness) < len(trials):
                valid = len(trial_fitness)
                trials = trials[:valid]
                trial_fitness = trial_fitness[:valid]
                if valid == 0:
                    break

            if stopping_condition():
                break

            population, fitness, improved = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )

            self._update_archive(population, fitness, trials, trial_fitness, improved)
            self._update_best_tracking(fitness)
            self._update_strategy_scores(strategy_idx, improved)
            self._adapt_parameters()
            self.generation += 1

            if self._check_stagnation():
                population, fitness = self._restart_if_stagnant(population, fitness, func)
                self._reset_stagnation()

        best_idx = np.argmin(fitness) if len(fitness) > 0 else 0
        return fitness[best_idx], population[best_idx].copy()

    def _initialize_population(self):
        return np.random.uniform(
            self.lb, self.ub, size=(self.NP, self.dim)
        )

    def _select_strategy_batch(self):
        probs = self.strategy_scores / self.strategy_scores.sum()
        strategies = np.random.choice(
            self.num_strategies, size=self.NP, p=probs
        )
        for s in range(self.num_strategies):
            self.strategy_attempt_counts[s] += np.sum(strategies == s)
        return strategies

    def _get_ring_topology_indices(self, indices, step=2):
        n = len(indices)
        ring_indices = np.zeros((n, step), dtype=int)
        for i in range(n):
            for j in range(step):
                ring_indices[i, j] = indices[(i + j + 1) % n]
        return ring_indices

    def _mutate_batch(self, population, fitness, strategy_indices):
        trials = np.empty_like(population)
        unique_strategies = np.unique(strategy_indices)

        for s in unique_strategies:
            mask = strategy_indices == s
            count = np.sum(mask)
            if count == 0:
                continue

            if s == 0:
                trials[mask] = self._mutate_rand_ring_pbest(population, fitness, count)
            elif s == 1:
                trials[mask] = self._mutate_best_2dir(population, count)
            elif s == 2:
                trials[mask] = self._mutate_current_to_pbest_archive(population, fitness, count)
            elif s == 3:
                trials[mask] = self._mutate_rand_2opt(population, count)
            elif s == 4:
                trials[mask] = self._mutate_best_ring(population, count)
            else:
                trials[mask] = self._mutate_current_adaptive(population, fitness, count)

        return trials

    def _mutate_rand_ring_pbest(self, population, fitness, count):
        sorted_idx = np.argsort(fitness)
        ring_idx = self._get_ring_topology_indices(sorted_idx, step=3)
        pbest_idx = sorted_idx[np.random.randint(0, max(1, int(self.p_best_rate * self.NP)))]

        r1 = np.random.choice(self.NP, size=count, replace=False)
        r2 = ring_idx[np.random.choice(self.NP, size=count, replace=False)]

        pbest = population[pbest_idx]
        F_var = np.random.uniform(0.4, 0.9, size=count).reshape(-1, 1)

        return population[r1] + F_var * (pbest - population[r1]) + F_var * (population[r2[:, 0]] - population[r2[:, 1]])

    def _mutate_best_2dir(self, population, count):
        best_idx = np.argmin(self.best_fitness_history) if self.best_fitness_history else 0
        sorted_idx = np.argsort(np.arange(self.NP))
        ring_idx = self._get_ring_topology_indices(sorted_idx, step=2)

        r1 = np.random.choice(self.NP, size=count, replace=False)
        r2 = ring_idx[np.random.choice(self.NP, size=count, replace=False)]

        best = population[0]
        F1 = np.random.uniform(0.5, 0.8, size=count).reshape(-1, 1)
        F2 = np.random.uniform(0.2, 0.5, size=count).reshape(-1, 1)

        return best + F1 * (population[r1] - population[r2[:, 0]]) + F2 * (population[r2[:, 1]] - population[r2[:, 0]])

    def _mutate_current_to_pbest_archive(self, population, fitness, count):
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(self.p_best_rate * self.NP))]
        n_archive = len(self.archive) if self.archive else 0

        r1 = np.random.choice(self.NP, size=count, replace=False)
        r2 = np.random.choice(pbest_idx, size=count)

        current = population[r1]
        pbest = population[r2]
        F_var = np.random.uniform(0.4, 1.0, size=count).reshape(-1, 1)

        donor = current + F_var * (pbest - current)

        if n_archive > 0 and count > 0:
            n_archive_samples = min(count, n_archive)
            archive_samples = np.array(self.archive[-n_archive_samples:])
            archive_idx = np.random.choice(n_archive, size=count, replace=True)
            archive_donors = archive_samples[archive_idx]
            weight = np.random.uniform(0.0, 0.3, size=count).reshape(-1, 1)
            donor = donor + weight * (archive_donors - current)

        return donor

    def _mutate_rand_2opt(self, population, count):
        indices = np.arange(self.NP)
        r1 = np.random.choice(indices, size=count, replace=False)
        r2 = np.random.choice(indices, size=count, replace=False)
        r3 = np.random.choice(indices, size=count, replace=False)
        r4 = np.random.choice(indices, size=count, replace=False)

        F1 = np.random.uniform(0.5, 0.9, size=count).reshape(-1, 1)
        F2 = np.random.uniform(0.3, 0.7, size=count).reshape(-1, 1)

        return population[r1] + F1 * (population[r2] - population[r3]) + F2 * (population[r4] - population[r1])

    def _mutate_best_ring(self, population, count):
        sorted_idx = np.argsort(np.arange(self.NP))
        ring_idx = self._get_ring_topology_indices(sorted_idx, step=3)

        best_idx = 0
        r1 = np.random.choice(self.NP, size=count, replace=False)
        r2 = ring_idx[np.random.choice(self.NP, size=count, replace=False)]

        best = population[best_idx]
        F_var = np.random.uniform(0.6, 1.0, size=count).reshape(-1, 1)

        return best + F_var * (population[r1] - population[r2[:, 0]] + population[r2[:, 1]] - population[r2[:, 2]])

    def _mutate_current_adaptive(self, population, fitness, count):
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:self.NP // 2]

        r1 = np.random.choice(self.NP, size=count, replace=False)
        r2 = np.random.choice(top_half, size=count, replace=False)
        r3 = np.random.choice(self.NP, size=count, replace=False)

        current_mean = np.mean(population, axis=0)
        F_adaptive = np.random.uniform(0.3, 0.8, size=count).reshape(-1, 1)

        return population[r1] + F_adaptive * (current_mean - population[r1]) + F_adaptive * (population[r2] - population[r3])

    def _crossover_batch(self, population, trials):
        CR_per_individual = np.random.uniform(0.7, 0.95, size=self.NP).reshape(-1, 1)
        mask = np.random.random((self.NP, self.dim)) < CR_per_individual
        mask[np.arange(self.NP), np.random.randint(0, self.dim, size=self.NP)] = True
        return np.where(mask, trials, population)

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        improved = trial_fitness < fitness
        survivors_mask = improved | (np.random.random(self.NP) < 0.1)

        new_population = np.where(survivors_mask.reshape(-1, 1), trials, population)
        new_fitness = np.where(survivors_mask, trial_fitness, fitness)

        return new_population, new_fitness, improved

    def _update_archive(self, population, fitness, trials, trial_fitness, improved):
        if len(self.archive) > self.archive_size_max * 2:
            self.archive = self.archive[-self.archive_size_max:]

        for i in range(len(improved)):
            if improved[i]:
                self.archive.append(population[i].copy())

        if len(self.archive) > self.archive_size_max:
            keep_indices = np.random.choice(len(self.archive), self.archive_size_max, replace=False)
            self.archive = [self.archive[j] for j in keep_indices]

    def _update_best_tracking(self, fitness):
        if len(fitness) > 0:
            current_best = np.min(fitness)
            if len(self.best_fitness_history) > 0:
                improvement = self.best_fitness_history[-1] - current_best
                self.improvement_history.append(improvement)
            self.best_fitness_history.append(current_best)

    def _update_strategy_scores(self, strategy_indices, improved):
        for s in range(self.num_strategies):
            mask = strategy_indices == s
            success_in_strategy = np.sum(improved[mask])
            attempts = np.sum(mask)

            if attempts > 0:
                success_rate = success_in_strategy / attempts
                self.strategy_scores[s] = (
                    0.9 * self.strategy_scores[s] + 0.1 * success_rate * 10
                )
                self.strategy_scores[s] = np.clip(self.strategy_scores[s], 0.01, None)

    def _adapt_parameters(self):
        if len(self.improvement_history) >= 10:
            recent_improvements = self.improvement_history[-10:]
            avg_improvement = np.mean(recent_improvements)

            if avg_improvement > 1e-6:
                self.F = np.clip(self.F * 1.02, 0.3, 1.0)
                self.CR = np.clip(self.CR * 1.01, 0.5, 0.99)
            elif avg_improvement < 1e-8:
                self.F = np.clip(self.F * 0.95, 0.3, 1.0)
                self.CR = np.clip(self.CR * 0.98, 0.5, 0.99)

    def _check_stagnation(self):
        if len(self.improvement_history) < 10:
            return False

        recent = self.improvement_history[-10:]
        max_imp = np.max(np.abs(recent))

        if max_imp < self.min_improvement:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        return self.stagnation_counter >= self.max_stagnation

    def _restart_if_stagnant(self, population, fitness, func):
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        best_fit = fitness[best_idx]

        new_population = self._initialize_population()
        n_elite = max(1, self.NP // 10)
        new_population[:n_elite] = best + np.random.uniform(-5, 5, size=(n_elite, self.dim))

        for i in range(n_elite, self.NP):
            dim_subset = np.random.choice(self.dim, max(1, self.dim // 3), replace=False)
            new_population[i, dim_subset] = best[dim_subset] + np.random.uniform(-10, 10, size=len(dim_subset))

        new_population = self._clip_to_bounds_batch(new_population)
        new_fitness = func(new_population)

        if len(new_fitness) < self.NP:
            new_population = new_population[:len(new_fitness)]
            new_fitness = new_fitness[:len(new_fitness)]

        self.archive = []
        self.strategy_scores = np.ones(self.num_strategies)

        return new_population, new_fitness

    def _reset_stagnation(self):
        self.stagnation_counter = 0
        self.improvement_history = []

    def _compute_population_diversity(self, population):
        if len(population) < 2:
            return 0.0
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)

    def _compute_fitness_variance(self, fitness):
        return np.var(fitness) if len(fitness) > 0 else 0.0
```