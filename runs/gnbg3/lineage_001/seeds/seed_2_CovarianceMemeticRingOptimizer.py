import numpy as np


class CovarianceMemeticRingOptimizer:
    """
    A population-based optimizer combining:
    - Ring topology for information exchange (cellular DE-inspired)
    - Covariance-adaptive step size (simplified CMA-ES)
    - Opposition-based learning for exploration
    - Embedded hill-climbing for local exploitation
    - Adaptive parameter control and restart mechanism
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.NP = min(max(4 * dim, 40), min(300, 8 * dim))
        self.F = 0.5
        self.CR = 0.5
        self.step_size = (self.bounds[0, 1] - self.bounds[0, 0]) * 0.1
        self.pc = 0.25
        self.penalty = 2.0
        self.counteval = 0
        self.best_fitness = np.inf
        self.best_solution = None
        self.stagnation = 0
        self.improvement_history = []
        self.strategy_scores = np.ones(3)
        self.cov_prev = None
        self.mean_prev = None
        self.diversity_threshold = 1e-6
        self.generation = 0
        self.restart_count = 0
        self.last_local_search_gen = 0
        self.local_search_interval = 25

    def _initialize_population(self, func):
        half = self.NP // 2
        uniform = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(half, self.dim))
        opposite = self.bounds[:, 0] + self.bounds[:, 1] - uniform
        population = np.vstack([uniform, opposite])
        if len(population) > self.NP:
            population = population[:self.NP]
        elif len(population) < self.NP:
            extra = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.NP - len(population), self.dim))
            population = np.vstack([population, extra])
        population = np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
            fitness = fitness[:len(fitness)]
        self.counteval += len(fitness)
        self.NP = len(population)
        self._update_best(population, fitness)
        self._init_covariance(population)
        return population, fitness

    def _init_covariance(self, population):
        self.cov_prev = np.cov(population.T) + 1e-6 * np.eye(self.dim)
        self.mean_prev = population.mean(axis=0)

    def _get_ring_indices(self, NP):
        idx = np.arange(NP)
        return np.column_stack([np.roll(idx, 1), np.roll(idx, -1)])

    def _mutate_ring_batch(self, population, fitness):
        NP, dim = population.shape
        ring = self._get_ring_indices(NP)
        sorted_idx = np.argsort(fitness)
        trials = np.empty_like(population)
        for i, target in enumerate(population):
            rank = np.where(sorted_idx == i)[0][0]
            p_best = population[sorted_idx[0]]
            if rank < NP * 0.2:
                base = p_best
            else:
                base = target
            n1_idx = ring[i, 0]
            n2_idx = ring[i, 1]
            n1 = population[n1_idx]
            n2 = population[n2_idx]
            scale = self.F * np.random.uniform(0.8, 1.2)
            mutant = base + scale * (n1 - n2)
            trials[i] = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
        return trials

    def _mutate_current_to_pbest_batch(self, population, fitness):
        NP, dim = population.shape
        sorted_idx = np.argsort(fitness)
        p_best = population[sorted_idx[0]]
        top_quartile = sorted_idx[:max(1, NP // 4)]
        trials = np.empty_like(population)
        for i in range(NP):
            r1, r2 = np.random.choice(top_quartile, 2, replace=False)
            donor = p_best + self.F * (population[r1] - population[r2])
            trials[i] = np.clip(donor, self.bounds[:, 0], self.bounds[:, 1])
        return trials

    def _mutate_rand_batch(self, population):
        NP, dim = population.shape
        indices = np.arange(NP)
        trials = np.empty_like(population)
        for i in range(NP):
            candidates = np.delete(indices, i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = population[r1] + self.F * (population[r2] - population[r3])
            trials[i] = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
        return trials

    def _build_mutant_batch(self, population, fitness):
        probs = self.strategy_scores / self.strategy_scores.sum()
        strat = np.random.choice(3, p=probs)
        if strat == 0:
            return self._mutate_ring_batch(population, fitness)
        elif strat == 1:
            return self._mutate_current_to_pbest_batch(population, fitness)
        else:
            return self._mutate_rand_batch(population)

    def _crossover_batch(self, targets, mutants):
        NP, dim = mutants.shape
        mask = np.random.random((NP, dim)) < self.CR
        j_rand = np.random.randint(dim, size=NP)
        mask[np.arange(NP), j_rand] = True
        trials = np.where(mask, mutants, targets)
        return trials

    def _adapt_covariance_step(self, population, fitness):
        sorted_idx = np.argsort(fitness)
        mean_curr = population.mean(axis=0)
        cov_curr = np.cov(population.T) + 1e-8 * np.eye(self.dim)
        if self.mean_prev is not None:
            mean_shift = np.linalg.norm(mean_curr - self.mean_prev)
            shift_ratio = mean_shift / (np.linalg.norm(self.mean_prev) + 1e-8)
            if mean_shift > 1e-8:
                self.step_size *= np.exp(0.2 * (shift_ratio - 0.5))
                self.step_size = np.clip(self.step_size, 1e-6, 50.0)
            alpha_cov = min(0.5, 0.1 + 0.4 * shift_ratio)
            self.cov_prev = (1 - alpha_cov) * self.cov_prev + alpha_cov * cov_curr
        self.mean_prev = mean_curr.copy()
        try:
            L = np.linalg.cholesky(self.cov_prev)
            perturbation = np.random.randn(self.dim)
            adjustment = L @ perturbation * self.step_size * 0.1
            return adjustment
        except np.linalg.LinAlgError:
            self.cov_prev = np.eye(self.dim) * (self.bounds[0, 1] - self.bounds[0, 0]) * 0.05
            return np.zeros(self.dim)

    def _apply_cov_adjustment(self, population, fitness):
        sorted_idx = np.argsort(fitness)
        adjustment = self._adapt_covariance_step(population, fitness)
        top_indices = sorted_idx[:max(2, self.NP // 10)]
        adjusted = population.copy()
        for idx in top_indices:
            candidate = population[idx] + adjustment
            adjusted[idx] = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])
        return adjusted

    def _hill_climb_one_step(self, x, func):
        best = x.copy()
        f_best = func(x.reshape(1, -1))[0]
        for _ in range(self.dim):
            direction = np.zeros(self.dim)
            dim_idx = np.random.randint(self.dim)
            direction[dim_idx] = np.random.uniform(-1, 1) * self.step_size * 2
            candidate = np.clip(x + direction, self.bounds[:, 0], self.bounds[:, 1])
            f_cand = func(candidate.reshape(1, -1))[0]
            if f_cand < f_best:
                best = candidate
                f_best = f_cand
                break
        return best, f_best

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        improved = trial_fitness < fitness
        new_population = population.copy()
        new_fitness = fitness.copy()
        for i in range(len(population)):
            if improved[i]:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
        return new_population, new_fitness, np.sum(improved)

    def _update_best(self, population, fitness):
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = population[best_idx].copy()
            self.stagnation = 0
        else:
            self.stagnation += 1

    def _adapt_parameters(self, n_improved):
        improv_rate = n_improved / self.NP
        if improv_rate > 0.3:
            self.F = np.clip(self.F * 1.1, 0.3, 1.5)
            self.CR = np.clip(self.CR * 1.05, 0.1, 0.95)
        elif improv_rate < 0.1:
            self.F = np.clip(self.F * 0.9, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.95, 0.1, 0.95)
        if self.stagnation > 5:
            self.F = np.clip(self.F * 1.2, 0.3, 1.5)
        if self.generation % 50 == 0 and self.generation > 0:
            self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1), 0.3, 1.5)
            self.CR = np.clip(self.CR + np.random.uniform(-0.1, 0.1), 0.1, 0.95)

    def _update_strategy_scores(self, n_improved, population, fitness):
        self.strategy_scores *= 0.95
        if n_improved > self.NP * 0.15:
            self.strategy_scores += 0.5
        else:
            self.strategy_scores += 0.05
        self.strategy_scores = np.clip(self.strategy_scores, 0.1, None)

    def _compute_diversity(self, population):
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        diameter = np.sqrt(self.dim) * (self.bounds[0, 1] - self.bounds[0, 0])
        return distances.mean() / diameter

    def _restart_if_needed(self, population, fitness):
        diversity = self._compute_diversity(population)
        need_restart = (diversity < self.diversity_threshold or
                        self.stagnation > max(20, 3 * self.NP // dim if 'dim' in dir() else 30) or
                        self.F > 1.4)
        if need_restart:
            self.restart_count += 1
            self.F = 0.5
            self.CR = 0.5
            self.step_size = (self.bounds[0, 1] - self.bounds[0, 0]) * 0.1
            new_pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.NP, self.dim))
            new_pop[0] = self.best_solution if self.best_solution is not None else new_pop[0]
            new_pop = np.clip(new_pop, self.bounds[:, 0], self.bounds[:, 1])
            self._init_covariance(new_pop)
            self.strategy_scores = np.ones(3)
            self.stagnation = 0
            return new_pop
        return population

    def __call__(self, func, stopping_condition):
        dim = self.dim
        population, fitness = self._initialize_population(func)
        while not stopping_condition():
            self.generation += 1
            mutants = self._build_mutant_batch(population, fitness)
            trials = self._crossover_batch(population, mutants)
            trials = np.clip(trials, self.bounds[:, 0], self.bounds[:, 1])
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            self.counteval += len(trial_fitness)
            if stopping_condition():
                break
            population, fitness, n_improved = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            self._update_best(population, fitness)
            self._adapt_parameters(n_improved)
            self._update_strategy_scores(n_improved, population, fitness)
            if self.generation - self.last_local_search_gen >= self.local_search_interval:
                if self.best_solution is not None:
                    ls_solution, ls_fitness = self._hill_climb_one_step(self.best_solution, func)
                    self.counteval += 1
                    if ls_fitness < self.best_fitness:
                        self.best_fitness = ls_fitness
                        self.best_solution = ls_solution.copy()
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ls_solution
                        fitness[worst_idx] = ls_fitness
                self.last_local_search_gen = self.generation
            if self.generation % 10 == 0:
                adjusted_pop = self._apply_cov_adjustment(population, fitness)
                adj_fitness = func(adjusted_pop)
                if len(adj_fitness) < len(adjusted_pop):
                    adj_fitness = adj_fitness[:len(adjusted_pop)]
                self.counteval += len(adj_fitness)
                if stopping_condition():
                    break
                for i in range(min(len(population), len(adj_fitness))):
                    if adj_fitness[i] < fitness[i]:
                        population[i] = adjusted_pop[i]
                        fitness[i] = adj_fitness[i]
                self._update_best(population, fitness)
            population = self._restart_if_needed(population, fitness)
            if len(population) < self.NP:
                extra = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.NP - len(population), self.dim))
                extra = np.clip(extra, self.bounds[:, 0], self.bounds[:, 1])
                population = np.vstack([population, extra])
                extra_fit = func(extra)
                if len(extra_fit) < len(extra):
                    extra_fit = extra_fit[:len(extra)]
                self.counteval += len(extra_fit)
                if stopping_condition():
                    break
                fitness = np.concatenate([fitness, extra_fit])
        return self.best_fitness, self.best_solution
