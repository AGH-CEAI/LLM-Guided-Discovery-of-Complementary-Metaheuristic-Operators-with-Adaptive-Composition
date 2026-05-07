import numpy as np


class CovarianceGuidedMemeticOptimizer:
    """
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with multiple mutation strategies and online strategy adaptation.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(8 * dim, 60), 200)
        self.bounds = (-100.0, 100.0)

        self.cov_matrix = np.eye(dim)
        self.mean = np.zeros(dim)
        self.step_size = 2.0
        self.pc = np.zeros(dim)

        self.cov_lr = 0.05
        self.mean_lr = 0.1
        self.step_decay = 0.95
        self.step_growth = 1.05

        self.F = 0.5
        self.CR = 0.7

        self.n_strategies = 3
        self.strategy_names = ['covariance', 'de_current_best', 'de_rand']
        self.strategy_scores = np.ones(self.n_strategies)
        self.strategy_probs = np.ones(self.n_strategies) / self.n_strategies
        self.strategy_weight_mult = 0.9

        self.success_buffer = []
        self.success_window = 10
        self.min_success_rate = 0.15
        self.max_success_rate = 0.25

        self.stagnation_tol = 1e-8
        self.stagnation_count = 0
        self.stagnation_max = 15

        self.diversity_threshold = 0.01 * dim
        self.restart_triggered = False

    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling."""
        sampler = np.linspace(self.bounds[0], self.bounds[1], self.NP)
        population = np.zeros((self.NP, self.dim))
        for d in range(self.dim):
            population[:, d] = np.random.permutation(sampler) + \
                              np.random.uniform(-1, 1, self.NP)
        jitter = np.random.randn(self.NP, self.dim) * (self.bounds[1] - self.bounds[0]) / self.NP
        population += jitter
        self.mean = np.mean(population, axis=0)
        self.cov_matrix = np.cov(population, rowvar=False) + 0.01 * np.eye(self.dim)
        self.cov_matrix = np.clip(self.cov_matrix, 1e-6, None)
        return np.clip(population, self.bounds[0], self.bounds[1])

    def _compute_population_diversity(self, population):
        """Compute average pairwise Euclidean distance for diversity assessment."""
        flat = population.reshape(self.NP, 1, self.dim)
        diffs = flat - flat.transpose(1, 0, 2)
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        upper_tri = distances[np.triu_indices(self.NP, k=1)]
        return np.mean(upper_tri) if len(upper_tri) > 0 else 0.0

    def _compute_centroid(self, population):
        """Update weighted centroid from population."""
        return np.mean(population, axis=0)

    def _update_covariance_adaptation(self, best_member):
        """Update covariance matrix and step size using evolution path and rank-1 update."""
        diff = best_member - self.mean
        self.pc = (1 - self.cov_lr) * self.pc + np.sqrt(self.cov_lr * (2 - self.cov_lr)) * diff / self.step_size

        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros((self.dim, self.dim))

        self.mean = self.mean_lr * best_member + (1 - self.mean_lr) * self.mean

        self.cov_matrix = (1 - self.cov_lr) * self.cov_matrix + \
                         self.cov_lr * (rank_one + (1 - self.cov_lr) * rank_mu)
        self.cov_matrix = np.clip(self.cov_matrix, 1e-6, None)

        if len(self.success_buffer) >= self.success_window:
            recent = self.success_buffer[-self.success_window:]
            rate = np.mean(recent)
            if rate > self.max_success_rate:
                self.step_size *= self.step_growth
            elif rate < self.min_success_rate:
                self.step_size *= self.step_decay
            self.step_size = np.clip(self.step_size, 1e-10, 10.0)

    def _mutate_covariance_batch(self, population):
        """Generate trial vectors using covariance-adapted multivariate normal sampling."""
        z = np.random.randn(self.NP, self.dim)
        try:
            L = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(self.dim)
        trials = self.mean + self.step_size * (z @ L.T)
        return np.clip(trials, self.bounds[0], self.bounds[1])

    def _mutate_de_current_best_batch(self, population, fitness):
        """Generate trials using DE/current-to-best/1 mutation."""
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        idx_a = np.random.randint(0, self.NP, size=self.NP)
        idx_b = np.random.randint(0, self.NP, size=self.NP)
        while np.any(idx_a == best_idx):
            idx_a = np.random.randint(0, self.NP, size=self.NP)
        while np.any(idx_b == best_idx) | np.any(idx_b == idx_a):
            idx_b = np.random.randint(0, self.NP, size=self.NP)
        trials = best + self.F * (population[idx_a] - population[idx_b])
        return np.clip(trials, self.bounds[0], self.bounds[1])

    def _mutate_de_rand_batch(self, population):
        """Generate trials using DE/rand/1 mutation for diversity."""
        indices = np.random.choice(self.NP, size=(self.NP, 3), replace=False)
        r1, r2, r3 = population[indices[:, 0]], population[indices[:, 1]], population[indices[:, 2]]
        trials = r1 + self.F * (r2 - r3)
        return np.clip(trials, self.bounds[0], self.bounds[1])

    def _crossover_batch(self, population, mutants):
        """Perform binomial crossover with guaranteed variable inheritance."""
        cr = self.CR
        mask = np.random.rand(self.NP, self.dim) < cr
        j_rand = np.random.randint(self.dim)
        mask[np.arange(self.NP), j_rand] = True
        return np.where(mask, mutants, population)

    def _select_survivors_batch(self, population, trials, fitness, trial_fitness):
        """Greedy (mu, lambda) selection keeping best of parent or trial."""
        improved = trial_fitness < fitness
        next_pop = np.where(improved[:, np.newaxis], trials, population)
        next_fit = np.where(improved, trial_fitness, fitness)
        return next_pop, next_fit

    def _update_strategy_scores(self, fitness, trial_fitness):
        """Update strategy selection scores based on improvements achieved."""
        improvements = fitness - trial_fitness
        self.strategy_scores *= self.strategy_weight_mult
        self.strategy_scores[self.current_strategy] += np.sum(improvements)
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-6)
        total = np.sum(self.strategy_scores)
        self.strategy_probs = self.strategy_scores / total

    def _select_strategy(self):
        """Select mutation strategy using roulette wheel selection on scores."""
        self.current_strategy = np.random.choice(self.n_strategies, p=self.strategy_probs)
        return self.current_strategy

    def _check_stagnation(self, fitness):
        """Detect stagnation based on lack of improvement in best fitness."""
        best_current = np.min(fitness)
        if self.stagnation_best is None:
            self.stagnation_best = best_current
            return False
        improvement = self.stagnation_best - best_current
        if improvement > self.stagnation_tol:
            self.stagnation_best = best_current
            self.stagnation_count = 0
            return False
        self.stagnation_count += 1
        return self.stagnation_count >= self.stagnation_max

    def _restart_if_needed(self, population, fitness):
        """Trigger restart if diversity is low or stagnation detected."""
        diversity = self._compute_population_diversity(population)
        stagnant = self._check_stagnation(fitness)
        self.restart_triggered = (diversity < self.diversity_threshold) or stagnant
        if self.restart_triggered:
            self.stagnation_count = 0
            self.stagnation_best = None
            self.step_size = max(self.step_size, 2.0)
            self.cov_matrix = np.eye(self.dim)
            self.success_buffer = []
            self.strategy_scores = np.ones(self.n_strategies)
            self.strategy_probs = np.ones(self.n_strategies) / self.n_strategies
        return self.restart_triggered

    def _apply_local_refinement(self, population, fitness):
        """Apply small-scale local search around best members (memetic component)."""
        n_refine = max(3, self.NP // 10)
        best_indices = np.argsort(fitness)[:n_refine]
        refined = population[best_indices].copy()
        for i in range(n_refine):
            step = self.step_size * 0.1
            perturbation = np.random.randn(self.dim) * step
            candidate = refined[i] + perturbation
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            refined[i] = candidate
        return refined

    def _track_success_rate(self, fitness, trial_fitness):
        """Track ratio of successful mutations for step size adaptation."""
        improved = trial_fitness < fitness
        rate = np.mean(improved)
        self.success_buffer.append(rate)
        if len(self.success_buffer) > self.success_window * 2:
            self.success_buffer = self.success_buffer[-self.success_window:]

    def __call__(self, func, stopping_condition):
        """Execute optimization loop with batched operations."""
        self.func = func
        population = self._initialize_population()
        fitness = func(population)

        if len(fitness) < self.NP:
            best_idx = np.nanargmin(fitness)
            return fitness[best_idx], population[best_idx]

        self.stagnation_best = None
        self.stagnation_count = 0
        self.current_strategy = 0

        while not stopping_condition():
            best_idx = np.nanargmin(fitness)
            best_member = population[best_idx].copy()

            self._update_covariance_adaptation(best_member)

            strategy_idx = self._select_strategy()

            if strategy_idx == 0:
                trials = self._mutate_covariance_batch(population)
            elif strategy_idx == 1:
                trials = self._mutate_de_current_best_batch(population, fitness)
            else:
                trials = self._mutate_de_rand_batch(population)

            trials = self._crossover_batch(population, trials)

            if stopping_condition():
                break

            trial_fitness = func(trials)

            if len(trial_fitness) < self.NP:
                break

            self._track_success_rate(fitness, trial_fitness)

            population, fitness = self._select_survivors_batch(population, trials, fitness, trial_fitness)

            self._update_strategy_scores(fitness, trial_fitness)

            if strategy_idx == 0:
                refined = self._apply_local_refinement(population, fitness)
                if stopping_condition():
                    break
                ref_fitness = func(refined)
                if len(ref_fitness) < len(refined):
                    break
                for i in range(len(refined)):
                    if ref_fitness[i] < fitness[i]:
                        population[i] = refined[i]
                        fitness[i] = ref_fitness[i]

            if self._restart_if_needed(population, fitness):
                population = self._initialize_population()
                if stopping_condition():
                    break
                fitness = func(population)
                if len(fitness) < self.NP:
                    break

        final_idx = np.nanargmin(fitness)
        return fitness[final_idx], population[final_idx]
