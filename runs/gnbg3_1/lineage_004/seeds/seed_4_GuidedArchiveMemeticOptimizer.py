import numpy as np


class GuidedArchiveMemeticOptimizer:
    """
    Hybrid optimizer combining DE-style mutation, adaptive archives,
    and pattern-search local refinement. Distinct from previous proposals:
    - Uses archive-guided mutation with age-based replacement
    - Adaptive strategy mixing with success-based weights
    - Pattern search (not Nelder-Mead) for local refinement
    - Covariance-free adaptation focusing on step-size and archive
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(4 * dim, 60), 400)
        self.bounds = np.array([-100.0, 100.0])
        self.lower = self.bounds[0]
        self.upper = self.bounds[1]

        self.F = 0.7
        self.CR = 0.85
        self.p_best = 0.1
        self.archive = np.empty((0, dim))
        self.max_archive = 5 * self.NP

        self.generation = 0
        self.ages = np.zeros(self.NP)
        self.max_age = 15
        self.stagnation_gen = 0

        self.strategy_weights = np.array([0.4, 0.35, 0.25])
        self.strategy_success = np.zeros(3)
        self.strategy_attempts = np.ones(3)

        self.local_search_active = False
        self.local_search_gen = 0
        self.local_step = 2.0
        self.local_interval = 8

        self.population = None
        self.fitness = None
        self.trial_population = None
        self.best_fitness = np.inf
        self.best_x = None

    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        while not stopping_condition():
            self.generation += 1
            self._update_strategy_weights()
            self._update_local_search_params()
            self._generate_trials_batch()
            self._clip_trial_bounds()
            trial_fitness = func(self.trial_population)
            if len(trial_fitness) < len(self.trial_population):
                self._handle_budget_exhausted(trial_fitness, func)
                break
            if stopping_condition():
                break
            self._evaluate_selection_batch(trial_fitness)
            self._update_best_tracking()
            self._update_archive_batch()
            self._age_population()
            self._replace_stagnant_individuals(func)
            if self.generation % self.local_interval == 0:
                self._apply_pattern_search_step(func)
            if self._detect_stagnation():
                self._restart_diversify(func)
        return self.best_fitness, self.best_x

    def _initialize_population(self, func):
        self.population = np.random.uniform(
            self.lower, self.upper, (self.NP, self.dim)
        )
        self.population = self._clip_to_bounds(self.population)
        self.fitness = func(self.population)
        self.ages = np.zeros(self.NP)
        self.archive = self.population.copy()
        self._update_best_tracking()

    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.lower, self.upper)

    def _clip_trial_bounds(self):
        self.trial_population = self._clip_to_bounds(self.trial_population)

    def _update_best_tracking(self):
        valid = np.isfinite(self.fitness)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, self.fitness, np.inf))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_x = self.population[best_idx].copy()
            self.stagnation_gen = 0
        else:
            self.stagnation_gen += 1

    def _select_mutation_strategy(self):
        r = np.random.random()
        cumsum = np.cumsum(self.strategy_weights)
        for i, threshold in enumerate(cumsum):
            if r < threshold:
                return i
        return len(cumsum) - 1

    def _mutate_rand_archive(self, idx):
        n = len(self.archive)
        if n >= 3:
            a, b, c = np.random.choice(n, 3, replace=False)
            donor = self.archive[a] + self.F * (self.archive[b] - self.archive[c])
        else:
            candidates = np.vstack([self.population, self.archive]) if n > 0 else self.population
            indices = np.random.choice(len(candidates), 3, replace=False)
            donor = candidates[indices[0]] + self.F * (
                candidates[indices[1]] - candidates[indices[2]]
            )
        return donor

    def _mutate_current_to_pbest(self, idx):
        pbest_size = max(1, int(self.p_best * self.NP))
        pbest_indices = np.argsort(self.fitness)[:pbest_size]
        pbest_idx = np.random.choice(pbest_indices)
        pbest = self.population[pbest_idx]
        current = self.population[idx]
        n = len(self.archive)
        if n >= 2:
            a, b = np.random.choice(n, 2, replace=False)
            archive_term = self.archive[a] - self.archive[b]
        else:
            others = [self.population[j] for j in range(self.NP) if j != idx]
            archive_term = np.mean(others, axis=0) - self.population[np.random.randint(self.NP)]
        donor = current + self.F * (pbest - current) + 0.5 * self.F * archive_term
        return donor

    def _mutate_guided_perturbation(self, idx):
        current = self.population[idx]
        fitness_weights = np.exp(-self.fitness / (np.mean(self.fitness) + 1e-8))
        fitness_weights /= fitness_weights.sum()
        if len(self.archive) > 0:
            combined = np.vstack([self.population, self.archive])
            combined_fitness = np.concatenate([self.fitness, self.fitness[:len(self.archive)]])
            weights = np.exp(-combined_fitness / (np.mean(combined_fitness) + 1e-8))
            weights /= weights.sum()
            candidates = combined
        else:
            candidates = self.population
            weights = fitness_weights
        a_idx = np.random.choice(len(candidates), p=weights)
        b_idx = np.random.choice(len(candidates), p=weights)
        donor = current + self.F * (candidates[a_idx] - candidates[b_idx])
        return donor

    def _generate_trials_batch(self):
        trials = np.empty((self.NP, self.dim))
        strategy_used = np.zeros(3, dtype=int)
        for i in range(self.NP):
            strategy = self._select_mutation_strategy()
            strategy_used[strategy] += 1
            if strategy == 0:
                trials[i] = self._mutate_rand_archive(i)
            elif strategy == 1:
                trials[i] = self._mutate_current_to_pbest(i)
            else:
                trials[i] = self._mutate_guided_perturbation(i)
        self.trial_population = trials
        for s in range(3):
            self.strategy_attempts[s] += strategy_used[s]

    def _crossover_batch(self, target, donor, idx):
        j_rand = np.random.randint(self.dim)
        mask = np.random.random(self.dim) < self.CR
        mask[j_rand] = True
        trial = np.where(mask, donor, target)
        return trial

    def _evaluate_selection_batch(self, trial_fitness):
        new_pop = np.empty((self.NP, self.dim))
        new_fit = np.empty(self.NP)
        improvements = np.zeros(3)
        attempts = np.zeros(3)
        for i in range(self.NP):
            if self.fitness[i] <= trial_fitness[i]:
                new_pop[i] = self.population[i]
                new_fit[i] = self.fitness[i]
            else:
                new_pop[i] = self.trial_population[i]
                new_fit[i] = trial_fitness[i]
                improvements[self._select_mutation_strategy()] += 1
        self.population = new_pop
        self.fitness = new_fit
        for s in range(3):
            if self.strategy_attempts[s] > 0:
                self.strategy_success[s] += improvements[s]

    def _update_archive_batch(self):
        n_new = self.NP // 4
        improved = self.trial_population[self.fitness < self.fitness]
        if len(improved) == 0:
            return
        n_take = min(n_new, len(improved))
        indices = np.random.choice(len(improved), n_take, replace=False)
        new_archive = improved[indices]
        self.archive = np.vstack([self.archive, new_archive])
        if len(self.archive) > self.max_archive:
            keep = np.random.choice(len(self.archive), self.max_archive, replace=False)
            self.archive = self.archive[keep]

    def _update_strategy_weights(self):
        for s in range(3):
            if self.strategy_attempts[s] > 10:
                success_rate = self.strategy_success[s] / max(1, self.strategy_attempts[s])
                self.strategy_weights[s] = 0.9 * self.strategy_weights[s] + 0.1 * success_rate
        total = self.strategy_weights.sum()
        if total > 0:
            self.strategy_weights /= total
        else:
            self.strategy_weights[:] = 1.0 / 3.0

    def _adapt_step_size(self):
        if self.stagnation_gen > 5:
            self.F = min(1.2, self.F * 1.05)
        elif self.stagnation_gen > 0:
            self.F = max(0.3, self.F * 0.95)

    def _update_local_search_params(self):
        if self.generation % (self.local_interval * 3) == 0:
            self.local_step *= 1.1
            self.local_step = min(self.local_step, 10.0)

    def _apply_pattern_search_step(self, func):
        if self.best_x is None:
            return
        center = self.best_x.copy()
        points = [center]
        for d in range(min(self.dim, 10)):
            coord = np.zeros(self.dim)
            coord[d] = self.local_step
            points.append(center + coord)
            points.append(center - coord)
        points = np.array(points)
        points = self._clip_to_bounds(points)
        fit_vals = func(points)
        best_local = np.argmin(fit_vals)
        if fit_vals[best_local] < self.best_fitness:
            improvement = self.best_fitness - fit_vals[best_local]
            self.best_fitness = fit_vals[best_local]
            self.best_x = points[best_local].copy()
            idx = np.argmin(self.fitness)
            self.population[idx] = self.best_x.copy()
            self.fitness[idx] = self.best_fitness
            if improvement > 1e-6:
                self.local_step = min(self.local_step * 1.2, 10.0)
            self.stagnation_gen = 0
        else:
            self.local_step = max(self.local_step * 0.5, 0.01)

    def _age_population(self):
        self.ages += 1

    def _replace_stagnant_individuals(self, func):
        old_mask = self.ages > self.max_age
        n_replace = np.sum(old_mask)
        if n_replace > 0:
            new_inds = np.random.uniform(
                self.lower, self.upper, (n_replace, self.dim)
            )
            new_inds = self._clip_to_bounds(new_inds)
            self.population[old_mask] = new_inds
            self.fitness[old_mask] = func(new_inds)
            self.ages[old_mask] = 0

    def _detect_stagnation(self):
        return self.stagnation_gen > 30

    def _restart_diversify(self, func):
        self.stagnation_gen = 0
        half = self.NP // 2
        best_part = self.population[np.argsort(self.fitness)[:half]]
        random_part = np.random.uniform(
            self.lower, self.upper, (self.NP - half, self.dim)
        )
        self.population[:half] = best_part
        self.population[half:] = random_part
        self.population = self._clip_to_bounds(self.population)
        self.fitness = func(self.population)
        self.ages = np.zeros(self.NP)
        self.F = 0.7
        self.local_step = 2.0

    def _handle_budget_exhausted(self, partial_fitness, func):
        n_received = len(partial_fitness)
        if n_received < len(self.trial_population):
            valid_fitness = np.full(len(self.trial_population), np.inf)
            valid_fitness[:n_received] = partial_fitness
            self._evaluate_selection_batch(valid_fitness)
        self._update_best_tracking()

    def _pad_fitness(self, fitness, target_len):
        padded = np.full(target_len, np.inf)
        padded[:len(fitness)] = fitness
        return padded
