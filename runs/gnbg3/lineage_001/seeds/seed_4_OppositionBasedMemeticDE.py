import numpy as np


class OppositionBasedMemeticDE:
    """
    Hybrid optimizer combining:
    - Opposition-Based Learning for population initialization and restarts
    - DE/current-to-pbest/1 mutation with archive
    - Adaptive F/Cr parameters with success-history tracking
    - Periodic Nelder-Mead local search on elite individuals
    - Adaptive restart on stagnation/convergence
    """

    def __init__(self, dim, NP=None, F_init=0.6, Cr_init=0.5,
                 local_search_interval=7, restart_interval=15,
                 diversity_threshold=1e-5, max_restarts=3):
        self.dim = dim
        self.NP = NP if NP is not None else 5 * dim
        self.NP = min(max(20, self.NP), 300)
        self.F = F_init
        self.Cr = Cr_init
        self.local_search_interval = local_search_interval
        self.restart_interval = restart_interval
        self.diversity_threshold = diversity_threshold
        self.max_restarts = max_restarts
        self.F_history = []
        self.Cr_history = []
        self.success_buffer_F = []
        self.success_buffer_Cr = []
        self.strategy_scores = {0: {'success': 0, 'total': 0}, 1: {'success': 0, 'total': 0}}
        self.restart_count = 0

    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        generation = 0
        while not stopping_condition():
            trials = self._build_trial_batch(self.population, generation)
            trial_fit = func(trials)
            if len(trial_fit) < len(trials):
                trials = trials[:len(trial_fit)]
                trial_fit = self._handle_budget_exhaustion(trial_fit, trials)
                if len(trial_fit) == 0:
                    break
            if stopping_condition():
                break
            improved_mask = trial_fit < self.fitness
            self._update_success_history(improved_mask, trials, trial_fit)
            self.population, self.fitness, self.archive = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fit, self.archive
            )
            self._update_best(self.fitness)
            if generation % self.local_search_interval == 0 and generation > 0:
                self.population, self.fitness = self._local_search_batch(
                    self.population, self.fitness, func
                )
                self._update_best(self.fitness)
            self._adapt_parameters()
            if generation > 0 and generation % self.restart_interval == 0:
                self._restart_if_needed(generation)
            generation += 1
        return self.best_fitness, self.best_position

    def _initialize_population(self, func):
        pop = np.random.uniform(-100, 100, (self.NP, self.dim))
        pop = self._clip_to_bounds(pop)
        fitness = func(pop)
        if len(fitness) < len(pop):
            pop = pop[:len(fitness)]
            fitness = fitness[:len(fitness)]
            self.NP = len(pop)
        opp_pop = self._generate_opposition_batch(pop)
        opp_fitness = func(opp_pop)
        combined_pop = np.vstack([pop, opp_pop])
        combined_fit = np.concatenate([fitness, opp_fitness])
        indices = np.argsort(combined_fit)[:self.NP]
        self.population = combined_pop[indices]
        self.fitness = combined_fit[indices]
        self.archive = np.empty((0, self.dim))
        self.best_fitness = np.min(self.fitness)
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx].copy()
        self.gens_no_improve = [0]

    def _generate_opposition_batch(self, pop):
        opp = 200.0 - pop
        return self._clip_to_bounds(opp)

    def _clip_to_bounds(self, pop):
        return np.clip(pop, -100.0, 100.0)

    def _build_trial_batch(self, pop, generation):
        mutant = self._mutate_batch(pop)
        trial = self._crossover_batch(pop, mutant)
        trial = self._clip_to_bounds(trial)
        return trial

    def _mutate_batch(self, pop):
        NP, dim = pop.shape
        pbest_frac = 0.15
        pbest_count = max(2, int(NP * pbest_frac))
        pbest_indices = np.argsort(self.fitness)[:pbest_count]
        pbest = pop[pbest_indices]
        targets = np.arange(NP)
        r1_idx = np.random.randint(0, NP, (NP, 3))
        r2_idx = np.random.randint(0, NP, (NP, 3))
        r3_idx = np.random.randint(0, NP + len(self.archive), (NP, 3))
        np.fill_diagonal(r1_idx, (r1_idx.diagonal() + 1) % NP)
        np.fill_diagonal(r2_idx, (r2_idx.diagonal() + 2) % NP)
        r1 = pop[r1_idx[:, 0]]
        r2 = pop[r1_idx[:, 1]]
        r3_base = np.vstack([pop, self.archive]) if len(self.archive) > 0 else pop
        r3 = r3_base[r3_idx[:, 0] % len(r3_base)]
        p_idx = np.random.randint(0, pbest_count, NP)
        F_adapted = np.clip(self.F + np.random.normal(0, 0.1, NP), 0.3, 1.0)
        mutant = pop + F_adapted[:, None] * (pbest[p_idx] - pop) + F_adapted[:, None] * (r1 - r2)
        mutant += 0.1 * F_adapted[:, None] * (r3 - pop)
        return mutant

    def _crossover_batch(self, pop, mutant):
        NP, dim = self.population.shape
        Cr_adapted = np.clip(self.Cr + np.random.normal(0, 0.1, NP), 0.1, 1.0)
        mask = np.random.rand(NP, dim) < Cr_adapted[:, None]
        mask[np.arange(NP), np.random.randint(0, dim, NP)] = True
        trial = np.where(mask, mutant, pop)
        return trial

    def _select_survivors_batch(self, pop, fitness, trial, trial_fit, archive):
        improved = trial_fit < fitness
        new_pop = np.where(improved[:, None], trial, pop)
        new_fit = np.where(improved, trial_fit, fitness)
        new_archive = archive
        if np.any(improved):
            improved_solutions = trial[improved]
            new_archive = np.vstack([archive, improved_solutions]) if len(archive) > 0 else improved_solutions
            max_archive = 3 * self.NP
            if len(new_archive) > max_archive:
                keep_idx = np.random.choice(len(new_archive), max_archive, replace=False)
                new_archive = new_archive[keep_idx]
        return new_pop, new_fit, new_archive

    def _update_success_history(self, improved_mask, trials, trial_fit):
        if np.any(improved_mask):
            self.success_buffer_F.append(self.F)
            self.success_buffer_Cr.append(self.Cr)
            strategy_id = 0
            self.strategy_scores[strategy_id]['total'] += 1
            self.strategy_scores[strategy_id]['success'] += 1
        else:
            strategy_id = 1
            self.strategy_scores[strategy_id]['total'] += 1

    def _adapt_parameters(self):
        if len(self.success_buffer_F) > 0:
            successful_F = np.mean(self.success_buffer_F)
            successful_Cr = np.mean(self.success_buffer_Cr)
            self.F = 0.8 * self.F + 0.2 * successful_F
            self.Cr = 0.8 * self.Cr + 0.2 * successful_Cr
        self.F = np.clip(self.F, 0.3, 1.0)
        self.Cr = np.clip(self.Cr, 0.1, 0.9)
        self.F_history.append(self.F)
        self.Cr_history.append(self.Cr)
        self.success_buffer_F = []
        self.success_buffer_Cr = []

    def _local_search_batch(self, pop, fitness, func):
        n_ls = max(1, self.NP // 10)
        top_indices = np.argsort(fitness)[:n_ls]
        for idx in top_indices:
            x = pop[idx].copy()
            f_curr = fitness[idx]
            step = 10.0
            for _ in range(15):
                improved_epoch = False
                for d in range(self.dim):
                    for delta in [-step, step]:
                        x_trial = x.copy()
                        x_trial[d] += delta
                        x_trial = self._clip_to_bounds(x_trial.reshape(1, -1))[0]
                        f_trial = func(x_trial.reshape(1, -1))
                        if len(f_trial) > 0 and f_trial[0] < f_curr:
                            x = x_trial
                            f_curr = f_trial[0]
                            improved_epoch = True
                if not improved_epoch:
                    step *= 0.5
                if step < 0.1:
                    break
            pop[idx] = x
            fitness[idx] = f_curr
        return pop, fitness

    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)

    def _restart_if_needed(self, generation):
        diversity = self._compute_diversity()
        recent_no_improve = self.gens_no_improve[-1] if len(self.gens_no_improve) > 0 else 0
        if (diversity < self.diversity_threshold or recent_no_improve >= self.restart_interval) \
                and self.restart_count < self.max_restarts:
            self._restart_population(func=None)
            self.restart_count += 1
            self.gens_no_improve.append(0)

    def _restart_population(self, func):
        n_elite = max(2, self.NP // 4)
        elite_indices = np.argsort(self.fitness)[:n_elite]
        elite = self.population[elite_indices].copy()
        opp_elite = self._generate_opposition_batch(elite)
        n_new = self.NP - len(elite) * 2
        if n_new > 0:
            new_pop = np.random.uniform(-100, 100, (n_new, self.dim))
            new_pop = self._clip_to_bounds(new_pop)
            self.population = np.vstack([elite, opp_elite, new_pop])
        else:
            combined = np.vstack([elite, opp_elite])
            self.population = combined[np.argsort(func(combined)[:len(combined)])][:self.NP]
        self.fitness = func(self.population)
        self.archive = np.empty((0, self.dim))
        self.F = np.clip(self.F * 1.2, 0.3, 1.0)
        self.Cr = np.clip(self.Cr * 1.2, 0.1, 0.9)

    def _handle_budget_exhaustion(self, trial_fit, trials):
        if len(trial_fit) == 0:
            return trial_fit
        valid_trials = trials[:len(trial_fit)]
        nan_mask = np.isnan(trial_fit)
        if np.all(nan_mask):
            return trial_fit
        valid_mask = ~nan_mask
        self.population = np.where(valid_mask[:, None], valid_trials, self.population[:len(valid_trials)])
        self.fitness = np.where(valid_mask, trial_fit, self.fitness[:len(trial_fit)])
        return trial_fit

    def _update_best(self, fitness):
        best_idx = np.argmin(fitness)
        if len(fitness) > 0 and not np.isnan(fitness[best_idx]):
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_position = self.population[best_idx].copy()
                self.gens_no_improve[-1] = 0
            else:
                self.gens_no_improve[-1] += 1
