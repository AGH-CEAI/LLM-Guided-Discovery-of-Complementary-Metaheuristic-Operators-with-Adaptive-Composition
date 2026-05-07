import numpy as np

class ClusteredMemeticDifferentialEvolution:
    """
    Hybrid optimizer combining:
    - DE/rand/1 mutation with bounded archive
    - Adaptive multi-strategy mutation portfolio
    - Adaptive cluster-based population topology
    - Dual local search: gradient-based + Nelder-Mead simplex
    - Opposition-based initialization and restart
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 10 * dim, 300)
        self.f_bounds = (-100.0, 100.0)
        self._setup_hyperparameters()
        self._setup_mutation_strategies()
        
    def _setup_hyperparameters(self):
        self.cr = 0.7
        self.f_scale = 0.6
        self.archive_rate = 0.5
        self.local_search_rate = 0.05
        self.diversity_threshold = 0.05
        self.stagnation_window = 15
        self.stagnation_tolerance = 1e-8
        self.max_local_evals_ratio = 0.1
        
    def _setup_mutation_strategies(self):
        self.strategy_scores = {'rand': 1.0, 'current': 1.0}
        self.strategy_weights = {'rand': 0.5, 'current': 0.5}
        self.strategy_trials = {'rand': 0, 'current': 0}
        self.strategy_successes = {'rand': 0, 'current': 0}
        
    def __call__(self, func, stopping_condition):
        self.func = func
        self.stopping_condition = stopping_condition
        self.local_evals_budget = int(self.np * self.max_local_evals_ratio)
        self._initialize()
        self._main_loop()
        return self._extract_best()
    
    def _initialize(self):
        self.population = self._initialize_population()
        self.fitness = self._batch_evaluate(self.population)
        self.archive = []
        self.best_history = []
        self.stagnation_counter = 0
        self.local_evals_used = 0
        self.generation = 0
        
    def _initialize_population(self):
        pop = np.random.uniform(self.f_bounds[0], self.f_bounds[1], (self.np, self.dim))
        opp_pop = self.f_bounds[0] + self.f_bounds[1] - pop
        combined = np.vstack([pop, opp_pop])
        combined_fit = self._batch_evaluate(combined)
        indices = np.argsort(combined_fit)[:self.np]
        return combined[indices]
    
    def _batch_evaluate(self, batch):
        fitness = self.func(batch)
        if len(fitness) < len(batch):
            return fitness[:len(fitness)]
        return fitness
    
    def _main_loop(self):
        while not self.stopping_condition():
            trials = self._build_trial_batch()
            trial_fit = self._batch_evaluate(trials)
            if len(trial_fit) < len(trials):
                trials = trials[:len(trial_fit)]
            if len(trial_fit) == 0:
                break
            if self.stopping_condition():
                break
            self._update_archive()
            self._update_strategy_weights(trials, trial_fit)
            self.population, self.fitness = self._select_survivors_batch(trials, trial_fit)
            self._track_best_and_stagnation()
            if self.local_evals_used < self.local_evals_budget:
                self.population, self.fitness, evals = self._local_search_batch(evals_budget=self.local_evals_budget - self.local_evals_used)
                self.local_evals_used += evals
            if self._compute_diversity() < self.diversity_threshold:
                self._opposition_jump()
            if self.stagnation_counter >= self.stagnation_window:
                self._restart_if_stagnant()
            self.generation += 1
    
    def _build_trial_batch(self):
        strategy = self._select_adaptive_strategy()
        if strategy == 'rand':
            mutants = self._mutate_rand_archive()
        else:
            mutants = self._mutate_current_to_best()
        trials = self._crossover_batch(mutants)
        trials = np.clip(trials, self.f_bounds[0], self.f_bounds[1])
        return trials
    
    def _select_adaptive_strategy(self):
        r = np.random.random()
        if r < self.strategy_weights['rand']:
            return 'rand'
        return 'current'
    
    def _mutate_rand_archive(self):
        base_idx = np.random.randint(0, self.np, self.np)
        idx_a = np.random.randint(0, self.np, self.np)
        idx_b = np.random.randint(0, self.np + len(self.archive), self.np)
        a_vecs = self.population[idx_a]
        combined = np.vstack([self.population, np.array(self.archive)]) if self.archive else self.population
        b_vecs = combined[idx_b]
        base_vecs = self.population[base_idx]
        mask_replace = (idx_a == base_idx) | (idx_b == base_idx)
        if np.any(mask_replace) and len(self.archive) > 0:
            alt_idx = np.random.randint(0, len(self.archive), self.np)
            alt_vecs = np.array(self.archive)[alt_idx]
            a_vecs = np.where(mask_replace[:, None], alt_vecs, a_vecs)
        return base_vecs + self.f_scale * (a_vecs - b_vecs)
    
    def _mutate_current_to_best(self):
        best_idx = np.argmin(self.fitness)
        best_vec = self.population[best_idx]
        idx_a = np.random.randint(0, self.np, self.np)
        idx_b = np.random.randint(0, self.np, self.np)
        mask_same = (idx_a == idx_b) | (idx_a == best_idx) | (idx_b == best_idx)
        if np.any(mask_same):
            alt_idx = np.random.randint(0, self.np, self.np)
            alt_vecs = self.population[alt_idx]
            idx_a = np.where(mask_same, alt_idx, idx_a)
            idx_b = np.where(mask_same, alt_idx, idx_b)
        a_vecs = self.population[idx_a]
        b_vecs = self.population[idx_b]
        return best_vec + self.f_scale * (a_vecs - b_vecs)
    
    def _crossover_batch(self, mutants):
        mask = np.random.random((self.np, self.dim)) < self.cr
        forced_dim = np.random.randint(0, self.dim)
        mask[:, forced_dim] = True
        return np.where(mask, mutants, self.population)
    
    def _update_archive(self):
        current_best_mask = np.ones(self.np, dtype=bool)
        for arch_vec in self.archive:
            matches = np.all(np.isclose(self.population, arch_vec, atol=1e-6), axis=1)
            current_best_mask &= ~matches
        new_archive_members = self.population[current_best_mask][:int(self.np * self.archive_rate)]
        self.archive.extend(new_archive_members.tolist())
        max_archive_size = 2 * self.np
        if len(self.archive) > max_archive_size:
            self.archive = self.archive[-max_archive_size:]
    
    def _update_strategy_weights(self, trials, trial_fit):
        improvements = trial_fit < self.fitness
        current_strategy = 'rand' if np.random.random() < self.strategy_weights['rand'] else 'current'
        self.strategy_trials[current_strategy] += len(trials)
        self.strategy_successes[current_strategy] += np.sum(improvements)
        if self.generation % 5 == 0:
            for key in self.strategy_scores:
                if self.strategy_trials[key] > 0:
                    success_rate = self.strategy_successes[key] / self.strategy_trials[key]
                    self.strategy_scores[key] = 0.9 * self.strategy_scores.get(key, 1.0) + 0.1 * success_rate
            total = sum(self.strategy_scores.values())
            self.strategy_weights = {k: v / total for k, v in self.strategy_scores.items()}
    
    def _select_survivors_batch(self, trials, trial_fit):
        combined_fit = np.concatenate([self.fitness, trial_fit])
        combined_pop = np.vstack([self.population, trials])
        winners_mask = np.zeros(2 * self.np, dtype=bool)
        winners_mask[:self.np] = True
        winners_mask[self.np:] = trial_fit < self.fitness
        survivors = combined_pop[winners_mask]
        survivor_fitness = combined_fit[winners_mask]
        return survivors[:self.np], survivor_fitness[:self.np]
    
    def _track_best_and_stagnation(self):
        best_gen = np.min(self.fitness)
        self.best_history.append(best_gen)
        if len(self.best_history) > self.stagnation_window:
            self.best_history.pop(0)
        if len(self.best_history) >= 2:
            recent_improvement = self.best_history[-1] - min(self.best_history)
            if abs(recent_improvement) < self.stagnation_tolerance:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
    
    def _local_search_batch(self, evals_budget=10):
        n_search = max(1, int(self.np * self.local_search_rate))
        sorted_indices = np.argsort(self.fitness)
        search_indices = sorted_indices[:n_search]
        evals_used = 0
        for idx in search_indices:
            if evals_used >= evals_budget:
                break
            x = self.population[idx].copy()
            f_x = self.fitness[idx]
            simplex = np.tile(x, (self.dim + 1, 1))
            for j in range(self.dim):
                simplex[j + 1, j] += np.random.uniform(-0.5, 0.5)
            simplex = np.clip(simplex, self.f_bounds[0], self.f_bounds[1])
            simplex_fitness = self._batch_evaluate(simplex)
            evals_used += len(simplex_fitness)
            for _ in range(max(3, self.dim)):
                if evals_used >= evals_budget:
                    break
                sorted_idx = np.argsort(simplex_fitness)
                centroid = np.mean(simplex_fitness[:-1], axis=0)
                worst_idx = sorted_idx[-1]
                reflected = 2 * centroid - simplex[worst_idx]
                reflected = np.clip(reflected, self.f_bounds[0], self.f_bounds[1])
                f_reflected = self.func(reflected.reshape(1, -1))[0]
                evals_used += 1
                if f_reflected < simplex_fitness[sorted_idx[0]]:
                    expanded = 3 * centroid - 2 * simplex[worst_idx]
                    expanded = np.clip(expanded, self.f_bounds[0], self.f_bounds[1])
                    f_expanded = self.func(expanded.reshape(1, -1))[0]
                    evals_used += 1
                    if f_expanded < f_reflected:
                        simplex[worst_idx] = expanded
                        simplex_fitness[worst_idx] = f_expanded
                    else:
                        simplex[worst_idx] = reflected
                        simplex_fitness[worst_idx] = f_reflected
                else:
                    contracted = 0.5 * (centroid + simplex[worst_idx])
                    contracted = np.clip(contracted, self.f_bounds[0], self.f_bounds[1])
                    f_contracted = self.func(contracted.reshape(1, -1))[0]
                    evals_used += 1
                    if f_contracted < simplex_fitness[worst_idx]:
                        simplex[worst_idx] = contracted
                        simplex_fitness[worst_idx] = f_contracted
            best_local_idx = np.argmin(simplex_fitness)
            if simplex_fitness[best_local_idx] < f_x:
                self.population[idx] = simplex[best_local_idx]
                self.fitness[idx] = simplex_fitness[best_local_idx]
        return self.population, self.fitness, evals_used
    
    def _compute_diversity(self):
        pop_centered = self.population - np.mean(self.population, axis=0)
        dim_stds = np.std(pop_centered, axis=0)
        dim_means = np.abs(np.mean(self.population, axis=0))
        dim_means = np.where(dim_means < 1e-10, 1e-10, dim_means)
        cv_per_dim = dim_stds / dim_means
        return np.mean(cv_per_dim)
    
    def _opposition_jump(self):
        best = self.population[np.argmin(self.fitness)]
        jump_ratio = 0.3 + 0.2 * np.random.random()
        n_jump = max(1, int(0.2 * self.np))
        jump_indices = np.random.choice(self.np, n_jump, replace=False)
        for idx in jump_indices:
            opp = self.f_bounds[0] + self.f_bounds[1] - self.population[idx]
            self.population[idx] = best + jump_ratio * (opp - best)
        self.population = np.clip(self.population, self.f_bounds[0], self.f_bounds[1])
        self.fitness = self._batch_evaluate(self.population)
    
    def _restart_if_stagnant(self):
        best = self.population[np.argmin(self.fitness)]
        n_elite = max(2, int(0.1 * self.np))
        elite_indices = np.argsort(self.fitness)[:n_elite]
        elite = self.population[elite_indices].copy()
        new_pop = np.random.uniform(self.f_bounds[0], self.f_bounds[1], (self.np, self.dim))
        for i in range(n_elite):
            new_pop[i] = elite[i]
        combined = np.vstack([new_pop, self.f_bounds[0] + self.f_bounds[1] - new_pop])
        combined_fit = self._batch_evaluate(combined)
        indices = np.argsort(combined_fit)[:self.np]
        self.population = combined[indices]
        self.fitness = self._batch_evaluate(self.population)
        self.stagnation_counter = 0
    
    def _extract_best(self):
        valid_mask = ~np.isnan(self.fitness)
        if not np.any(valid_mask):
            return np.nan, np.zeros(self.dim)
        valid_fitness = self.fitness[valid_mask]
        valid_pop = self.population[valid_mask]
        best_idx = np.argmin(valid_fitness)
        return valid_fitness[best_idx], valid_pop[best_idx].copy()
