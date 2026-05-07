```python
import numpy as np


class AdaptiveMemeticDifferentialEvolution:
    """
    Hybrid optimizer combining DE-style mutation with strategic Nelder-Mead
    local search, performance-weighted strategy selection, and diversity-triggered
    restarts. Uses a random directed graph topology for mutation targets.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = min(max(4 * dim, 20), min(10 * dim, 300))
        self.bounds = np.array([-100.0, 100.0])
        self._init_state()

    def _init_state(self):
        """Initialize all mutable algorithm state."""
        self.F = 0.7
        self.CR = 0.85
        self.min_F = 0.1
        self.max_F = 1.5
        self.min_CR = 0.0
        self.max_CR = 1.0
        self.generation = 0
        self.stagnation_gen = 0
        self.last_improvement_gen = 0
        self.local_search_interval = 8
        self.diversity_threshold = 1e-6
        self.adaptive_params = {
            'F_history': [0.7],
            'CR_history': [0.85],
            'F_success': [],
            'CR_success': [],
        }
        self.strategies = ['rand_1', 'best_1', 'target_to_best_1', 'rand_2']
        self.strategy_weights = np.ones(len(self.strategies))
        self.strategy_success = np.zeros(len(self.strategies))
        self.current_strategy_idx = 0
        self.neighborhood_size = max(3, self.pop_size // 10)
        self._build_neighborhood()

    def _build_neighborhood(self):
        """Create random directed graph neighborhood for each individual."""
        np.random.seed(None)
        n = self.pop_size
        k = self.neighborhood_size
        self.neighbors = np.zeros((n, k), dtype=int)
        for i in range(n):
            candidates = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
            self.neighbors[i] = np.random.choice(candidates, size=k, replace=False)

    def _initialize_population(self):
        """Generate initial population with Latin Hypercube Sampling."""
        pop = np.zeros((self.pop_size, self.dim))
        for d in range(self.dim):
            pop[:, d] = np.random.uniform(self.bounds[0], self.bounds[1], self.pop_size)
            indices = np.random.permutation(self.pop_size)
            offsets = np.random.uniform(0, 1, self.pop_size)
            pop[:, d] = self.bounds[0] + (indices + offsets) / self.pop_size * (self.bounds[1] - self.bounds[0])
        return np.clip(pop, self.bounds[0], self.bounds[1])

    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batched call."""
        fitness = func(population)
        if len(fitness) < len(population):
            return fitness[:len(population)]
        return fitness

    def _select_mutation_strategy(self):
        """Select mutation strategy based on weighted performance history."""
        weights = self.strategy_weights.copy()
        weights = weights / weights.sum()
        self.current_strategy_idx = np.random.choice(len(self.strategies), p=weights)
        return self.strategies[self.current_strategy_idx]

    def _mutate_batch(self, population, fitness):
        """Generate mutant vectors using current strategy with neighborhood."""
        n = self.pop_size
        dim = self.dim
        strategy = self._select_mutation_strategy()
        mutants = np.empty_like(population)
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        target_idx = np.arange(n)
        np.random.shuffle(target_idx)

        if strategy == 'rand_1':
            for i in range(n):
                r = self.neighbors[i]
                mutants[i] = population[r[0]] + self.F * (population[r[1]] - population[r[2]])

        elif strategy == 'best_1':
            for i in range(n):
                r = self.neighbors[i]
                mutants[i] = population[best_idx] + self.F * (population[r[0]] - population[r[1]])

        elif strategy == 'target_to_best_1':
            for i in range(n):
                r = self.neighbors[i]
                mutants[i] = population[i] + self.F * (population[best_idx] - population[i]) + \
                             self.F * (population[r[0]] - population[r[1]])

        elif strategy == 'rand_2':
            for i in range(n):
                r = self.neighbors[i]
                mutants[i] = population[r[0]] + self.F * (population[r[1]] - population[r[2]]) + \
                             self.F * (population[r[3] if len(r) > 3 else r[0]] - population[r[4] if len(r) > 4 else r[1]])

        mutants = np.clip(mutants, self.bounds[0], self.bounds[1])
        return mutants

    def _crossover_batch(self, population, mutants):
        """Binomial crossover with adaptive CR."""
        n, dim = population.shape
        trials = np.empty_like(population)
        j_random = np.random.randint(0, dim, n)

        for i in range(n):
            j = j_random[i]
            mask = np.random.random(dim) < self.CR
            mask[j] = True
            trials[i] = np.where(mask, mutants[i], population[i])

        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors using greedy (mu + lambda) selection."""
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        sorted_idx = np.argsort(combined_fit)[:self.pop_size]
        return combined_pop[sorted_idx], combined_fit[sorted_idx]

    def _adapt_parameters(self, fitness):
        """Adapt F and CR based on recent improvements."""
        current_best = fitness[0]
        history = self.best_fitness_history

        if len(history) > 0 and current_best < history[-1]:
            self.stagnation_gen = 0
            self.last_improvement_gen = self.generation
            if len(self.adaptive_params['F_success']) > 0:
                self.adaptive_params['F_history'].append(np.mean(self.adaptive_params['F_success']))
                self.adaptive_params['CR_history'].append(np.mean(self.adaptive_params['CR_success']))
                self.adaptive_params['F_success'] = []
                self.adaptive_params['CR_success'] = []

        self.stagnation_gen += 1

        if len(self.adaptive_params['F_history']) > 0:
            base_F = self.adaptive_params['F_history'][-1]
            base_CR = self.adaptive_params['CR_history'][-1]
        else:
            base_F = 0.7
            base_CR = 0.85

        self.F = np.clip(base_F + np.random.normal(0, 0.1), self.min_F, self.max_F)
        self.CR = np.clip(base_CR + np.random.normal(0, 0.1), self.min_CR, self.max_CR)

    def _update_strategy_scores(self, improved_mask):
        """Update strategy weights based on success rate."""
        if improved_mask.any():
            self.strategy_success[self.current_strategy_idx] += improved_mask.sum()
            self.strategy_weights[self.current_strategy_idx] *= 1.1
        else:
            self.strategy_weights[self.current_strategy_idx] *= 0.9

        self.strategy_weights = np.clip(self.strategy_weights, 0.1, 10.0)
        self.strategy_weights = self.strategy_weights / self.strategy_weights.sum()

    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        if len(population) < 2:
            return 1.0
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)

    def _apply_local_search_batch(self, population, fitness, func):
        """Apply Nelder-Mead refinement to top candidates in batch."""
        n_top = max(1, self.pop_size // 10)
        top_indices = np.argsort(fitness)[:n_top]
        refined_pop = population.copy()
        refined_fit = fitness.copy()
        alpha = 1.0
        gamma = 2.0
        sigma = 0.5
        rho = 0.5

        for idx in top_indices:
            x = population[idx].copy()
            simplex = np.array([x + np.random.uniform(-0.1, 0.1, self.dim) for _ in range(self.dim + 1)])
            simplex = np.vstack([x, simplex])
            simplex = np.clip(simplex, self.bounds[0], self.bounds[1])

            for _ in range(20):
                fitness_vals = func(simplex)
                if len(fitness_vals) < len(simplex):
                    fitness_vals = np.pad(fitness_vals, (0, len(simplex) - len(fitness_vals)), constant_values=np.inf)

                sorted_idx = np.argsort(fitness_vals)
                simplex = simplex[sorted_idx]
                fitness_vals = fitness_vals[sorted_idx]

                centroid = np.mean(simplex[:-1], axis=0)
                xo = centroid
                xr = xo + alpha * (xo - simplex[-1])
                xr = np.clip(xr, self.bounds[0], self.bounds[1])

                f_xr = func(xr.reshape(1, -1))[0] if len(func(xr.reshape(1, -1))) > 0 else np.inf

                if f_xr < fitness_vals[0]:
                    xe = xo + gamma * (xr - xo)
                    xe = np.clip(xe, self.bounds[0], self.bounds[1])
                    f_xe = func(xe.reshape(1, -1))[0] if len(func(xe.reshape(1, -1))) > 0 else np.inf
                    simplex[-1] = xe if f_xe < f_xr else xr
                elif f_xr < fitness_vals[-2]:
                    simplex[-1] = xr
                else:
                    xc = xo + rho * (simplex[-1] - xo)
                    xc = np.clip(xc, self.bounds[0], self.bounds[1])
                    simplex[-1] = xc

                if fitness_vals[-1] < fitness[-1] if idx == top_indices[0] else True:
                    break

            if fitness_vals[0] < fitness[idx]:
                refined_pop[idx] = simplex[0]
                refined_fit[idx] = fitness_vals[0]

        return refined_pop, refined_fit

    def _restart_if_needed(self, population, fitness):
        """Check stagnation and diversity, trigger restart if needed."""
        should_restart = False
        restart_reason = ""

        diversity = self._compute_diversity(population)
        if diversity < self.diversity_threshold:
            should_restart = True
            restart_reason = "low_diversity"

        if self.stagnation_gen > 50 + self.dim:
            should_restart = True
            restart_reason = "stagnation"

        if should_restart:
            n_elite = max(1, self.pop_size // 5)
            elite_idx = np.argsort(fitness)[:n_elite]
            elite_pop = population[elite_idx].copy()
            elite_fit = fitness[elite_idx].copy()

            new_pop = self._initialize_population()
            new_fit = self._evaluate_batch(new_pop, func) if 'func' in dir() else np.full(self.pop_size, np.inf)

            for i in range(n_elite):
                new_pop[i] = elite_pop[i]
                new_fit[i] = elite_fit[i]

            self.stagnation_gen = 0
            self._build_neighborhood()
            return new_pop, new_fit, restart_reason

        return population, fitness, None

    def _record_improvement(self, old_fitness, new_fitness):
        """Record parameter adaptation on improvement."""
        if new_fitness[0] < old_fitness[0]:
            self.adaptive_params['F_success'].append(self.F)
            self.adaptive_params['CR_success'].append(self.CR)

    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population = self._initialize_population()
        fitness = self._evaluate_batch(population, func)

        if len(fitness) < self.pop_size:
            fitness = np.pad(fitness, (0, self.pop_size - len(fitness)), constant_values=np.inf)

        self.best_fitness_history = [fitness[0]]
        self.best_idx = 0
        self.best_fitness = fitness[0]

        while not stopping_condition():
            old_fitness = fitness.copy()
            old_best = fitness[0]

            mutants = self._mutate_batch(population, fitness)
            trials = self._crossover_batch(population, mutants)
            trials = np.clip(trials, self.bounds[0], self.bounds[1])

            if stopping_condition():
                break

            trial_fitness = self._evaluate_batch(trials, func)
            if len(trial_fitness) < len(trials):
                valid_mask = np.arange(len(trials)) < len(trial_fitness)
                trials = trials[valid_mask]
                trial_fitness = trial_fitness[:len(trials)]

            if len(trial_fitness) == 0:
                break

            improved_mask = trial_fitness < fitness
            if improved_mask.any():
                self._update_strategy_scores(improved_mask)

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            self._record_improvement(old_fitness, fitness)
            self._adapt_parameters(fitness)

            if self.generation - self.last_improvement_gen >= self.local_search_interval:
                if len(fitness) >= self.pop_size:
                    population, fitness = self._apply_local_search_batch(population, fitness, func)
                    self.last_improvement_gen = self.generation

            population, fitness, restart_reason = self._restart_if_needed(population, fitness)

            if fitness[0] < old_best:
                self.best_idx = np.argmin(fitness)
                self.best_fitness = fitness[self.best_idx]

            self.best_fitness_history.append(self.best_fitness)
            self.generation += 1

            if len(fitness) < self.pop_size:
                fitness = np.pad(fitness, (0, self.pop_size - len(fitness)), constant_values=np.inf)

        best_local_idx = np.argmin(fitness)
        return fitness[best_local_idx], population[best_local_idx]
```