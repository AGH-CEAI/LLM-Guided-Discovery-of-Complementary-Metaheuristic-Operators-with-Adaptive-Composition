```python
import numpy as np


class CulturalMemeticDifferentialEvolution:
    """
    Hybrid optimizer combining:
    - DE-style mutation guided by a cultural belief space
    - Multiple mutation strategies selected adaptively
    - Nelder-Mead local search on elite individuals
    - Age-based population management
    - Success-history adaptive parameter control
    """

    def __init__(self, pop_size=100, dim=30, bounds=(-100, 100), seed=None):
        self.pop_size = max(10, int(pop_size))
        self.dim = dim
        self.bounds = bounds
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func, stopping_condition):
        self._initialize_attributes(func)
        self._initialize_population()
        self._initialize_cultural_knowledge()
        self._evaluate_initial_population()

        while not stopping_condition():
            self._evolve_one_generation()
            self._adapt_parameters()
            self._update_cultural_knowledge()
            self._manage_archive()
            self._restart_if_stagnant()

        return self.best_fitness, self.best_position.copy()

    # ─── Initialization ───────────────────────────────────────────────────────

    def _initialize_attributes(self, func):
        self.func = func
        self.lower = np.full(self.dim, self.bounds[0])
        self.upper = np.full(self.dim, self.bounds[1])
        self.fitness_evals = 0
        self.gen = 0
        self.best_fitness = np.inf
        self.best_position = None
        self.ages = None
        self.success_history_F = []
        self.success_history_CR = []
        self.archive_best = []
        self.archive_size = 5
        self.local_search_counter = 0
        self.local_search_period = 10
        self.stagnation_gen = 0

    def _initialize_population(self):
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.pop_size, self.dim)
        )
        self._clip_to_bounds(self.population)
        self.fitness = np.full(self.pop_size, np.inf)
        self.mutation_strategy = np.random.randint(0, 3, size=self.pop_size)
        self.strategy_ages = np.zeros(self.pop_size)
        self.strategy_success = np.zeros((self.pop_size, 3))

    def _initialize_cultural_knowledge(self):
        self.cultural_belief = {
            'norms_lower': self.lower.copy(),
            'norms_upper': self.upper.copy(),
            'elite_history': [],
            'successful_directions': np.zeros((5, self.dim)),
            'direction_weights': np.ones(5),
            'knowledge_size': 20
        }

    def _evaluate_initial_population(self):
        for i in range(self.pop_size):
            self.fitness[i] = self.func(self.population[i:i+1])[0]
        self.fitness_evals += self.pop_size
        self._update_best_solution()

    # ─── Core evolutionary operators ─────────────────────────────────────────

    def _evolve_one_generation(self):
        self.gen += 1
        trial_population = np.empty_like(self.population)
        trial_fitness = np.empty(self.pop_size)

        for i in range(self.pop_size):
            mutant = self._cultural_mutation(i)
            trial = self._binomial_recombine(i, mutant)
            self._clip_to_bounds(trial)
            trial_fitness[i] = self.func(trial.reshape(1, -1))[0]
            self.fitness_evals += 1
            trial_population[i] = trial

        self._select_survivors(trial_population, trial_fitness)
        self._update_best_solution()
        self._apply_local_search_to_elite()

    def _cultural_mutation(self, idx):
        strategy = self.mutation_strategy[idx]
        f = self._get_current_F()
        population = self.population

        if strategy == 0:
            mutant = self._DE_current_to_best(idx, f, population)
        elif strategy == 1:
            mutant = self._cultural_guidance_mutation(idx, f, population)
        else:
            mutant = self._hybrid_mutation(idx, f, population)

        mutant = self._add_cultural_bias(mutant, idx)
        return mutant

    def _DE_current_to_best(self, idx, f, pop):
        best_idx = np.argmin(self.fitness)
        r1, r2 = self._select_random_indices(2, exclude=[idx, best_idx])
        mutant = pop[idx] + f * (pop[best_idx] - pop[idx]) + f * (pop[r1] - pop[r2])
        return mutant

    def _cultural_guidance_mutation(self, idx, f, pop):
        n_elites = len(self.cultural_belief['elite_history'])
        if n_elites >= 2:
            elite_indices = np.random.choice(
                n_elites, size=min(2, n_elites), replace=False
            )
            elite_avg = np.mean(
                [self.cultural_belief['elite_history'][e] for e in elite_indices],
                axis=0
            )
            r1, r2 = self._select_random_indices(2, exclude=[idx])
            mutant = pop[idx] + f * (elite_avg - pop[idx]) + f * (pop[r1] - pop[r2])
        else:
            mutant = self._DE_current_to_best(idx, f, pop)
        return mutant

    def _hybrid_mutation(self, idx, f, pop):
        r1, r2, r3 = self._select_random_indices(3, exclude=[idx])
        mutant = pop[r1] + f * (pop[r2] - pop[r3])

        weighted_dir = self._compute_weighted_direction()
        if weighted_dir is not None and np.random.rand() < 0.3:
            mutant += 0.2 * f * weighted_dir

        return mutant

    def _compute_weighted_direction(self):
        directions = self.cultural_belief['successful_directions']
        weights = self.cultural_belief['direction_weights']
        if np.sum(weights) > 0:
            normalized_weights = weights / np.sum(weights)
            weighted = np.sum(directions * normalized_weights[:, np.newaxis], axis=0)
            norm = np.linalg.norm(weighted)
            if norm > 1e-10:
                return weighted / norm
        return None

    def _add_cultural_bias(self, mutant, idx):
        bias_strength = 0.1
        if len(self.cultural_belief['elite_history']) > 0:
            elite = self.cultural_belief['elite_history'][
                np.random.randint(len(self.cultural_belief['elite_history']))
            ]
            bias = np.random.rand() * bias_strength
            mutant = mutant + bias * (elite - mutant)
        return mutant

    def _select_random_indices(self, count, exclude=None):
        exclude = set(exclude) if exclude else set()
        candidates = [i for i in range(self.pop_size) if i not in exclude]
        return list(np.random.choice(candidates, size=min(count, len(candidates)),
                                      replace=False))

    def _binomial_recombine(self, idx, mutant):
        cr = self._get_current_CR()
        trial = self.population[idx].copy()
        d = np.random.randint(self.dim)
        for j in range(self.dim):
            if j == d or np.random.rand() < cr:
                trial[j] = mutant[j]
        return trial

    def _select_survivors(self, trial_pop, trial_fitness):
        for i in range(self.pop_size):
            if trial_fitness[i] <= self.fitness[i]:
                self.fitness[i] = trial_fitness[i]
                self.population[i] = trial_pop[i]
                self.ages[i] = 0
                self.strategy_ages[i] += 1
                self.strategy_success[i, self.mutation_strategy[i]] += 1
            else:
                self.ages[i] += 1
                self._maybe_change_strategy(i)

    def _maybe_change_strategy(self, idx):
        self.strategy_ages[idx] += 1
        if self.strategy_ages[idx] > 5:
            current = self.mutation_strategy[idx]
            other_strategies = [s for s in range(3) if s != current]
            new_strategy = np.random.choice(other_strategies)
            self.mutation_strategy[idx] = new_strategy
            self.strategy_ages[idx] = 0

    # ─── Local search ─────────────────────────────────────────────────────────

    def _apply_local_search_to_elite(self):
        self.local_search_counter += 1
        if self.local_search_counter >= self.local_search_period:
            self.local_search_counter = 0
            elite_idx = np.argmin(self.fitness)
            improved, nm_solution = self._nelder_mead_refine(
                self.population[elite_idx], self.fitness[elite_idx]
            )
            if improved:
                self.population[elite_idx] = nm_solution
                self.fitness[elite_idx] = self.func(
                    nm_solution.reshape(1, -1)
                )[0]
                self.fitness_evals += 1

    def _nelder_mead_refine(self, x_start, f_start, alpha=1.0, gamma=2.0,
                            rho=0.5, sigma=0.5, max_iter=20):
        n = len(x_start)
        simplex = np.empty((n + 1, n))
        simplex[0] = x_start
        for i in range(1, n + 1):
            simplex[i] = x_start + np.random.uniform(-0.5, 0.5, n)
        self._clip_to_bounds(simplex)

        values = np.array([self.func(x.reshape(1, -1))[0] for x in simplex])
        self.fitness_evals += n + 1

        for _ in range(max_iter):
            sorted_idx = np.argsort(values)
            if values[sorted_idx[0]] < f_start - 1e-8:
                return True, simplex[sorted_idx[0]]

            worst = sorted_idx[-1]
            second_worst = sorted_idx[-2]
            centroid = np.mean(simplex[sorted_idx[:-1]], axis=0)

            reflected = centroid + alpha * (centroid - simplex[worst])
            reflected = np.clip(reflected, self.lower, self.upper)
            f_reflected = self.func(reflected.reshape(1, -1))[0]
            self.fitness_evals += 1

            if f_reflected < values[sorted_idx[0]]:
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, self.lower, self.upper)
                f_expanded = self.func(expanded.reshape(1, -1))[0]
                self.fitness_evals += 1
                if f_expanded < f_reflected:
                    simplex[worst] = expanded
                    values[worst] = f_expanded
                else:
                    simplex[worst] = reflected
                    values[worst] = f_reflected
            elif f_reflected < values[second_worst]:
                simplex[worst] = reflected
                values[worst] = f_reflected
            else:
                contracted = centroid - rho * (centroid - simplex[worst])
                contracted = np.clip(contracted, self.lower, self.upper)
                f_contracted = self.func(contracted.reshape(1, -1))[0]
                self.fitness_evals += 1
                if f_contracted < values[worst]:
                    simplex[worst] = contracted
                    values[worst] = f_contracted
                else:
                    for j in sorted_idx[1:]:
                        simplex[j] = simplex[sorted_idx[0]] + \
                                     sigma * (simplex[j] - simplex[sorted_idx[0]])
                        simplex[j] = np.clip(simplex[j], self.lower, self.upper)
                        values[j] = self.func(simplex[j].reshape(1, -1))[0]
                        self.fitness_evals += 1

        best_idx = np.argmin(values)
        return values[best_idx] < f_start - 1e-8, simplex[best_idx]

    # ─── Parameter adaptation ─────────────────────────────────────────────────

    def _adapt_parameters(self):
        self._update_success_history()
        self.F = self._sample_adaptive_F()
        self.CR = self._sample_adaptive_CR()

    def _update_success_history(self):
        if self.gen > 0:
            if len(self.success_history_F) < 5:
                self.success_history_F.append(self.F)
            else:
                self.success_history_F.pop(0)
                self.success_history_F.append(self.F)

    def _get_current_F(self):
        return getattr(self, 'F', 0.5)

    def _get_current_CR(self):
        return getattr(self, 'CR', 0.7)

    def _sample_adaptive_F(self):
        if len(self.success_history_F) >= 3:
            mu_f = np.mean(self.success_history_F[-3:])
            return np.clip(mu_f + 0.1 * np.random.randn(), 0.1, 1.0)
        return np.clip(0.5 + 0.3 * np.random.randn(), 0.1, 1.0)

    def _sample_adaptive_CR(self):
        return np.clip(0.7 + 0.2 * np.random.randn(), 0.0, 1.0)

    # ─── Cultural knowledge management ────────────────────────────────────────

    def _update_cultural_knowledge(self):
        best_idx = np.argmin(self.fitness)
        best_pos = self.population[best_idx].copy()
        best_fit = self.fitness[best_idx]

        if best_fit < self.best_fitness:
            if len(self.cultural_belief['successful_directions']) > 0:
                last_best = self.cultural_belief['elite_history'][0] \
                    if self.cultural_belief['elite_history'] else best_pos
                direction = best_pos - last_best
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    direction = direction / direction_norm
                    self.cultural_belief['successful_directions'] = np.roll(
                        self.cultural_belief['successful_directions'], -1, axis=0
                    )
                    self.cultural_belief['successful_directions'][-1] = direction

        self.cultural_belief['elite_history'].insert(0, best_pos)
        max_knowledge = self.cultural_belief['knowledge_size']
        self.cultural_belief['elite_history'] = \
            self.cultural_belief['elite_history'][:max_knowledge]

        self.cultural_belief['norms_lower'] = np.minimum(
            self.cultural_belief['norms_lower'],
            np.min(self.population, axis=0)
        )
        self.cultural_belief['norms_upper'] = np.maximum(
            self.cultural_belief['norms_upper'],
            np.max(self.population, axis=0)
        )

    # ─── Archive management ───────────────────────────────────────────────────

    def _manage_archive(self):
        self.archive_best.append(self.best_position.copy())
        if len(self.archive_best) > self.archive_size:
            self.archive_best.pop(0)

    # ─── Restart and diversity ────────────────────────────────────────────────

    def _restart_if_stagnant(self):
        diversity = self._compute_diversity()
        if diversity < 0.01 * (self.bounds[1] - self.bounds[0]):
            self.stagnation_gen += 1
        else:
            self.stagnation_gen = 0

        if self.stagnation_gen > 20 or self.gen > 500:
            self._inject_diverse_individuals()

    def _compute_diversity(self):
        if self.population.shape[0] < 2:
            return 0.0
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)

    def _inject_diverse_individuals(self):
        inject_count = max(5, self.pop_size // 10)
        inject_indices = np.random.choice(
            self.pop_size, size=inject_count, replace=False
        )
        for idx in inject_indices:
            self.population[idx] = self._generate_diverse_candidate()
            self._clip_to_bounds(self.population[idx:idx+1])
            self.fitness[idx] = self.func(self.population[idx:idx+1])[0]
            self.fitness_evals += 1
            self.ages[idx] = 0
        self.stagnation_gen = 0

    def _generate_diverse_candidate(self):
        if len(self.cultural_belief['elite_history']) > 0:
            elite = self.cultural_belief['elite_history'][
                np.random.randint(len(self.cultural_belief['elite_history']))
            ]
            candidate = elite + np.random.uniform(-50, 50, self.dim)
            return candidate
        return np.random.uniform(self.lower, self.upper)

    # ─── Utility ──────────────────────────────────────────────────────────────

    def _clip_to_bounds(self, population):
        np.clip(population, self.lower, self.upper, out=population)

    def _update_best_solution(self):
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_position = self.population[min_idx].copy()
```