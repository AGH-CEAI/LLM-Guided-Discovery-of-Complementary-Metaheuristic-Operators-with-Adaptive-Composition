```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive version that automatically selects the best mutation strategy
    using Thompson Sampling with sliding window credit assignment.
    
    Combines:
    - original.py (default operator)
    - variant_01_idea_0.py (JADE-style with archive)
    - variant_07_idea_0.py (global best with stagnation adaptation)
    - variant_09_idea_0.py (improvement-ratio adaptive F)
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

        # === ADAPTIVE OPERATOR SELECTION (Multi-Armed Bandit) ===
        self.num_operators = 4
        self.operator_names = ['original', 'jade_archive', 'global_best', 'improvement_adaptive']
        
        # Sliding window for rewards (sliding_window_size recent rewards per operator)
        self.sliding_window_size = 50
        self.operator_rewards = [[] for _ in range(self.num_operators)]
        
        # Cumulative statistics for Thompson Sampling
        self.operator_successes = np.zeros(self.num_operators, dtype=float)
        self.operator_trials = np.zeros(self.num_operators, dtype=float)
        self.operator_total_rewards = np.zeros(self.num_operators, dtype=float)
        
        # Exploration bonus and initialization
        self.exploration_bonus = 0.1
        self._init_operator_stats()

        # Archive for JADE-style mutation
        self.archive = None
        self._pending_archive_additions = None

    def _init_operator_stats(self):
        """Initialize operator statistics with optimistic priors."""
        for i in range(self.num_operators):
            self.operator_successes[i] = 5.0
            self.operator_trials[i] = 10.0
            self.operator_total_rewards[i] = 2.0

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        # Filter out invalid fitness values
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return float('nan'), np.zeros(self.dim)
        population = population[valid_mask]
        fitness = fitness[valid_mask]

        best_idx = np.argmin(fitness)
        self.best_fitness = float(np.min(fitness))
        self.best_solution = population[best_idx].copy()

        while not stopping_condition():
            # Select mutation operator using Thompson Sampling
            selected_op = self._select_mutation_operator_thompson()
            
            # Store fitness before mutation for reward calculation
            fitness_before = fitness.copy()

            mutants = self._apply_mutation_operator(selected_op, population, fitness)
            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                fitness_before = fitness_before[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)

            best_idx = np.argmin(fitness)
            current_best = float(np.min(fitness))
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Update archive if using JADE operator
            if selected_op == 1:
                self._update_jade_archive(population, fitness, trials, trial_fitness)

            # Update operator statistics with reward
            improved_mask = np.isfinite(trial_fitness) & np.isfinite(fitness) & (trial_fitness < fitness)
            self._update_operator_stats(selected_op, improved_mask, fitness_before, trial_fitness)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return float(self.best_fitness), np.array(self.best_solution)

    def _select_mutation_operator_thompson(self):
        """Thompson Sampling: sample from posterior Beta distributions."""
        samples = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            alpha = max(self.operator_successes[i], 1e-10)
            beta_param = max(self.operator_trials[i] - self.operator_successes[i], 1e-10)
            samples[i] = np.random.beta(alpha, beta_param)
        
        # Add small exploration bonus to under-sampled operators
        for i in range(self.num_operators):
            if self.operator_trials[i] < 5:
                samples[i] += self.exploration_bonus * (5 - self.operator_trials[i]) / 5
        
        return int(np.argmax(samples))

    def _apply_mutation_operator(self, op_idx, population, fitness):
        """Dispatch to the selected mutation operator."""
        if op_idx == 0:
            return self._mutate_original(population, fitness)
        elif op_idx == 1:
            return self._mutate_jade_archive(population, fitness)
        elif op_idx == 2:
            return self._mutate_global_best(population, fitness)
        elif op_idx == 3:
            return self._mutate_improvement_adaptive(population, fitness)
        else:
            return self._mutate_original(population, fitness)

    def _mutate_original(self, population, fitness):
        """Original mutation strategy."""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        base = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return donors

    def _mutate_jade_archive(self, population, fitness):
        """JADE-style mutation with archive and per-individual F."""
        if self.archive is None:
            self.archive = np.zeros((0, self.dim))

        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        p_best_indices = sorted_idx[:n_best]
        p_best_idx = p_best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        r1 = np.random.randint(0, self.NP, size=self.NP)
        r2 = np.zeros(self.NP, dtype=int)
        for i in range(self.NP):
            while True:
                r2[i] = np.random.randint(0, self.NP)
                if r2[i] != i and r2[i] != r1[i]:
                    break

        archive_size = len(self.archive)
        total_size = self.NP + archive_size

        if archive_size > 0:
            all_indices = np.concatenate([np.arange(self.NP), self.archive])
            r3_indices = np.random.randint(0, total_size, size=self.NP)
            r3 = np.where(r3_indices < self.NP, r3_indices, -(r3_indices - self.NP + 1))
            for i in range(self.NP):
                if r3[i] == i:
                    if r3[i] < self.NP - 1:
                        r3[i] += 1
                    else:
                        r3[i] = 0
            r3_pop = np.where(r3 >= 0, population[r3], self.archive[-(r3 + 1)])
        else:
            r3 = np.random.randint(0, self.NP, size=self.NP)
            for i in range(self.NP):
                while r3[i] == i or r3[i] == r1[i]:
                    r3[i] = (r3[i] + 1) % self.NP
            r3_pop = population[r3]

        F_i = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)
        donors = population + F_i[:, np.newaxis] * (p_best - population) + F_i[:, np.newaxis] * (r3_pop - population[r1])
        donors = np.clip(donors, self.lower, self.upper)

        return donors

    def _update_jade_archive(self, population, fitness, trials, trial_fitness):
        """Update JADE archive with rejected solutions."""
        if self.archive is None:
            self.archive = np.zeros((0, self.dim))
        
        rejected_mask = np.isfinite(trial_fitness) & (trial_fitness >= fitness)
        if np.any(rejected_mask):
            rejected_solutions = trials[rejected_mask]
            self._pending_archive_additions = rejected_solutions
            
            max_archive_size = self.NP * 5
            if len(self.archive) + len(rejected_solutions) > max_archive_size:
                remove_count = len(self.archive) + len(rejected_solutions) - max_archive_size
                self.archive = self.archive[remove_count:]
            
            self.archive = np.vstack([self.archive, rejected_solutions])

    def _mutate_global_best(self, population, fitness):
        """Global best with stagnation adaptation."""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        global_best_idx = best_indices[0]
        global_best = population[global_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        F_adapted = self.F
        if self.stagnation_count > 10:
            F_adapted = min(self.F * (1.0 + 0.2 * (self.stagnation_count - 10)), 2.0)

        base = population + F_adapted * (global_best - population) + F_adapted * (population[r1] - population[r2])

        if self.stagnation_count > 5:
            perturbation_scale = min(0.3 * np.log1p(self.stagnation_count), 2.0)
            random_perturbation = np.random.randn(self.NP, self.dim) * perturbation_scale
            base = base + random_perturbation

        if self.generation > 5 and np.any(self.cultural_weights > 0):
            weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
            cultural_contribution = np.dot(weights_norm, self.cultural_memory)
            similarity = np.array([np.linalg.norm(population[i] - cultural_contribution)
                                   for i in range(self.NP)])
            sim_std = np.std(similarity) + 1e-10
            weights_culture = np.exp(-similarity / sim_std)
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return donors

    def _mutate_improvement_adaptive(self, population, fitness):
        """Improvement-ratio adaptive F with historical best."""
        sorted_idx = np.argsort(fitness)
        n_best = max(1, int(self.NP * self.p_best_rate))
        best_indices = sorted_idx[:n_best]
        p_best_idx = best_indices[np.random.randint(0, n_best, size=self.NP)]
        p_best = population[p_best_idx]

        idx_a = np.random.randint(0, self.NP, (self.NP, 3))
        idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                                (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
        idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                               (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
        r1, r2 = idx_a[:, 1], idx_a[:, 2]

        if np.isfinite(self.best_fitness) and self.generation > 5:
            current_best = float(np.min(fitness))
            improvement_ratio = (current_best - self.best_fitness) / (abs(self.best_fitness) + 1e-30)

            F_escape = np.clip(self.F * (1.0 + max(0, -improvement_ratio) * 0.5), 0.5, 2.0)
            w_hist = np.clip(0.7 + max(0, -improvement_ratio) * 0.2, 0.5, 0.9)
            w_curr = 1.0 - w_hist

            if np.any(self.cultural_weights > 0):
                weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
                cultural_centroid = np.dot(weights_norm, self.cultural_memory)
                target = w_hist * self.best_solution + w_curr * p_best + 0.1 * cultural_centroid
            else:
                target = w_hist * self.best_solution + w_curr * p_best

            donors = population + F_escape * (target - population) + self.F * (population[r1] - population[r2])
        else:
            donors = population + self.F * (p_best - population) + self.F * (population[r1] - population[r2])

        return donors

    def _update_operator_stats(self, operator_idx, improved_mask, fitness_before, fitness_after):
        """Update operator statistics using sliding window credit assignment."""
        valid_improved = improved_mask & np.isfinite(fitness_before) & np.isfinite(fitness_after)
        n_improved = int(np.sum(valid_improved))
        n_total = int(np.sum(np.isfinite(fitness_after)))
        
        if n_improved > 0 and n_total > 0:
            improvements = fitness_before[valid_improved] - fitness_after[valid_improved]
            reward = float(np.mean(improvements))
            reward = max(reward, 0.0)
        else:
            reward = 0.0
        
        self.operator_rewards[operator_idx].append(reward)
        if len(self.operator_rewards[operator_idx]) > self.sliding_window_size:
            self.operator_rewards[operator_idx].pop(0)
        
        if n_improved > 0:
            self.operator_successes[operator_idx] += float(n_improved)
        if n_total > 0:
            self.operator_trials[operator_idx] += float(n_total)
        if reward > 0:
            self.operator_total_rewards[operator_idx] += reward

    def _crossover_binomial_batch(self, population, donors):
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        trials = np.where(mask, donors, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = np.isfinite(trial_fitness) & (trial_fitness < fitness)
        new_population = np.copy(population)
        new_population[better] = trials[better]
        new_fitness = np.copy(fitness)
        new_fitness[better] = trial_fitness[better]
        return new_population, new_fitness

    def _update_cultural_memory_on_improvement(self, best_individual, best_mutant):
        if self.stagnation_count == 0:
            pattern = best_mutant - best_individual
            self.cultural_memory[self.culture_idx] = pattern
            self.cultural_weights[self.culture_idx] = 1.0 / (1.0 + float(self.stagnation_count))
            self.culture_idx = (self.culture_idx + 1) % self.cultural_memory_size
            decay = 0.95
            self.cultural_weights *= decay

    def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        F_candidates = np.random.uniform(0.4, 1.0, size=5)
        self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))

        if np.random.rand() < 0.2:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3))

    def _apply_adaptive_local_search(self, func):
        if self.best_solution is None:
            return

        center = np.array(self.best_solution, copy=True)
        current_fitness = float(self.best_fitness)
        radius = float(self.local_radius)

        for _ in range(self.local_max_iter):
            if radius < 1e-6:
                break

            directions = np.random.randn(self.dim)
            dir_norm = np.linalg.norm(directions) + 1e-10
            directions = directions / dir_norm
            candidates = np.clip(center + radius * directions, self.lower, self.upper)
            candidates = np.vstack([candidates, np.clip(center - radius * directions, self.lower, self.upper)])
            candidates = np.vstack([candidates, center])

            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = int(np.argmin(fit_candidates))
            best_local_fitness = float(fit_candidates[best_local_idx])

            if best_local_fitness < current_fitness:
                center = np.array(candidates[best_local_idx], copy=True)
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
        return float(np.mean(distances))

    def _check_diversity_low(self, population):
        diversity = self._compute_diversity_batch(population)
        return (diversity < self.diversity_threshold or
                self.stagnation_count > self.stagnation_limit)

    def _reinitialize_population(self, population, fitness):
        if self.best_solution is not None:
            best_idx = int(np.argmin(fitness))
            population[0] = np.array(self.best_solution, copy=True)
            fitness[0] = float(self.best_fitness)

        n_random = max(self.min_pop_size, self.NP // 2)
        new_part = self._initialize_population_lhs()[:n_random]
        population[1:1+n_random] = new_part
        fitness[1:1+n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5

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
        return np.array(result, dtype=float)
```