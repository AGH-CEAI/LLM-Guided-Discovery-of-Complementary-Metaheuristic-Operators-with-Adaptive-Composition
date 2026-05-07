import numpy as np


class AdaptiveCulturalMemeticDE:
    """
    Adaptive version of CulturalMemeticDE that automatically selects the best
    mutation strategy using Multi-Armed Bandit with UCB selection and sliding
    window credit assignment.
    
    Operators:
        0: original.py - Standard current-to-pbest with cultural memory
        1: variant_01_idea_0.py - JADE-style with archive
        2: variant_07_idea_0.py - Global best with stagnation adaptation
        3: variant_09_idea_0.py - Adaptive F with historical best weighting
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
        
        # ==== Adaptive Operator Selection (Multi-Armed Bandit) ====
        self.num_operators = 4
        self.operator_rewards = np.zeros(self.num_operators)
        self.operator_counts = np.zeros(self.num_operators)
        self.ucb_c = 1.0
        self.exploration_rate = 0.1
        
        # Sliding window for credit assignment
        self.credit_window = 15
        self.recent_credits = [[] for _ in range(self.num_operators)]
        
        # Track operator used in current generation
        self.current_operator = 0
        
        # Archive for JADE-style operator
        self.archive = None
        
        # Minimum generations before switching operators
        self.min_operator_gens = 3
        self.operator_gen_counter = 0

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(np.asarray(fitness[best_idx]).item())
        self.best_solution = population[best_idx].copy()

        while not stopping_condition():
            # Select operator using UCB
            self.current_operator = self._select_operator_ucb()
            self.operator_gen_counter += 1
            
            mutants = self._mutate_adaptive(population, fitness)
            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            # Store fitness as scalar floats
            fitness_before = np.copy(fitness)
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            
            # Credit assignment for operator selection
            self._assign_credit(fitness_before, fitness)
            
            # Update operator counts and rewards
            self._update_operator_stats()

            best_idx = np.argmin(fitness)
            current_best = float(np.asarray(fitness[best_idx]).item())
            if current_best < self.best_fitness:
                improvement = self.best_fitness - current_best
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
                self.improvement_buffer.append(float(improvement))
                if len(self.improvement_buffer) > self.credit_window:
                    self.improvement_buffer.pop(0)
            else:
                self.stagnation_count += 1

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

    def _select_operator_ucb(self):
        """Select operator using UCB1 with epsilon-greedy fallback."""
        t = np.sum(self.operator_counts) + 1
        ucb_values = np.zeros(self.num_operators)
        
        for i in range(self.num_operators):
            if self.operator_counts[i] > 0:
                exploitation = self.operator_rewards[i] / max(1.0, self.operator_counts[i])
                exploration = np.sqrt(2.0 * np.log(max(1.0, t)) / max(1.0, self.operator_counts[i]))
                ucb_values[i] = exploitation + self.ucb_c * exploration
            else:
                ucb_values[i] = float('inf')
        
        selected = int(np.argmax(ucb_values))
        
        # Force minimum generations per operator
        if self.operator_gen_counter < self.min_operator_gens:
            return self.current_operator
        
        # Epsilon-greedy exploration
        if np.random.rand() < self.exploration_rate:
            selected = np.random.randint(self.num_operators)
        
        return selected

    def _assign_credit(self, fitness_before, fitness_after):
        """Assign credit based on improvement ratio using sliding window."""
        improvements = fitness_before - fitness_after
        valid_improvements = improvements[np.isfinite(improvements) & np.isfinite(fitness_before)]
        
        if len(valid_improvements) == 0:
            return
        
        # Credit based on mean improvement
        mean_improvement = float(np.mean(valid_improvements).item())
        max_improvement = float(np.max(valid_improvements).item())
        
        # Normalize credit
        if max_improvement > 1e-12:
            credit = mean_improvement / (max_improvement + 1e-12)
        else:
            credit = mean_improvement
        
        # Store in sliding window
        self.recent_credits[self.current_operator].append(float(credit))
        if len(self.recent_credits[self.current_operator]) > self.credit_window:
            self.recent_credits[self.current_operator].pop(0)

    def _update_operator_stats(self):
        """Update operator rewards and counts based on recent credits."""
        for i in range(self.num_operators):
            if len(self.recent_credits[i]) > 0:
                # Weighted average with recency bias
                credits = self.recent_credits[i]
                weights = np.exp(np.linspace(-1, 0, len(credits)))
                weights = weights / np.sum(weights)
                weighted_credit = float(np.dot(weights, credits))
                
                self.operator_rewards[i] = 0.9 * self.operator_rewards[i] + 0.1 * weighted_credit
                self.operator_counts[i] += 1.0

    def _mutate_adaptive(self, population, fitness):
        """Dispatch to appropriate mutation strategy."""
        if self.current_operator == 0:
            return self._mutate_original(population, fitness)
        elif self.current_operator == 1:
            return self._mutate_variant_01(population, fitness)
        elif self.current_operator == 2:
            return self._mutate_variant_07(population, fitness)
        elif self.current_operator == 3:
            return self._mutate_variant_09(population, fitness)
        else:
            return self._mutate_original(population, fitness)

    def _mutate_original(self, population, fitness):
        """Operator 0: Standard current-to-pbest with cultural memory."""
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
            weights_culture = np.exp(-similarity / (np.std(similarity) + 1e-10))
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return donors

    def _mutate_variant_01(self, population, fitness):
        """Operator 1: JADE-style with archive."""
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
                r2_i = np.random.randint(0, self.NP)
                if r2_i != i and r2_i != r1[i]:
                    r2[i] = r2_i
                    break

        archive_size = len(self.archive)
        total_size = self.NP + archive_size

        if archive_size > 0:
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

    def _mutate_variant_07(self, population, fitness):
        """Operator 2: Global best with stagnation adaptation."""
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
            weights_culture = np.exp(-similarity / (np.std(similarity) + 1e-10))
            weights_culture = weights_culture / (np.sum(weights_culture) + 1e-10)
            cultural_offset = cultural_contribution * 0.15
            donors = (1 - 0.15) * base + 0.15 * (population + cultural_offset)
        else:
            donors = base

        return donors

    def _mutate_variant_09(self, population, fitness):
        """Operator 3: Adaptive F with historical best weighting."""
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
            current_best = np.min(fitness)
            improvement_ratio = (current_best - self.best_fitness) / (abs(self.best_fitness) + 1e-30)

            F_escape = np.clip(self.F * (1.0 + max(0.0, -improvement_ratio) * 0.5), 0.5, 2.0)
            w_hist = np.clip(0.7 + max(0.0, -improvement_ratio) * 0.2, 0.5, 0.9)
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
        return result

    def _crossover_binomial_batch(self, population, donors):
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        trials = np.where(mask, donors, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
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
        self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5).item())

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95).item())

        dim_adapt = np.random.rand() < 0.2
        if dim_adapt:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3).item())

    def _apply_adaptive_local_search(self, func):
        if self.best_solution is None:
            return

        center = self.best_solution.copy()
        current_fitness = self.best_fitness
        radius = self.local_radius

        for _ in range(self.local_max_iter):
            if radius < 1e-6:
                break

            directions = np.random.randn(self.dim)
            directions = directions / (np.linalg.norm(directions) + 1e-10)
            candidates = np.clip(center + radius * directions, self.lower, self.upper)
            candidates = np.vstack([candidates, np.clip(center - radius * directions, self.lower, self.upper)])
            candidates = np.vstack([candidates, center])

            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = np.argmin(fit_candidates)
            best_local_fitness = float(np.asarray(fit_candidates[best_local_idx]).item())

            if best_local_fitness < current_fitness:
                center = candidates[best_local_idx].copy()
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
        return float(np.mean(distances).item())

    def _check_diversity_low(self, population):
        diversity = self._compute_diversity_batch(population)
        return (diversity < self.diversity_threshold or 
                self.stagnation_count > self.stagnation_limit)

    def _reinitialize_population(self, population, fitness):
        if self.best_solution is not None:
            best_idx = np.argmin(fitness)
            population[0] = self.best_solution.copy()
            fitness[0] = self.best_fitness

        n_random = max(self.min_pop_size, self.NP // 2)
        new_part = self._initialize_population_lhs()[:n_random]
        population[1:1+n_random] = new_part
        fitness[1:1+n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5
        self.operator_gen_counter = 0
