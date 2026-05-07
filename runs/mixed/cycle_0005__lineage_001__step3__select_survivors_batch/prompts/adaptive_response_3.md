```python
import numpy as np


class CulturalMemeticDE:
    """
    Adaptive DE with Thompson Sampling for survivor selection strategy.
    Automatically learns which _select_survivors_batch strategy works best during optimization.
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

        # === Adaptive Operator Selection (Thompson Sampling for Survivor Selection) ===
        self.num_survivor_operators = 6
        self.survivor_operator_names = [
            'original_greedy',
            'mu_plus_lambda',
            'diversity_weighted',
            'boltzmann_elite',
            'age_tracking_injection',
            'boltzmann_adaptive'
        ]
        # Beta distribution parameters for Thompson Sampling
        self.operator_alpha = np.ones(self.num_survivor_operators)
        self.operator_beta = np.ones(self.num_survivor_operators)
        # Sliding window reward tracking
        self.reward_window_size = 30
        self.operator_rewards = [[] for _ in range(self.num_survivor_operators)]
        self.current_survivor_op_idx = 0
        self.last_operator_switch_gen = 0
        self.operator_min_switch_gens = 5
        self.operator_switches = []

        # Archive for mu_plus_lambda (variant_01 style)
        self.archive = None
        self._pending_archive_additions = None

        # Age tracking for age_tracking_injection strategy
        self.individual_ages = None

    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions"""
        samples = np.random.beta(self.operator_alpha, self.operator_beta)
        selected = int(np.argmax(samples))
        return selected

    def _compute_reward(self, fitness, trial_fitness):
        """Compute reward based on improvement quality"""
        # Guard against invalid values
        fitness = np.clip(fitness, -1e50, 1e50)
        trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

        improvements = fitness - trial_fitness
        total_improvement = float(np.sum(improvements[improvements > 0]))
        n_improved = int(np.sum(improvements > 0))

        if n_improved == 0:
            reward = -0.01
        else:
            # Normalize by population size and number of improvements
            reward = total_improvement / max(float(n_improved), 1.0)

        return float(np.clip(reward, -1.0, 1.0))

    def _update_operator_rewards(self, operator_idx, reward):
        """Update rewards for operator with sliding window"""
        rewards = self.operator_rewards[operator_idx]
        rewards.append(reward)

        # Maintain sliding window
        if len(rewards) > self.reward_window_size:
            rewards.pop(0)

        # Update Beta distribution parameters based on recent performance
        recent = rewards[-min(10, len(rewards)):]
        if len(recent) >= 3:
            successes = sum(1 for r in recent if r > 0)
            success_rate = successes / len(recent)
            avg_reward = np.mean(recent)

            # Update alpha (successes + 1) and beta (failures + 1)
            self.operator_alpha[operator_idx] = 1.0 + success_rate * 5.0 + avg_reward * 2.0
            self.operator_beta[operator_idx] = 1.0 + (1.0 - success_rate) * 5.0 - avg_reward * 2.0

            # Ensure valid parameters
            self.operator_alpha[operator_idx] = float(np.clip(self.operator_alpha[operator_idx], 0.1, 100.0))
            self.operator_beta[operator_idx] = float(np.clip(self.operator_beta[operator_idx], 0.1, 100.0))

    def __call__(self, func, stopping_condition):
        population = self._initialize_population_lhs()
        fitness = self._eval_wrapper_batch(population, func)

        # Guard against invalid initial fitness
        fitness = np.clip(fitness, -1e50, 1e50)

        best_idx = np.argmin(fitness)
        self.best_fitness = float(fitness[best_idx])
        self.best_solution = population[best_idx].copy()

        # Initialize operator selection
        self.current_survivor_op_idx = self._select_operator_thompson()

        while not stopping_condition():
            mutants = self._mutate_current_to_pbest_with_culture(population, fitness)
            mutants = np.clip(mutants, self.lower, self.upper)

            trials = self._crossover_binomial_batch(population, mutants)
            trials = self._clip_to_bounds_batch(trials)

            trial_fitness = self._eval_wrapper_batch(trials, func)

            # Guard against invalid trial fitness
            trial_fitness = np.clip(trial_fitness, -1e50, 1e50)

            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break

            if stopping_condition():
                break

            # Compute reward before selection
            reward = self._compute_reward(fitness, trial_fitness)

            # Dispatch to selected survivor selection strategy
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            fitness = np.clip(fitness, -1e50, 1e50)

            best_idx = np.argmin(fitness)
            current_best = float(fitness[best_idx])
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Update operator performance
            self._update_operator_rewards(self.current_survivor_op_idx, reward)

            self._update_cultural_memory_on_improvement(population[best_idx], mutants[best_idx])
            self._adapt_parameters_batch(population, fitness, mutants, trials)

            if self.generation % self.local_search_interval == 0:
                self._apply_adaptive_local_search(func)

            self.generation += 1

            # Operator switching with minimum generations between switches
            if self.generation - self.last_operator_switch_gen >= self.operator_min_switch_gens:
                new_op = self._select_operator_thompson()
                if new_op != self.current_survivor_op_idx:
                    self.current_survivor_op_idx = new_op
                    self.last_operator_switch_gen = self.generation
                    self.operator_switches.append((self.generation, new_op))

            if self._check_diversity_low(population):
                self._reinitialize_population(population, fitness)

        return self.best_fitness, self.best_solution

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

    def _mutate_current_to_pbest_with_culture(self, population, fitness):
        """Original mutation strategy: current-to-pbest/1 with cultural memory"""
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

        return np.clip(donors, self.lower, self.upper)

    def _crossover_binomial_batch(self, population, donors):
        j_rand = np.random.randint(0, self.dim, size=self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        trials = np.where(mask, donors, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Dispatch to selected survivor selection strategy"""
        if self.current_survivor_op_idx == 0:
            return self._select_survivors_original(population, fitness, trials, trial_fitness)
        elif self.current_survivor_op_idx == 1:
            return self._select_survivors_mu_plus_lambda(population, fitness, trials, trial_fitness)
        elif self.current_survivor_op_idx == 2:
            return self._select_survivors_diversity_weighted(population, fitness, trials, trial_fitness)
        elif self.current_survivor_op_idx == 3:
            return self._select_survivors_boltzmann_elite(population, fitness, trials, trial_fitness)
        elif self.current_survivor_op_idx == 4:
            return self._select_survivors_age_tracking(population, fitness, trials, trial_fitness)
        else:
            return self._select_survivors_boltzmann_adaptive(population, fitness, trials, trial_fitness)

    def _select_survivors_original(self, population, fitness, trials, trial_fitness):
        """Original greedy selection"""
        better = trial_fitness < fitness
        new_population = np.copy(population)
        new_population[better] = trials[better]
        new_fitness = np.copy(fitness)
        new_fitness[better] = trial_fitness[better]
        return new_population, new_fitness

    def _select_survivors_mu_plus_lambda(self, population, fitness, trials, trial_fitness):
        """Variant 01: μ+λ style selection with argpartition"""
        # Guard against empty or malformed inputs
        if population.shape[0] == 0 or trials.shape[0] == 0:
            return np.copy(population), np.copy(fitness)

        # Ensure matching dimensions
        np_pop = min(population.shape[0], trials.shape[0])
        pop_subset = population[:np_pop]
        fit_subset = fitness[:np_pop]
        trial_subset = trials[:np_pop]
        trial_fit_subset = trial_fitness[:np_pop]

        # Combine parent and trial pools
        combined_pop = np.vstack([pop_subset, trial_subset])
        combined_fit = np.concatenate([fit_subset, trial_fit_subset])

        # Guard against invalid fitness values
        combined_fit = np.clip(combined_fit, -1e50, 1e50)

        # Select best NP individuals (μ+λ style)
        if len(combined_fit) > np_pop:
            # Use argpartition for O(n) partial sort, then sort the selected k
            selected_indices = np.argpartition(combined_fit, np_pop)[:np_pop]
            selected_indices = selected_indices[np.argsort(combined_fit[selected_indices])]
        else:
            selected_indices = np.argsort(combined_fit)[:np_pop]

        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        new_population[:np_pop] = combined_pop[selected_indices]
        new_fitness[:np_pop] = combined_fit[selected_indices]

        return new_population, new_fitness

    def _select_survivors_diversity_weighted(self, population, fitness, trials, trial_fitness):
        """Variant 05: Diversity-weighted selection"""
        # Compute diversity contribution: average distance to other population members
        def diversity_contribution(solution, pop):
            if len(pop) < 2:
                return 0.0
            dists = np.linalg.norm(pop - solution, axis=1)
            return float(np.mean(dists))

        candidates_pop = np.vstack([population, trials])
        candidates_fit = np.concatenate([fitness, trial_fitness])

        # Compute diversity contributions for all candidates
        div_contribs = np.array([diversity_contribution(candidates_pop[i], candidates_pop)
                                 for i in range(len(candidates_pop))])

        # Normalize diversity contributions
        div_min, div_max = np.min(div_contribs), np.max(div_contribs)
        if div_max > div_min + 1e-10:
            div_normalized = (div_contribs - div_min) / (div_max - div_min)
        else:
            div_normalized = np.ones_like(div_contribs) * 0.5

        # Compute fitness scores (lower fitness = better)
        fit_min, fit_max = np.min(candidates_fit), np.max(candidates_fit)
        if fit_max > fit_min + 1e-10:
            fitness_score = 1.0 - (candidates_fit - fit_min) / (fit_max - fit_min)
        else:
            fitness_score = np.ones_like(candidates_fit) * 0.5

        # Adaptive diversity weight: increases when stagnant
        diversity_weight = min(0.4, 0.1 + 0.02 * self.stagnation_count)

        # Combined score: balance fitness vs diversity
        combined_score = (1.0 - diversity_weight) * fitness_score + diversity_weight * div_normalized

        # Select: for each pair (original, trial), pick the one with higher combined score
        n = len(population)
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)

        for i in range(n):
            orig_idx = i
            trial_idx = n + i
            if combined_score[trial_idx] > combined_score[orig_idx]:
                new_population[i] = candidates_pop[trial_idx]
                new_fitness[i] = candidates_fit[trial_idx]
            else:
                new_population[i] = candidates_pop[orig_idx]
                new_fitness[i] = candidates_fit[orig_idx]

        return new_population, new_fitness

    def _select_survivors_boltzmann_elite(self, population, fitness, trials, trial_fitness):
        """Variant 06: Boltzmann selection with deterministic elitism"""
        # Track global best separately for deterministic elitism
        best_idx = np.argmin(fitness)
        global_best = population[best_idx].copy()
        global_best_fit = float(fitness[best_idx])

        # Adaptive temperature based on stagnation
        if self.stagnation_count > 20:
            temperature = min(0.5, 0.01 * (self.stagnation_count - 20))
        elif self.stagnation_count > 10:
            temperature = 0.01
        else:
            temperature = 0.001

        new_population = np.copy(population)
        new_fitness = np.copy(fitness)

        for i in range(len(population)):
            delta = float(fitness[i]) - float(trial_fitness[i])
            if delta > 0:
                # Trial is better: always accept
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
            elif self.stagnation_count > 15 and temperature > 1e-6:
                # Trial is worse but we're stagnant: Boltzmann acceptance
                accept_prob = np.exp(delta / temperature)
                if np.random.random() < accept_prob:
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_fitness[i]

        # Deterministic elitism: always preserve global best
        new_best_idx = np.argmin(new_fitness)
        if float(new_fitness[new_best_idx]) > global_best_fit:
            new_population[new_best_idx] = global_best
            new_fitness[new_best_idx] = global_best_fit

        return new_population, new_fitness

    def _select_survivors_age_tracking(self, population, fitness, trials, trial_fitness):
        """Variant 08: Age tracking with diversity injection"""
        NP = len(population)

        # Initialize age tracking if needed
        if not hasattr(self, 'individual_ages') or self.individual_ages is None:
            self.individual_ages = np.zeros(NP)

        # Ensure age array matches population size
        if len(self.individual_ages) != NP:
            self.individual_ages = np.zeros(NP)

        # Increment ages of all existing individuals
        self.individual_ages += 1

        # Elitism: always keep the single best individual unchanged
        best_idx = int(np.argmin(fitness))

        # Compute improvement scores
        improvements = fitness - trial_fitness
        is_better = improvements > 0

        # Create replacement mask and tracking arrays
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        new_ages = np.copy(self.individual_ages)

        # For individuals that improve: compute replacement score
        # Score = improvement + age_bonus (favor replacing older individuals)
        age_bonus = 0.01 * self.individual_ages
        replacement_scores = np.where(is_better, improvements + age_bonus, -np.inf)

        # Find best replacement for each trial
        for i in range(NP):
            if i == best_idx:
                continue  # Skip elitist
            if replacement_scores[i] > 0:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
                new_ages[i] = 0  # Reset age on successful replacement

        # Periodically inject diversity by replacing worst 20% with random
        # but only if stagnation is high (adaptive diversity injection)
        if self.stagnation_count > 15:
            n_replace = max(1, NP // 5)
            worst_indices = np.argsort(new_fitness)[-n_replace:]
            for idx in worst_indices:
                if idx != best_idx:
                    # Generate random individual near best but with perturbation
                    random_ind = self.best_solution + np.random.randn(self.dim) * (10.0 + self.stagnation_count * 0.1)
                    random_ind = np.clip(random_ind, self.lower, self.upper)
                    new_population[idx] = random_ind
                    new_fitness[idx] = np.inf
                    new_ages[idx] = 0

        self.individual_ages = new_ages
        return new_population, new_fitness

    def _select_survivors_boltzmann_adaptive(self, population, fitness, trials, trial_fitness):
        """Variant 10: Boltzmann selection with adaptive temperature"""
        # Adaptive temperature based on stagnation state
        if self.stagnation_count > 30:
            temperature = 2.0 + 0.5 * min(self.stagnation_count - 30, 150)
        elif self.stagnation_count > 15:
            temperature = 0.5 + 0.2 * (self.stagnation_count - 15)
        elif self.stagnation_count > 5:
            temperature = 0.2 + 0.15 * (self.stagnation_count - 5)
        else:
            temperature = 0.1

        temperature = max(float(temperature), 0.01)

        # Compute fitness delta (positive = trial is worse)
        fitness_delta = trial_fitness - fitness

        # Boltzmann acceptance probability
        # Better or equal trials (delta <= 0): accept with P = 1.0
        # Worse trials (delta > 0): P = exp(-delta / T)
        # Clamp exponent for numerical stability
        exponent = np.clip(-fitness_delta / temperature, -700, 700)
        accept_prob = np.where(fitness_delta <= 0, 1.0, np.exp(exponent))

        # Probabilistic selection
        roll = np.random.rand(len(fitness))
        accept = roll < accept_prob

        # Build new population/fitness
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        new_population[accept] = trials[accept]
        new_fitness[accept] = trial_fitness[accept]

        return new_population, new_fitness

    def _update_cultural_memory_on_improvement(self, best_individual, best_mutant):
        if self.stagnation_count == 0:
            pattern = best_mutant - best_individual
            self.cultural_memory[self.culture_idx] = pattern
            self.cultural_weights[self.culture_idx] = 1.0 / (1.0 + self.stagnation_count)
            self.culture_idx = (self.culture_idx + 1) % self.cultural_memory_size
            decay = 0.95
            self.cultural_weights *= decay

    def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        sorted_idx = np.argsort(fitness)
        top_half = sorted_idx[:self.NP // 2]

        F_candidates = np.random.uniform(0.4, 1.0, size=5)
        self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))

        CR_candidates = np.random.uniform(0.3, 0.9, size=5)
        self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))

        if np.random.rand() < 0.2:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.3))

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
            best_local_fitness = float(fit_candidates[best_local_idx])

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
        return float(np.mean(distances))

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
        population[1:1 + n_random] = new_part
        fitness[1:1 + n_random] = np.inf

        self.stagnation_count = 0
        self.local_radius = max(self.local_radius * 0.5, 0.1)
        self.cultural_weights *= 0.5
```