import numpy as np


class CovarianceRingMemeticOptimizer:
    """
    Population-based optimizer combining:
    - Ring topology for neighborhood mutation (local information flow)
    - Multiple competing mutation strategies with adaptation
    - Simplified CMA-ES covariance adaptation
    - Step-size adaptation (CMA-ES style)
    - Diversity-triggered restart mechanism
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(8 * dim, 300)  # Population size
        self.bounds = np.array([-100.0, 100.0])
        self._init_parameters()
        self._init_history()

    def _init_parameters(self):
        """Initialize algorithm-specific hyperparameters."""
        # Step size (similar to CMA-ES sigma)
        self.sigma = 0.5 * (self.bounds[1] - self.bounds[0]) / np.sqrt(self.dim)
        # Covariance matrix (diagonal approximation for efficiency)
        self.cov_diag = np.ones(self.dim)
        # Learning rates for covariance adaptation
        self.c_c = 2.0 / self.dim  # Covariance adaptation rate
        self.c_s = 0.3  # Step-size adaptation rate
        # Number of mutation strategies
        self.n_strategies = 4
        self.strategy_weights = np.ones(self.n_strategies) / self.n_strategies
        self.strategy_success = np.zeros(self.n_strategies)
        # Ring neighborhood: each individual has 2 neighbors
        self.neighbor_indices = self._build_ring_topology()
        # Diversity threshold for restart
        self.min_diversity = 1e-6
        # Stagnation counter
        self.stagnation_count = 0
        self.max_stagnation = 50

    def _build_ring_topology(self):
        """Build ring neighborhood indices for each population member."""
        n = self.np
        neighbors = np.zeros((n, 2), dtype=int)
        for i in range(n):
            neighbors[i, 0] = (i - 1) % n
            neighbors[i, 1] = (i + 1) % n
        return neighbors

    def _init_history(self):
        """Initialize history tracking for adaptation."""
        self.fitness_history = []
        self.strategy_uses = np.zeros(self.n_strategies)

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize population
        population = self._initialize_population()
        fitness = self._evaluate_batch(func, population)
        best_idx = np.argmin(fitness)
        f_opt, x_opt = fitness[best_idx], population[best_idx].copy()

        # Main optimization loop
        while not stopping_condition():
            # Generate trial population using multiple strategies
            trials, trial_strategies = self._generate_trials_batch(population, fitness)
            trial_fitness = self._evaluate_batch(func, trials)

            if len(trial_fitness) < len(trials):
                # Budget exhausted
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                trial_strategies = trial_strategies[:valid_len]

            if len(trial_fitness) == 0:
                break

            if stopping_condition():
                break

            # Selection: keep better of trial vs population
            population, fitness, trials, trial_fitness, trial_strategies = \
                self._select_survivors_batch(population, fitness, trials, trial_fitness, trial_strategies)

            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < f_opt:
                f_opt, x_opt = fitness[best_idx], population[best_idx].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Adapt step size based on success rate
            self._adapt_step_size(fitness, trial_fitness)

            # Adapt covariance matrix
            self._adapt_covariance_variant_08(population, x_opt)

            # Adapt mutation strategies
            self._adapt_mutation_strategies(trial_strategies, fitness, trial_fitness)

            # Check for restart conditions
            if self._check_restart_conditions(fitness):
                population = self._restart_population(population, fitness, x_opt)
                fitness = self._evaluate_batch(func, population)

            # Record history
            self.fitness_history.append(f_opt)

        return f_opt, x_opt

    def _initialize_population(self):
        """Initialize population using latin hypercube sampling."""
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.np, self.dim))
        # Apply LHS for better spread
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.bounds[0] + (perm + np.random.random(self.np)) / self.np * (self.bounds[1] - self.bounds[0])
        return np.clip(pop, self.bounds[0], self.bounds[1])

    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population at once."""
        clipped = np.clip(population, self.bounds[0], self.bounds[1])
        return func(clipped)

    def _generate_trials_batch(self, population, fitness):
        """Generate trial solutions using multiple adaptive strategies."""
        trials = np.empty((self.np, self.dim))
        trial_strategies = np.empty(self.np, dtype=int)

        # Sample strategies based on weights
        strategy_probs = self.strategy_weights / self.strategy_weights.sum()
        for i in range(self.np):
            trial_strategies[i] = np.random.choice(self.n_strategies, p=strategy_probs)
            self.strategy_uses[trial_strategies[i]] += 1

        # Apply each strategy to assigned individuals
        for s in range(self.n_strategies):
            mask = trial_strategies == s
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            pop_subset = population[indices]

            if s == 0:
                trials[mask] = self._strategy_ring_mutation(pop_subset, population)
            elif s == 1:
                trials[mask] = self._strategy_covariance_mutation(pop_subset, population, fitness)
            elif s == 2:
                trials[mask] = self._strategy_current_to_best(pop_subset, population, fitness)
            else:
                trials[mask] = self._strategy_dithering_mutation(pop_subset, population)

        # Clip to bounds
        trials = np.clip(trials, self.bounds[0], self.bounds[1])
        return trials, trial_strategies

    def _strategy_ring_mutation(self, pop_subset, population):
        """Ring topology mutation: use neighbors in ring."""
        n = len(pop_subset)
        mutated = np.empty((n, self.dim))
        for i in range(n):
            idx = np.random.randint(self.np)
            r1, r2 = self.neighbor_indices[idx]
            # Randomly pick one neighbor
            donor = population[r1] if np.random.random() < 0.5 else population[r2]
            mutated[i] = pop_subset[i] + self.sigma * (donor - pop_subset[i])
        return mutated

    def _strategy_covariance_mutation(self, pop_subset, population, fitness):
        """Mutation guided by covariance matrix."""
        n = len(pop_subset)
        mutated = np.empty((n, self.dim))
        # Use top performers to update covariance direction
        top_k = max(3, self.np // 4)
        top_indices = np.argsort(fitness)[:top_k]
        top_pop = population[top_indices]
        mean_top = top_pop.mean(axis=0)
        # Weighted combination of ring and covariance-guided mutation
        for i in range(n):
            cov_dir = mean_top - pop_subset[i]
            ring_idx = np.random.randint(self.np)
            donor = population[ring_idx]
            ring_dir = donor - pop_subset[i]
            alpha = np.random.random()
            mutated[i] = pop_subset[i] + self.sigma * (alpha * cov_dir + (1 - alpha) * ring_dir)
        return mutated

    def _strategy_current_to_best(self, pop_subset, population, fitness):
        """Current-to-best mutation (DE-style)."""
        n = len(pop_subset)
        mutated = np.empty((n, self.dim))
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        for i in range(n):
            idx = np.random.randint(self.np)
            r1, r2 = self.neighbor_indices[idx]
            donor = population[r1] + population[r2] - 2 * pop_subset[i]
            mutated[i] = pop_subset[i] + self.sigma * (best - pop_subset[i] + donor)
        return mutated

    def _strategy_dithering_mutation(self, pop_subset, population):
        """Dithering mutation with random scaling."""
        n = len(pop_subset)
        mutated = np.empty((n, self.dim))
        for i in range(n):
            idx = np.random.randint(self.np)
            donor = population[idx]
            scale = np.random.uniform(0.3, 1.2)
            mutated[i] = pop_subset[i] + self.sigma * scale * (donor - pop_subset[i])
        return mutated

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, trial_strategies):
        """Selection: greedy (DE-style) with batched operations."""
        # Compare population vs trials
        better_mask = trial_fitness < fitness
        # Update population with improvements
        for i in range(len(population)):
            if better_mask[i]:
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
            else:
                trials[i] = population[i]
                trial_fitness[i] = fitness[i]
                trial_strategies[i] = -1  # Mark as not a new trial
        return population, fitness, trials, trial_fitness, trial_strategies

    def _adapt_step_size(self, fitness, trial_fitness):
        """Adapt step size based on improvement rate."""
        improvements = trial_fitness < fitness
        success_rate = np.mean(improvements)
        # Increase step size if success rate is good, decrease otherwise
        if success_rate > 0.2:
            self.sigma *= np.exp(0.2 * (success_rate - 0.2))
        else:
            self.sigma *= np.exp(-0.5)
        # Bound step size
        max_sigma = (self.bounds[1] - self.bounds[0]) * 0.5
        min_sigma = 1e-6
        self.sigma = np.clip(self.sigma, min_sigma, max_sigma)

    def _adapt_covariance_variant_08(self, population, x_opt):
        """Simplified covariance adaptation variant 08."""
        # Update diagonal covariance based on deviation from best
        deviation = population - x_opt
        var_est = np.var(population, axis=0) + 1e-8
        # Smooth update
        self.cov_diag = (1 - self.c_c) * self.cov_diag + self.c_c * var_est
        # Ensure positive
        self.cov_diag = np.maximum(self.cov_diag, 1e-8)

    def _adapt_covariance_variant_01(self, population, x_opt):
        """Alternative covariance adaptation."""
        # Use rank-1 update approximation
        diff = population.mean(axis=0) - x_opt
        rank1 = diff[:, np.newaxis] * diff[np.newaxis, :] / (np.dot(diff, diff) + 1e-8)
        # Simplified diagonal update
        self.cov_diag = (1 - 0.5 * self.c_c) * self.cov_diag + 0.5 * self.c_c * np.diag(rank1 + 1)

    def _adapt_covariance(self, population, best):
        """General covariance adaptation."""
        centered = population - best
        self.cov_diag = (1 - self.c_c) * self.cov_diag + self.c_c * (centered ** 2).mean(axis=0)

    def _adapt_covariance_variant_09(self, population, x_opt):
        """Covariance adaptation variant 09 with momentum."""
        # Compute weighted mean deviation
        deviation = np.mean(population - x_opt, axis=0)
        self.cov_diag = 0.9 * self.cov_diag + 0.1 * (deviation ** 2 + 1)

    def _adapt_mutation_strategies(self, trial_strategies, fitness, trial_fitness):
        """Adapt mutation strategy weights based on success."""
        for s in range(self.n_strategies):
            mask = trial_strategies == s
            if np.sum(mask) > 0:
                trial_fitness_s = trial_fitness[mask]
                fitness_s = fitness[mask]
                # Success if trial is better
                success_s = np.mean(trial_fitness_s < fitness_s)
                self.strategy_success[s] = 0.9 * self.strategy_success[s] + 0.1 * success_s

        # Update weights based on success
        if np.sum(self.strategy_success) > 0:
            self.strategy_weights = 0.9 * self.strategy_weights + 0.1 * self.strategy_success
            self.strategy_weights = np.maximum(self.strategy_weights, 0.01)
            self.strategy_weights /= self.strategy_weights.sum()

    def _compute_diversity(self, population):
        """Compute population diversity (average pairwise distance)."""
        mean_pop = population.mean(axis=0)
        distances = np.sqrt(np.sum((population - mean_pop) ** 2, axis=1))
        return np.mean(distances)

    def _check_restart_conditions(self, fitness):
        """Check if restart is needed."""
        # Check diversity
        if len(self.fitness_history) > 10:
            recent_diversity = self._compute_diversity(population if 'population' in dir() else np.zeros((self.np, self.dim)))
            if recent_diversity < self.min_diversity:
                return True
        # Check stagnation
        if self.stagnation_count > self.max_stagnation:
            return True
        return False

    def _restart_population(self, population, fitness, x_opt):
        """Restart population maintaining some good solutions."""
        n_keep = self.np // 4
        best_indices = np.argsort(fitness)[:n_keep]
        new_pop = np.empty((self.np, self.dim))
        new_pop[:n_keep] = population[best_indices]
        # Generate new random solutions for the rest
        for i in range(n_keep, self.np):
            # Bias towards promising regions
            if np.random.random() < 0.5:
                new_pop[i] = x_opt + np.random.normal(0, self.sigma, self.dim)
            else:
                new_pop[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        new_pop = np.clip(new_pop, self.bounds[0], self.bounds[1])
        # Reset adaptation parameters
        self.sigma = 0.5 * (self.bounds[1] - self.bounds[0]) / np.sqrt(self.dim)
        self.cov_diag = np.ones(self.dim)
        self.stagnation_count = 0
        return new_pop
