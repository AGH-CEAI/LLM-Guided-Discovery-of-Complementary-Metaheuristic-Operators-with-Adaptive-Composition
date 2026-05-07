import numpy as np


class AdaptiveRankWeightedDE:
    """
    Differential Evolution with rank-weighted mutation, adaptive CR/F per individual,
    opposition-based learning restarts, and archive-assisted mutation.
    
    Key novelties:
    - Mutation uses rank-weighted selection of base vectors (better individuals more likely chosen)
    - Each individual maintains its own CR and F, adapted via Cauchy/normal distributions
    - Successful parameters are stored and used to guide future parameter generation
    - An external archive of recently replaced solutions is used in "current-to-pbest/archive" mutation
    - Population restarts with opposition-based learning when diversity collapses
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.NP = max(20, 6 * dim)
        self.NP = min(self.NP, 300)
        self.p_best_rate = 0.15
        self.archive_max_size = self.NP
        self.memory_size = 10
        self.mu_cr = np.full(self.memory_size, 0.5)
        self.mu_f = np.full(self.memory_size, 0.5)
        self.memory_index = 0
        self.min_pop_size = max(10, dim)

    def _initialize_population(self):
        pop = np.random.uniform(self.lb, self.ub, size=(self.NP, self.dim))
        return pop

    def _initialize_archive(self):
        return np.empty((0, self.dim))

    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.lb, self.ub)

    def _generate_cr_f_batch(self, size):
        """Generate CR and F for each individual using memory-guided distributions."""
        ri = np.random.randint(0, self.memory_size, size=size)
        cr_values = np.random.normal(self.mu_cr[ri], 0.1, size=size)
        cr_values = np.clip(cr_values, 0.0, 1.0)
        f_values = self._cauchy_sample(self.mu_f[ri], 0.1, size=size)
        f_values = np.clip(f_values, 0.01, 1.5)
        return cr_values, f_values

    def _cauchy_sample(self, loc, scale, size):
        """Sample from Cauchy distribution, re-sampling negatives."""
        samples = loc + scale * np.tan(np.pi * (np.random.random(size=size) - 0.5))
        negative_mask = samples <= 0
        max_retries = 10
        for _ in range(max_retries):
            if not np.any(negative_mask):
                break
            n_resample = np.sum(negative_mask)
            loc_sub = loc[negative_mask] if isinstance(loc, np.ndarray) else loc
            samples[negative_mask] = loc_sub + scale * np.tan(
                np.pi * (np.random.random(size=n_resample) - 0.5)
            )
            negative_mask = samples <= 0
        samples[samples <= 0] = 0.01
        return samples

    def _compute_ranks_and_weights(self, fitness):
        """Compute rank-based weights: best individual gets highest weight."""
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(fitness))
        # Weight: exponential decay based on rank (rank 0 = best)
        weights = np.exp(-0.05 * ranks.astype(float))
        weights /= weights.sum()
        return ranks, weights

    def _select_pbest_indices(self, fitness, size):
        """Select random indices from the top-p fraction of the population."""
        p = max(2, int(self.p_best_rate * len(fitness)))
        top_indices = np.argsort(fitness)[:p]
        chosen = top_indices[np.random.randint(0, p, size=size)]
        return chosen

    def _select_weighted_base_vectors(self, weights, size, exclude_indices=None):
        """Select base vectors using rank-based weights, avoiding self-selection."""
        NP = len(weights)
        bases = np.empty(size, dtype=int)
        for i in range(size):
            w = weights.copy()
            if exclude_indices is not None:
                w[exclude_indices[i]] = 0.0
            w_sum = w.sum()
            if w_sum <= 0:
                w = np.ones(NP) / NP
            else:
                w /= w_sum
            bases[i] = np.random.choice(NP, p=w)
        return bases

    def _mutate_batch(self, population, fitness, archive, cr_values, f_values):
        """
        Current-to-pbest/archive mutation with rank-weighted base selection.
        v_i = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2_or_archive)
        where r1 is selected with rank-based weights.
        """
        NP, dim = population.shape
        indices = np.arange(NP)

        # Select pbest
        pbest_idx = self._select_pbest_indices(fitness, NP)

        # Rank-weighted selection for r1
        _, weights = self._compute_ranks_and_weights(fitness)
        r1 = self._select_weighted_base_vectors(weights, NP, exclude_indices=indices)

        # r2 from population + archive combined
        combined = np.vstack([population, archive]) if len(archive) > 0 else population.copy()
        n_combined = len(combined)
        r2 = np.random.randint(0, n_combined, size=NP)
        # Ensure r2 != i and r2 != r1 (when r2 < NP)
        for attempt in range(5):
            conflicts = (r2 == indices) | (r2 == r1)
            if not np.any(conflicts):
                break
            r2[conflicts] = np.random.randint(0, n_combined, size=np.sum(conflicts))

        F = f_values[:, np.newaxis]
        mutants = (
            population
            + F * (population[pbest_idx] - population)
            + F * (population[r1] - combined[r2])
        )
        return mutants

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to the whole population at once."""
        NP, dim = population.shape
        rand_matrix = np.random.random((NP, dim))
        j_rand = np.random.randint(0, dim, size=NP)

        cross_mask = rand_matrix < cr_values[:, np.newaxis]
        # Ensure at least one dimension is from mutant
        cross_mask[np.arange(NP), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection: keep the better of parent and trial."""
        valid = np.isfinite(trial_fitness)
        improved = valid & (trial_fitness <= fitness)

        new_pop = population.copy()
        new_fit = fitness.copy()
        replaced = np.zeros(len(fitness), dtype=bool)

        new_pop[improved] = trials[improved]
        new_fit[improved] = trial_fitness[improved]
        replaced[improved] = True

        return new_pop, new_fit, replaced

    def _update_archive(self, archive, replaced_individuals):
        """Add replaced individuals to archive, trimming if oversized."""
        if len(replaced_individuals) == 0:
            return archive
        archive = np.vstack([archive, replaced_individuals]) if len(archive) > 0 else replaced_individuals.copy()
        if len(archive) > self.archive_max_size:
            keep = np.random.choice(len(archive), self.archive_max_size, replace=False)
            archive = archive[keep]
        return archive

    def _update_parameter_memory(self, cr_values, f_values, fitness_improvements, replaced):
        """Update memory with weighted Lehmer mean of successful CR/F."""
        if not np.any(replaced):
            return
        s_cr = cr_values[replaced]
        s_f = f_values[replaced]
        deltas = fitness_improvements[replaced]
        deltas = np.maximum(deltas, 1e-30)
        weights = deltas / deltas.sum()

        # Weighted Lehmer mean for F
        mean_f = np.sum(weights * s_f ** 2) / max(np.sum(weights * s_f), 1e-30)
        # Weighted arithmetic mean for CR
        mean_cr = np.sum(weights * s_cr)

        self.mu_f[self.memory_index] = mean_f
        self.mu_cr[self.memory_index] = mean_cr
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance to centroid."""
        centroid = np.mean(population, axis=0)
        dists = np.sqrt(np.sum((population - centroid) ** 2, axis=1))
        return np.mean(dists)

    def _opposition_based_restart(self, population, fitness, best_x, best_f):
        """
        When diversity collapses, generate opposition-based population
        centered around the current best, keeping the best solution.
        """
        NP, dim = population.shape
        # Opposition: x_opp = lb + ub - x
        opp_pop = self.lb + self.ub - population
        # Add noise to avoid exact mirroring
        noise = np.random.normal(0, 5.0, size=(NP, dim))
        opp_pop = opp_pop + noise
        opp_pop = self._clip_to_bounds(opp_pop)
        # Keep best individual
        opp_pop[0] = best_x.copy()
        return opp_pop

    def _local_search_best(self, best_x, best_f, func, stopping_condition):
        """
        Small Gaussian perturbation around the best solution.
        Evaluate a small batch and see if any improve.
        """
        if stopping_condition():
            return best_x, best_f
        n_local = min(self.dim, 20)
        sigma = max(0.1, 0.01 * (self.ub - self.lb))
        perturbations = best_x + np.random.normal(0, sigma, size=(n_local, self.dim))
        perturbations = self._clip_to_bounds(perturbations)
        local_fit = func(perturbations)
        valid = np.isfinite(local_fit)
        if np.any(valid):
            best_local_idx = np.argmin(np.where(valid, local_fit, np.inf))
            if local_fit[best_local_idx] < best_f:
                return perturbations[best_local_idx].copy(), local_fit[best_local_idx]
        return best_x, best_f

    def _should_restart(self, diversity, generation, last_improvement_gen):
        """Decide if we should trigger an opposition-based restart."""
        stagnation = generation - last_improvement_gen
        diversity_threshold = 1e-6 * (self.ub - self.lb) * np.sqrt(self.dim)
        stagnation_limit = max(50, 10 * self.dim)
        return (diversity < diversity_threshold) or (stagnation > stagnation_limit)

    def _track_best(self, population, fitness, best_x, best_f):
        """Update global best from current population."""
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return best_x, best_f
        min_idx = np.argmin(np.where(valid, fitness, np.inf))
        if fitness[min_idx] < best_f:
            return population[min_idx].copy(), fitness[min_idx]
        return best_x, best_f

    def __call__(self, func, stopping_condition):
        # Initialize
        population = self._initialize_population()
        population = self._clip_to_bounds(population)
        archive = self._initialize_archive()

        # Reset memory
        self.mu_cr = np.full(self.memory_size, 0.5)
        self.mu_f = np.full(self.memory_size, 0.5)
        self.memory_index = 0

        fitness = func(population)
        if stopping_condition():
            valid = np.isfinite(fitness)
            if np.any(valid):
                idx = np.argmin(np.where(valid, fitness, np.inf))
                return fitness[idx], population[idx].copy()
            return np.inf, population[0].copy()

        best_x, best_f = self._track_best(
            population, fitness,
            population[0].copy(), fitness[0] if np.isfinite(fitness[0]) else np.inf
        )

        generation = 0
        last_improvement_gen = 0

        while not stopping_condition():
            generation += 1
            NP = len(population)

            # Generate adaptive parameters
            cr_values, f_values = self._generate_cr_f_batch(NP)

            # Mutation
            mutants = self._mutate_batch(population, fitness, archive, cr_values, f_values)
            mutants = self._clip_to_bounds(mutants)

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values)
            trials = self._clip_to_bounds(trials)

            # Evaluate trials
            trial_fitness = func(trials)
            if stopping_condition():
                # Still try to extract any improvements
                valid_len = min(len(trial_fitness), NP)
                if valid_len > 0:
                    tf_sub = trial_fitness[:valid_len]
                    valid = np.isfinite(tf_sub)
                    if np.any(valid):
                        idx = np.argmin(np.where(valid, tf_sub, np.inf))
                        if tf_sub[idx] < best_f:
                            best_f = tf_sub[idx]
                            best_x = trials[idx].copy()
                break

            # Handle truncated returns
            if len(trial_fitness) < NP:
                trial_fitness = np.concatenate([
                    trial_fitness,
                    np.full(NP - len(trial_fitness), np.inf)
                ])

            # Selection
            old_fitness = fitness.copy()
            old_population = population.copy()
            population, fitness, replaced = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )

            # Update archive with replaced parents
            if np.any(replaced):
                self._update_archive_result = self._update_archive(
                    archive, old_population[replaced]
                )
                archive = self._update_archive_result

                # Compute fitness improvements for parameter adaptation
                improvements = old_fitness - fitness
                self._update_parameter_memory(cr_values, f_values, improvements, replaced)

            # Track best
            prev_best_f = best_f
            best_x, best_f = self._track_best(population, fitness, best_x, best_f)
            if best_f < prev_best_f:
                last_improvement_gen = generation

            # Periodic local search around best
            if generation % 25 == 0 and not stopping_condition():
                best_x, best_f = self._local_search_best(
                    best_x, best_f, func, stopping_condition
                )
                if stopping_condition():
                    break

            # Check for restart
            diversity = self._compute_diversity(population)
            if self._should_restart(diversity, generation, last_improvement_gen):
                population = self._opposition_based_restart(
                    population, fitness, best_x, best_f
                )
                population = self._clip_to_bounds(population)
                if stopping_condition():
                    break
                fitness = func(population)
                if stopping_condition():
                    valid = np.isfinite(fitness)
                    if np.any(valid):
                        idx = np.argmin(np.where(valid, fitness, np.inf))
                        if fitness[idx] < best_f:
                            best_f = fitness[idx]
                            best_x = population[idx].copy()
                    break
                # Ensure best is in population
                population[0] = best_x.copy()
                fitness[0] = best_f
                best_x, best_f = self._track_best(population, fitness, best_x, best_f)
                last_improvement_gen = generation
                archive = self._initialize_archive()

        return best_f, best_x
