

```python
import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with:
    - SHADE-like parameter adaptation (F and CR)
    - Current-to-pbest/1 mutation with archive
    - Rank-based weighted recombination ideas
    - Stagnation detection and strategic restarts
    - Opposition-based learning on restart
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 60), 300)
        self.archive_max = self.np_size
        self.h_size = 50  # memory size for SHADE
        self.p_min = 0.05
        self.p_max = 0.25
        self.stagnation_threshold = 50
        self.restart_count = 0
        self.max_restarts = 10

    def _initialize_population(self):
        """Create initial random population uniformly in bounds."""
        pop = np.random.uniform(self.lb, self.ub, size=(self.np_size, self.dim))
        return pop

    def _opposition_based_population(self, pop):
        """Generate opposition-based population for diversity."""
        opp = self.lb + self.ub - pop
        return np.clip(opp, self.lb, self.ub)

    def _initialize_memory(self):
        """Initialize SHADE memory for F and CR."""
        self.memory_f = np.full(self.h_size, 0.5)
        self.memory_cr = np.full(self.h_size, 0.5)
        self.memory_idx = 0

    def _initialize_archive(self):
        """Initialize empty archive for storing replaced individuals."""
        self.archive = np.empty((0, self.dim))

    def _initialize_tracking(self):
        """Initialize best solution tracking and stagnation counter."""
        self.best_f = np.inf
        self.best_x = None
        self.stagnation_counter = 0
        self.prev_best_f = np.inf

    def _generate_parameters_batch(self):
        """Generate F and CR for the entire population using Cauchy/Normal from memory."""
        ri = np.random.randint(0, self.h_size, size=self.np_size)
        mu_f = self.memory_f[ri]
        mu_cr = self.memory_cr[ri]

        # Cauchy distribution for F
        f_vals = mu_f + 0.1 * np.tan(np.pi * (np.random.uniform(size=self.np_size) - 0.5))
        # Truncate F: regenerate if <= 0, cap at 1
        for _ in range(10):
            bad = f_vals <= 0
            if not np.any(bad):
                break
            ri_bad = np.random.randint(0, self.h_size, size=np.sum(bad))
            f_vals[bad] = self.memory_f[ri_bad] + 0.1 * np.tan(
                np.pi * (np.random.uniform(size=np.sum(bad)) - 0.5)
            )
        f_vals = np.clip(f_vals, 0.01, 1.0)

        # Normal distribution for CR
        cr_vals = np.random.normal(mu_cr, 0.1, size=self.np_size)
        cr_vals = np.clip(cr_vals, 0.0, 1.0)

        return f_vals, cr_vals

    def _select_pbest_indices(self, fitness):
        """Select p-best indices for current-to-pbest mutation."""
        p = np.random.uniform(self.p_min, self.p_max)
        p_count = max(1, int(np.ceil(p * self.np_size)))
        sorted_idx = np.argsort(fitness)
        pbest_pool = sorted_idx[:p_count]
        chosen = pbest_pool[np.random.randint(0, p_count, size=self.np_size)]
        return chosen

    def _select_distinct_indices(self, pop_size, archive_size):
        """Select r1 from population and r2 from population+archive, distinct from each other and i."""
        r1 = np.empty(self.np_size, dtype=int)
        r2 = np.empty(self.np_size, dtype=int)
        total = pop_size + archive_size

        for i in range(self.np_size):
            candidates_r1 = list(range(pop_size))
            candidates_r1.remove(i)
            r1[i] = candidates_r1[np.random.randint(len(candidates_r1))]

            candidates_r2 = list(range(total))
            candidates_r2 = [c for c in candidates_r2 if c != i and c != r1[i]]
            r2[i] = candidates_r2[np.random.randint(len(candidates_r2))]

        return r1, r2

    def _select_distinct_indices_vectorized(self, pop_size, archive_size):
        """Vectorized selection of r1 and r2 indices, distinct from i."""
        total = pop_size + archive_size
        indices = np.arange(self.np_size)

        # r1 from population, != i
        r1 = np.random.randint(0, pop_size - 1, size=self.np_size)
        r1 = r1 + (r1 >= indices).astype(int)

        # r2 from population + archive, != i and != r1
        r2 = np.random.randint(0, total - 2, size=self.np_size)
        # Shift to avoid i
        r2 = r2 + (r2 >= np.minimum(indices, r1)).astype(int)
        r2 = r2 + (r2 >= np.maximum(indices, r1)).astype(int)
        # Ensure within bounds
        r2 = np.clip(r2, 0, total - 1)

        return r1, r2

    def _mutate_batch(self, pop, fitness, f_vals):
        """Current-to-pbest/1 mutation with archive."""
        pbest_idx = self._select_pbest_indices(fitness)
        archive_size = len(self.archive)

        # Combine pop and archive for donor selection
        if archive_size > 0:
            combined = np.vstack([pop, self.archive])
        else:
            combined = pop.copy()

        r1, r2 = self._select_distinct_indices_vectorized(len(pop), archive_size)

        x_pbest = pop[pbest_idx]
        x_r1 = pop[r1]
        x_r2 = combined[r2]

        f_col = f_vals[:, np.newaxis]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)

        return mutant

    def _crossover_batch(self, pop, mutant, cr_vals):
        """Binomial crossover applied to entire population batch."""
        mask = np.random.uniform(size=(self.np_size, self.dim)) < cr_vals[:, np.newaxis]
        # Ensure at least one dimension is from mutant
        j_rand = np.random.randint(0, self.dim, size=self.np_size)
        mask[np.arange(self.np_size), j_rand] = True

        trial = np.where(mask, mutant, pop)
        return trial

    def _clip_to_bounds(self, trial, pop):
        """Clip trial vectors to bounds using midpoint reflection."""
        below = trial < self.lb
        above = trial > self.ub
        # Midpoint between parent and bound for out-of-bounds dimensions
        trial = np.where(below, (pop + self.lb) / 2.0, trial)
        trial = np.where(above, (pop + self.ub) / 2.0, trial)
        trial = np.clip(trial, self.lb, self.ub)
        return trial

    def _select_survivors_batch(self, pop, fitness, trial, trial_fitness, f_vals, cr_vals):
        """Greedy selection: keep better of parent or trial. Update archive and memory."""
        improved = trial_fitness < fitness
        valid = ~np.isnan(trial_fitness)
        improved = improved & valid

        # Collect successful F and CR values with fitness improvement weights
        s_f = f_vals[improved]
        s_cr = cr_vals[improved]
        s_delta = np.abs(fitness[improved] - trial_fitness[improved])

        # Update archive with replaced parents
        if np.any(improved):
            replaced = pop[improved]
            self._update_archive(replaced)

        # Update memory
        if len(s_f) > 0:
            self._update_memory(s_f, s_cr, s_delta)

        # Survivor selection
        new_pop = np.where(improved[:, np.newaxis], trial, pop)
        new_fitness = np.where(improved, trial_fitness, fitness)

        return new_pop, new_fitness

    def _update_archive(self, replaced_individuals):
        """Add replaced individuals to archive, trim if too large."""
        self.archive = np.vstack([self.archive, replaced_individuals]) if len(self.archive) > 0 else replaced_individuals.copy()
        if len(self.archive) > self.archive_max:
            keep = np.random.choice(len(self.archive), self.archive_max, replace=False)
            self.archive = self.archive[keep]

    def _update_memory(self, s_f, s_cr, s_delta):
        """Update SHADE memory with weighted Lehmer mean for F and weighted mean for CR."""
        if len(s_f) == 0:
            return
        weights = s_delta / (np.sum(s_delta) + 1e-30)

        # Weighted Lehmer mean for F
        mean_f = np.sum(weights * s_f ** 2) / (np.sum(weights * s_f) + 1e-30)
        mean_f = np.clip(mean_f, 0.01, 1.0)

        # Weighted arithmetic mean for CR
        mean_cr = np.sum(weights * s_cr)
        mean_cr = np.clip(mean_cr, 0.0, 1.0)

        self.memory_f[self.memory_idx] = mean_f
        self.memory_cr[self.memory_idx] = mean_cr
        self.memory_idx = (self.memory_idx + 1) % self.h_size

    def _update_best(self, pop, fitness):
        """Track the global best solution found."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_idx = np.where(valid)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.best_f:
            self.best_f = fitness[best_idx]
            self.best_x = pop[best_idx].copy()

    def _detect_stagnation(self, fitness):
        """Check if the population has stagnated based on improvement history."""
        current_best = np.nanmin(fitness)
        rel_improvement = abs(self.prev_best_f - current_best) / (abs(self.prev_best_f) + 1e-30)

        if rel_improvement < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        self.prev_best_f = current_best
        return self.stagnation_counter > self.stagnation_threshold

    def _compute_diversity(self, pop):
        """Compute population diversity as mean pairwise distance to centroid."""
        centroid = np.mean(pop, axis=0)
        dists = np.sqrt(np.sum((pop - centroid) ** 2, axis=1))
        return np.mean(dists)

    def _restart(self, pop, fitness):
        """Restart population with opposition-based learning, keeping best."""
        self.restart_count += 1
        self.stagnation_counter = 0

        # Keep top 10% of population
        n_keep = max(2, self.np_size // 10)
        sorted_idx = np.argsort(fitness)
        elite = pop[sorted_idx[:n_keep]].copy()
        elite_fit = fitness[sorted_idx[:n_keep]].copy()

        # Generate new random individuals
        n_new = self.np_size - 2 * n_keep
        n_opp = n_keep

        new_random = np.random.uniform(self.lb, self.ub, size=(max(0, n_new), self.dim))

        # Opposition of elite
        opp_elite = self._opposition_based_population(elite)

        parts = [elite]
        if len(opp_elite) > 0:
            parts.append(opp_elite)
        if n_new > 0:
            parts.append(new_random)

        new_pop = np.vstack(parts)[:self.np_size]

        # Add some perturbation around best known solution
        if self.best_x is not None and len(new_pop) >= 3:
            n_local = min(5, len(new_pop) - n_keep)
            scale = self._compute_diversity(pop) * 0.1 + 1.0
            local_pop = self.best_x + np.random.normal(0, scale, size=(n_local, self.dim))
            local_pop = np.clip(local_pop, self.lb, self.ub)
            new_pop[n_keep:n_keep + n_local] = local_pop

        new_pop = np.clip(new_pop, self.lb, self.ub)

        # Re-initialize memory with slightly different values
        self._initialize_memory()
        self.memory_f[:] = np.random.uniform(0.4, 0.7, self.h_size)
        self.memory_cr[:] = np.random.uniform(0.3, 0.9, self.h_size)
        self._initialize_archive()

        return new_pop

    def _local_search_batch(self, pop, fitness, func):
        """Small local search around the best individual using Gaussian perturbation."""
        if self.best_x is None:
            return pop, fitness

        n_local = min(self.dim, self.np_size // 4)
        scale = max(0.01, self._compute_diversity(pop) * 0.01)

        local_trials = self.best_x + np.random.normal(0, scale, size=(n_local, self.dim))
        local_trials = np.clip(local_trials, self.lb, self.ub)

        local_fit = func(local_trials)
        valid = ~np.isnan(local_fit)

        if np.any(valid):
            best_local_idx = np.where(valid)[0][np.argmin(local_fit[valid])]
            if local_fit[best_local_idx] < self.best_f:
                self.best_f = local_fit[best_local_idx]
                self.best_x = local_trials[best_local_idx].copy()

            # Replace worst individuals in population with good local trials
            good_local = local_fit < np.median(fitness)
            if np.any(good_local & valid):
                worst_idx = np.argsort(fitness)[-np.sum(good_local & valid):]
                replace_trials = local_trials[good_local & valid]
                replace_fit = local_fit[good_local & valid]
                n_replace = min(len(worst_idx), len(replace_trials))
                pop[worst_idx[:n_replace]] = replace_trials[:n_replace]
                fitness[worst_idx[:n_replace]] = replace_fit[:n_replace]

        return pop, fitness

    def _handle_truncated_results(self, trial_fitness, trial, pop, fitness):
        """Handle case where func returns fewer results than expected."""
        n_valid = len(trial_fitness)
        if n_valid < len(trial):
            trial = trial[:n_valid]
            # Only process the valid portion
        return trial, trial_fitness

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize all components
        pop = self._initialize_population()
        self._initialize_memory()
        self._initialize_archive()
        self._initialize_tracking()

        # Evaluate initial population
        fitness = func(pop)
        if stopping_condition():
            self._update_best(pop, fitness)
            return self.best_f, self.best_x

        if len(fitness) < len(pop):
            pop = pop[:len(fitness)]
            self.np_size = len(pop)

        self._update_best(pop, fitness)
        generation = 0

        while not stopping_condition():
            generation += 1

            # Generate adaptive parameters
            f_vals, cr_vals = self._generate_parameters_batch()

            # Mutation
            mutant = self._mutate_batch(pop, fitness, f_vals)

            # Crossover
            trial = self._crossover_batch(pop, mutant, cr_vals)

            # Clip to bounds
            trial = self._clip_to_bounds(trial, pop)

            # Evaluate trials
            trial_fitness = func(trial)

            if stopping_condition():
                # Still update best from whatever we got
                valid = ~np.isnan(trial_fitness[:len(trial_fitness)])
                if np.any(valid):
                    for i in range(len(trial_fitness)):
                        if valid[i] and trial_fitness[i] < self.best_f:
                            self.best_f = trial_fitness[i]
                            self.best_x = trial[i].copy()
                break

            # Handle truncated results
            if len(trial_fitness) < len(trial):
                trial = trial[:len(trial_fitness)]
                f_vals = f_vals[:len(trial_fitness)]
                cr_vals = cr_vals[:len(trial_fitness)]
                pop = pop[:len(trial_fitness)]
                fitness = fitness[:len(trial_fitness)]
                self.np_size = len(trial_fitness)

            # Selection
            pop, fitness = self._select_survivors_batch(
                pop, fitness, trial, trial_fitness, f_vals, cr_vals
            )

            # Update best
            self._update_best(pop, fitness)

            # Periodic local search every 25 generations
            if generation % 25 == 0 and not stopping_condition():
                pop, fitness = self._local_search_batch(pop, fitness, func)
                if stopping_condition():
                    break
                self._update_best(pop, fitness)

            # Check stagnation and restart if needed
            if self._detect_stagnation(fitness):
                if self.restart_count < self.max_restarts:
                    pop = self._restart(pop, fitness)
                    fitness = func(pop)
                    if stopping_condition():
                        self._update_best(pop, fitness)
                        break
                    if len(fitness) < len(pop):
                        pop = pop[:len(fitness)]
                        self.np_size = len(pop)
                    self._update_best(pop, fitness)
                else:
                    # Reset stagnation counter even if we don't restart
                    self.stagnation_counter = 0

        return self.best_f, self.best_x
```