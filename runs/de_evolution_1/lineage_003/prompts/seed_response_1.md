

```python
import numpy as np


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with:
    - Composite mutation: current-to-pbest/1 + archive-based mutation
    - Self-adaptive F and CR via success-history (SHADE-like)
    - Population size reduction over time
    - Stagnation detection with restarts
    - Opposition-based learning for restarts
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_init = min(8 * dim, 300)
        self.np_min = max(4, dim // 2)
        self.np_current = self.np_init
        self.history_size = 6
        self.p_best_rate = 0.11
        self.archive_rate = 2.6
        self.stagnation_threshold = 0.15 * dim
        self.restart_count = 0

    def _initialize_population(self):
        """Create initial random population within bounds."""
        pop = np.random.uniform(self.lb, self.ub, (self.np_current, self.dim))
        return pop

    def _initialize_history(self):
        """Initialize success history for F and CR parameters."""
        self.memory_f = np.full(self.history_size, 0.5)
        self.memory_cr = np.full(self.history_size, 0.5)
        self.memory_idx = 0
        self.archive = np.empty((0, self.dim))
        self.archive_max = int(self.archive_rate * self.np_current)
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def _generate_parameters_batch(self, np_current):
        """Generate F and CR for each individual from success history."""
        ri = np.random.randint(0, self.history_size, size=np_current)
        mu_f = self.memory_f[ri]
        mu_cr = self.memory_cr[ri]

        # Cauchy for F
        f_vals = mu_f + 0.1 * np.tan(np.pi * (np.random.random(np_current) - 0.5))
        f_vals = np.clip(f_vals, 0.01, 1.0)
        # Regenerate any <= 0
        bad = f_vals <= 0.01
        while np.any(bad):
            ri_bad = np.random.randint(0, self.history_size, size=np.sum(bad))
            f_vals[bad] = self.memory_f[ri_bad] + 0.1 * np.tan(
                np.pi * (np.random.random(np.sum(bad)) - 0.5)
            )
            f_vals = np.clip(f_vals, 0.01, 1.0)
            bad = f_vals <= 0.01

        # Normal for CR
        cr_vals = np.random.normal(mu_cr, 0.1, size=np_current)
        cr_vals = np.clip(cr_vals, 0.0, 1.0)

        return f_vals, cr_vals

    def _select_pbest_indices(self, fitness, np_current):
        """Select random p-best indices for each individual."""
        p = max(2, int(np.round(self.p_best_rate * np_current)))
        sorted_indices = np.argsort(fitness)
        pbest_pool = sorted_indices[:p]
        pbest_idx = pbest_pool[np.random.randint(0, p, size=np_current)]
        return pbest_idx

    def _mutate_batch(self, population, fitness, f_vals):
        """Current-to-pbest/1 mutation with optional archive usage."""
        np_current = len(population)
        pbest_idx = self._select_pbest_indices(fitness, np_current)

        # Random distinct indices r1, r2 (r2 can come from archive)
        r1 = np.random.randint(0, np_current, size=np_current)
        # Make sure r1 != current index
        indices = np.arange(np_current)
        mask = r1 == indices
        while np.any(mask):
            r1[mask] = np.random.randint(0, np_current, size=np.sum(mask))
            mask = r1 == indices

        # For r2, combine population with archive
        combined = np.vstack([population, self.archive]) if len(self.archive) > 0 else population
        n_combined = len(combined)
        r2 = np.random.randint(0, n_combined, size=np_current)
        # Make sure r2 != r1 and r2 != current index
        bad_mask = (r2 == r1) | (r2 == indices)
        while np.any(bad_mask):
            r2[bad_mask] = np.random.randint(0, n_combined, size=np.sum(bad_mask))
            bad_mask = (r2 == r1) | (r2 == indices)

        f_col = f_vals[:, np.newaxis]
        mutant = (
            population
            + f_col * (population[pbest_idx] - population)
            + f_col * (population[r1] - combined[r2])
        )
        return mutant

    def _crossover_batch(self, population, mutant, cr_vals):
        """Binomial crossover applied batch-wise."""
        np_current, dim = population.shape
        cr_col = cr_vals[:, np.newaxis]

        rand_matrix = np.random.random((np_current, dim))
        j_rand = np.random.randint(0, dim, size=np_current)

        cross_mask = rand_matrix < cr_col
        # Ensure at least one dimension is from mutant
        cross_mask[np.arange(np_current), j_rand] = True

        trial = np.where(cross_mask, mutant, population)
        return trial

    def _clip_to_bounds(self, trial, population):
        """Clip trials to bounds; use midpoint reflection towards parent."""
        below = trial < self.lb
        above = trial > self.ub
        # Midpoint between bound and parent for reflection
        trial = np.where(below, (self.lb + population) / 2.0, trial)
        trial = np.where(above, (self.ub + population) / 2.0, trial)
        trial = np.clip(trial, self.lb, self.ub)
        return trial

    def _select_survivors_batch(self, population, fitness, trial, trial_fitness):
        """Greedy selection: keep better of parent and trial."""
        n_valid = min(len(fitness), len(trial_fitness))
        population = population[:n_valid]
        fitness = fitness[:n_valid]
        trial = trial[:n_valid]
        trial_fitness = trial_fitness[:n_valid]

        improved = trial_fitness <= fitness
        # Filter NaN
        valid = np.isfinite(trial_fitness)
        improved = improved & valid

        # Collect successful parameters info
        success_mask = improved

        # Archive replaced individuals
        replaced = population[improved]
        if len(replaced) > 0:
            self.archive = np.vstack([self.archive, replaced]) if len(self.archive) > 0 else replaced.copy()
            if len(self.archive) > self.archive_max:
                keep_idx = np.random.choice(len(self.archive), self.archive_max, replace=False)
                self.archive = self.archive[keep_idx]

        new_pop = np.where(improved[:, np.newaxis], trial, population)
        new_fit = np.where(improved, trial_fitness, fitness)

        return new_pop, new_fit, success_mask

    def _update_memory(self, f_vals, cr_vals, fitness, trial_fitness, success_mask):
        """Update success history memory for F and CR (weighted Lehmer mean)."""
        if not np.any(success_mask):
            return

        s_f = f_vals[success_mask]
        s_cr = cr_vals[success_mask]
        delta_f = np.abs(fitness[success_mask] - trial_fitness[success_mask])

        if np.sum(delta_f) < 1e-30:
            return

        weights = delta_f / np.sum(delta_f)

        # Weighted Lehmer mean for F
        mean_f_num = np.sum(weights * s_f ** 2)
        mean_f_den = np.sum(weights * s_f)
        if mean_f_den > 1e-30:
            new_f = mean_f_num / mean_f_den
        else:
            new_f = self.memory_f[self.memory_idx]

        # Weighted mean for CR
        new_cr = np.sum(weights * s_cr)

        self.memory_f[self.memory_idx] = new_f
        self.memory_cr[self.memory_idx] = new_cr
        self.memory_idx = (self.memory_idx + 1) % self.history_size

    def _detect_stagnation(self, best_fitness):
        """Detect if the search has stagnated based on fitness history."""
        self.best_fitness_history.append(best_fitness)
        window = int(self.stagnation_threshold)

        if len(self.best_fitness_history) < window:
            return False

        recent = self.best_fitness_history[-window:]
        improvement = abs(recent[0] - recent[-1])
        scale = max(abs(recent[0]), 1e-10)
        relative_improvement = improvement / scale

        if relative_improvement < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        return self.stagnation_counter >= 3

    def _restart(self, best_x, best_f):
        """Restart population using opposition-based learning around best."""
        self.restart_count += 1
        np_new = max(self.np_min, self.np_current)
        
        # Half random, half opposition-based around best
        n_random = np_new // 2
        n_opposition = np_new - n_random

        random_pop = np.random.uniform(self.lb, self.ub, (n_random, self.dim))

        # Opposition-based: reflect through midpoint with perturbation
        center = (self.lb + self.ub) / 2.0
        opp_base = 2.0 * center - best_x
        perturbation = np.random.normal(0, 20.0, (n_opposition, self.dim))
        opp_pop = opp_base + perturbation
        opp_pop = np.clip(opp_pop, self.lb, self.ub)

        new_pop = np.vstack([random_pop, opp_pop])
        # Keep best solution
        new_pop[0] = best_x.copy()

        # Reset adaptation
        self.archive = np.empty((0, self.dim))
        self.stagnation_counter = 0
        self.best_fitness_history = []
        # Partially reset memory (keep some learned info)
        self.memory_f = np.clip(self.memory_f * 0.8 + 0.1, 0.1, 0.9)
        self.memory_cr = np.clip(self.memory_cr * 0.8 + 0.1, 0.1, 0.9)

        return new_pop

    def _reduce_population(self, population, fitness, generation, max_generations):
        """Linearly reduce population size over time (L-SHADE style)."""
        progress = min(generation / max(max_generations, 1), 1.0)
        target_np = int(round(self.np_init - (self.np_init - self.np_min) * progress))
        target_np = max(self.np_min, target_np)

        if target_np < len(population) and len(population) > self.np_min:
            n_remove = len(population) - target_np
            sorted_idx = np.argsort(fitness)
            keep_idx = sorted_idx[:target_np]
            population = population[keep_idx]
            fitness = fitness[keep_idx]
            self.np_current = target_np
            self.archive_max = int(self.archive_rate * self.np_current)
            if len(self.archive) > self.archive_max:
                keep = np.random.choice(len(self.archive), self.archive_max, replace=False)
                self.archive = self.archive[keep]

        return population, fitness

    def _local_search_batch(self, population, fitness, best_x, best_f):
        """Small local search around the best individual for exploitation."""
        n_local = min(self.dim, 10)
        sigma = 0.01 * (self.ub - self.lb)
        perturbations = np.random.normal(0, sigma, (n_local, self.dim))
        local_trials = best_x + perturbations
        local_trials = np.clip(local_trials, self.lb, self.ub)
        return local_trials

    def _handle_eval_result(self, trial_fitness, n_expected):
        """Handle truncated evaluation results gracefully."""
        if trial_fitness is None or len(trial_fitness) == 0:
            return np.array([]), True
        n_valid = len(trial_fitness)
        if n_valid < n_expected:
            return trial_fitness, True
        return trial_fitness, False

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        self.np_current = self.np_init
        population = self._initialize_population()
        self._initialize_history()

        # Evaluate initial population
        fitness = func(population)
        if stopping_condition():
            best_idx = np.nanargmin(fitness)
            return float(fitness[best_idx]), population[best_idx].copy()

        fitness, truncated = self._handle_eval_result(fitness, self.np_current)
        if truncated:
            population = population[: len(fitness)]

        # Track best
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return 0.0, population[0].copy()
        best_idx = np.nanargmin(fitness)
        best_f = float(fitness[best_idx])
        best_x = population[best_idx].copy()

        generation = 0
        # Rough estimate of max generations
        est_max_gen = 10000 // max(self.np_current, 1) * 3

        while not stopping_condition():
            generation += 1
            np_current = len(population)

            # Generate adaptive parameters
            f_vals, cr_vals = self._generate_parameters_batch(np_current)

            # Mutation
            mutant = self._mutate_batch(population, fitness, f_vals)

            # Crossover
            trial = self._crossover_batch(population, mutant, cr_vals)

            # Clip
            trial = self._clip_to_bounds(trial, population)

            # Evaluate trials
            trial_fitness = func(trial)
            if stopping_condition():
                # Check if any trial is better before breaking
                tf, trunc = self._handle_eval_result(trial_fitness, np_current)
                if len(tf) > 0:
                    valid = np.isfinite(tf)
                    if np.any(valid):
                        bi = np.nanargmin(tf)
                        if tf[bi] < best_f:
                            best_f = float(tf[bi])
                            best_x = trial[bi].copy()
                break

            trial_fitness, truncated = self._handle_eval_result(trial_fitness, np_current)
            if truncated or len(trial_fitness) < np_current:
                n_valid = len(trial_fitness)
                trial = trial[:n_valid]
                population = population[:n_valid]
                fitness = fitness[:n_valid]
                f_vals = f_vals[:n_valid]
                cr_vals = cr_vals[:n_valid]
                np_current = n_valid
                if np_current == 0:
                    break

            # Selection
            old_fitness = fitness.copy()
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trial, trial_fitness
            )

            # Update best
            valid_mask = np.isfinite(fitness)
            if np.any(valid_mask):
                curr_best_idx = np.nanargmin(fitness)
                if fitness[curr_best_idx] < best_f:
                    best_f = float(fitness[curr_best_idx])
                    best_x = population[curr_best_idx].copy()

            # Update memory
            n_sel = min(len(f_vals), len(success_mask))
            self._update_memory(
                f_vals[:n_sel], cr_vals[:n_sel],
                old_fitness[:n_sel], trial_fitness[:n_sel],
                success_mask[:n_sel]
            )

            # Population reduction
            population, fitness = self._reduce_population(
                population, fitness, generation, est_max_gen
            )

            # Periodic local search around best
            if generation % 25 == 0 and not stopping_condition():
                local_trials = self._local_search_batch(population, fitness, best_x, best_f)
                local_fit = func(local_trials)
                if stopping_condition():
                    lf, _ = self._handle_eval_result(local_fit, len(local_trials))
                    if len(lf) > 0:
                        valid = np.isfinite(lf)
                        if np.any(valid):
                            bi = np.nanargmin(lf)
                            if lf[bi] < best_f:
                                best_f = float(lf[bi])
                                best_x = local_trials[bi].copy()
                    break
                lf, _ = self._handle_eval_result(local_fit, len(local_trials))
                if len(lf) > 0:
                    valid = np.isfinite(lf)
                    if np.any(valid):
                        bi = np.nanargmin(lf)
                        if lf[bi] < best_f:
                            best_f = float(lf[bi])
                            best_x = local_trials[bi].copy()

            # Detect stagnation and restart
            if self._detect_stagnation(best_f) and not stopping_condition():
                population = self._restart(best_x, best_f)
                self.np_current = len(population)
                fitness = func(population)
                if stopping_condition():
                    ff, _ = self._handle_eval_result(fitness, len(population))
                    if len(ff) > 0:
                        valid = np.isfinite(ff)
                        if np.any(valid):
                            bi = np.nanargmin(ff)
                            if ff[bi] < best_f:
                                best_f = float(ff[bi])
                                best_x = population[bi].copy()
                    break
                fitness, truncated = self._handle_eval_result(fitness, len(population))
                if truncated:
                    population = population[: len(fitness)]
                # Inject best
                if len(population) > 0:
                    worst_idx = np.nanargmax(fitness)
                    population[worst_idx] = best_x.copy()
                    fitness[worst_idx] = best_f
                valid_mask = np.isfinite(fitness)
                if np.any(valid_mask):
                    curr_best_idx = np.nanargmin(fitness)
                    if fitness[curr_best_idx] < best_f:
                        best_f = float(fitness[curr_best_idx])
                        best_x = population[curr_best_idx].copy()

        return best_f, best_x
```