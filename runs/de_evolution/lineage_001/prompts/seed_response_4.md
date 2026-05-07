

```python
import numpy as np


class HybridDESimplexOptimizer:
    """
    Hybrid optimizer combining Differential Evolution with Simplex-based
    local refinement. Uses adaptive parameter control, archive-based mutation,
    stagnation detection with restarts, and periodic Nelder-Mead-like
    simplex moves on elite subsets.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 60), 300)
        self.archive_max = self.np_size
        self.F = 0.5
        self.CR = 0.9
        self.F_memory = np.full(5, 0.5)
        self.CR_memory = np.full(5, 0.9)
        self.memory_idx = 0
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.restart_count = 0
        self.simplex_fraction = 0.15  # fraction of population for simplex moves

    def __call__(self, func, stopping_condition):
        population, fitness = self._initialize_population(func)
        if stopping_condition():
            return self._get_best(population, fitness)

        archive = np.empty((0, self.dim))
        best_f = np.nanmin(fitness)
        best_x = population[np.nanargmin(fitness)].copy()
        prev_best_f = best_f
        generation = 0

        while not stopping_condition():
            # Adaptive parameters
            F_vals, CR_vals = self._generate_parameters_batch()

            # DE mutation and crossover
            mutants = self._mutate_batch(population, fitness, archive, F_vals)
            trials = self._crossover_batch(population, mutants, CR_vals)
            trials = self._clip_bounds(trials)

            # Evaluate trials
            trial_fitness = func(trials)
            if stopping_condition():
                self._update_best_from_eval(trials, trial_fitness, best_x, best_f)
                valid = ~np.isnan(trial_fitness)
                if np.any(valid):
                    idx = np.nanargmin(trial_fitness)
                    if trial_fitness[idx] < best_f:
                        best_f = trial_fitness[idx]
                        best_x = trials[idx].copy()
                break

            # Handle truncated returns
            if len(trial_fitness) < len(trials):
                trial_fitness = np.append(trial_fitness, 
                    np.full(len(trials) - len(trial_fitness), np.nan))

            # Selection and archive update
            population, fitness, archive, success_F, success_CR = self._select_survivors_batch(
                population, fitness, trials, trial_fitness, archive, F_vals, CR_vals
            )

            # Update parameter memory
            self._adapt_parameters(success_F, success_CR)

            # Track best
            curr_min_idx = np.nanargmin(fitness)
            curr_min_f = fitness[curr_min_idx]
            if curr_min_f < best_f:
                best_f = curr_min_f
                best_x = population[curr_min_idx].copy()

            # Stagnation detection
            stagnated = self._detect_stagnation(best_f, prev_best_f)
            prev_best_f = best_f

            # Simplex refinement on elite subset periodically
            if generation % 5 == 0 and not stopping_condition():
                population, fitness, sx, sf = self._simplex_refine_batch(
                    population, fitness, func, stopping_condition
                )
                if sf is not None and sf < best_f:
                    best_f = sf
                    best_x = sx.copy()
                if stopping_condition():
                    break

            # Restart if stagnated
            if stagnated and not stopping_condition():
                population, fitness, archive = self._restart(
                    population, fitness, best_x, best_f, func, stopping_condition
                )
                if stopping_condition():
                    break
                curr_min_idx = np.nanargmin(fitness)
                if fitness[curr_min_idx] < best_f:
                    best_f = fitness[curr_min_idx]
                    best_x = population[curr_min_idx].copy()

            generation += 1

        return best_f, best_x

    def _initialize_population(self, func):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < self.np_size:
            fitness = np.append(fitness, np.full(self.np_size - len(fitness), np.nan))
        return population, fitness

    def _clip_bounds(self, x):
        """Clip all candidates to search bounds."""
        return np.clip(x, self.lb, self.ub)

    def _generate_parameters_batch(self):
        """Generate F and CR values for entire population using Cauchy/Normal from memory."""
        n = self.np_size
        # Pick random memory indices
        mem_indices = np.random.randint(0, len(self.F_memory), size=n)
        
        F_vals = np.empty(n)
        for i in range(n):
            f = -1
            while f <= 0:
                f = self.F_memory[mem_indices[i]] + 0.1 * np.random.standard_cauchy()
                if f > 1.0:
                    f = 1.0
            F_vals[i] = f
        
        CR_vals = np.clip(
            self.CR_memory[mem_indices] + 0.1 * np.random.randn(n),
            0.0, 1.0
        )
        return F_vals, CR_vals

    def _mutate_batch(self, population, fitness, archive, F_vals):
        """Current-to-pbest/1 mutation with archive."""
        n = self.np_size
        dim = self.dim
        
        # Select pbest indices (top p fraction, p in [0.05, 0.2])
        p = max(2, int(0.15 * n))
        sorted_indices = np.argsort(fitness)
        pbest_candidates = sorted_indices[:p]
        pbest_idx = pbest_candidates[np.random.randint(0, p, size=n)]
        
        # Random indices r1 != i
        r1 = np.random.randint(0, n, size=n)
        for i in range(n):
            while r1[i] == i:
                r1[i] = np.random.randint(0, n)
        
        # r2 from population + archive, != i, != r1
        combined = np.vstack([population, archive]) if len(archive) > 0 else population.copy()
        n_combined = len(combined)
        r2 = np.random.randint(0, n_combined, size=n)
        for i in range(n):
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, n_combined)
        
        F_broad = F_vals[:, np.newaxis]
        mutants = (population + 
                   F_broad * (population[pbest_idx] - population) +
                   F_broad * (population[r1] - combined[r2]))
        
        return mutants

    def _crossover_batch(self, population, mutants, CR_vals):
        """Binomial crossover for entire population."""
        n, dim = population.shape
        CR_broad = CR_vals[:, np.newaxis]
        
        # Random crossover mask
        rand_matrix = np.random.rand(n, dim)
        mask = rand_matrix < CR_broad
        
        # Ensure at least one dimension from mutant
        j_rand = np.random.randint(0, dim, size=n)
        mask[np.arange(n), j_rand] = True
        
        trials = np.where(mask, mutants, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, 
                                 archive, F_vals, CR_vals):
        """Greedy selection: keep better of parent vs trial. Update archive."""
        valid = ~np.isnan(trial_fitness) & ~np.isnan(fitness)
        improved = valid & (trial_fitness <= fitness)
        
        # Collect successful parameters
        success_F = F_vals[improved] if np.any(improved) else np.array([])
        success_CR = CR_vals[improved] if np.any(improved) else np.array([])
        
        # Add replaced parents to archive
        if np.any(improved):
            new_archive_entries = population[improved]
            archive = np.vstack([archive, new_archive_entries]) if len(archive) > 0 else new_archive_entries.copy()
            # Trim archive
            if len(archive) > self.archive_max:
                keep = np.random.choice(len(archive), self.archive_max, replace=False)
                archive = archive[keep]
        
        # Replace improved individuals
        population[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]
        
        return population, fitness, archive, success_F, success_CR

    def _adapt_parameters(self, success_F, success_CR):
        """Update parameter memory with successful F and CR values (SHADE-style)."""
        if len(success_F) == 0:
            return
        
        # Lehmer mean for F
        weights = success_F  # weight by improvement magnitude proxy
        w_sum = np.sum(weights)
        if w_sum > 0:
            w = weights / w_sum
            new_F = np.sum(w * success_F**2) / (np.sum(w * success_F) + 1e-30)
            new_CR = np.sum(w * success_CR)
        else:
            new_F = np.mean(success_F)
            new_CR = np.mean(success_CR)
        
        self.F_memory[self.memory_idx] = np.clip(new_F, 0.1, 1.0)
        self.CR_memory[self.memory_idx] = np.clip(new_CR, 0.0, 1.0)
        self.memory_idx = (self.memory_idx + 1) % len(self.F_memory)

    def _detect_stagnation(self, best_f, prev_best_f):
        """Detect if optimization has stagnated."""
        rel_improvement = abs(prev_best_f - best_f) / (abs(prev_best_f) + 1e-30)
        if rel_improvement < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 2)
        
        if self.stagnation_counter >= self.stagnation_limit:
            self.stagnation_counter = 0
            return True
        return False

    def _restart(self, population, fitness, best_x, best_f, func, stopping_condition):
        """Restart population while keeping best solution and some elites."""
        self.restart_count += 1
        n = self.np_size
        dim = self.dim
        
        # Keep top 10% elites
        n_elite = max(2, n // 10)
        sorted_idx = np.argsort(fitness)
        elites = population[sorted_idx[:n_elite]].copy()
        elite_fit = fitness[sorted_idx[:n_elite]].copy()
        
        # Generate new random individuals
        n_new = n - n_elite - 1
        new_pop = np.random.uniform(self.lb, self.ub, (n_new, dim))
        
        # Also include perturbed best
        scale = max(10.0 / (self.restart_count + 1), 0.5)
        perturbed_best = best_x + scale * np.random.randn(1, dim)
        perturbed_best = self._clip_bounds(perturbed_best)
        
        population = np.vstack([elites, perturbed_best, new_pop])
        population = self._clip_bounds(population)
        
        if stopping_condition():
            fitness = np.full(n, np.nan)
            fitness[:n_elite] = elite_fit
            return population, fitness, np.empty((0, dim))
        
        # Evaluate new members
        new_members = population[n_elite:]
        new_fit = func(new_members)
        if len(new_fit) < len(new_members):
            new_fit = np.append(new_fit, np.full(len(new_members) - len(new_fit), np.nan))
        
        fitness = np.concatenate([elite_fit, new_fit])
        
        # Reset parameter memory partially
        self.F_memory = np.full(5, 0.5)
        self.CR_memory = np.full(5, 0.9)
        self.memory_idx = 0
        
        archive = np.empty((0, dim))
        return population, fitness, archive

    def _simplex_refine_batch(self, population, fitness, func, stopping_condition):
        """Apply Nelder-Mead-like simplex operations on elite subset."""
        n = self.np_size
        n_simplex = max(3, int(self.simplex_fraction * n))
        
        sorted_idx = np.argsort(fitness)
        elite_idx = sorted_idx[:n_simplex]
        elites = population[elite_idx].copy()
        elite_fit = fitness[elite_idx].copy()
        
        # Compute centroid of all but worst
        centroid = np.mean(elites[:-1], axis=0)
        
        # Generate reflection, expansion, contraction points as a batch
        worst = elites[-1]
        alpha, gamma, rho = 1.0, 2.0, 0.5
        
        reflected = centroid + alpha * (centroid - worst)
        expanded = centroid + gamma * (reflected - centroid)
        contracted_out = centroid + rho * (reflected - centroid)
        contracted_in = centroid + rho * (worst - centroid)
        
        # Also generate shrink moves toward best for remaining elites
        best_point = elites[0]
        shrunk = best_point + 0.5 * (elites[1:] - best_point)
        
        # Batch all candidates together
        candidates = np.vstack([
            reflected.reshape(1, -1),
            expanded.reshape(1, -1),
            contracted_out.reshape(1, -1),
            contracted_in.reshape(1, -1),
            shrunk
        ])
        candidates = self._clip_bounds(candidates)
        
        if stopping_condition():
            return population, fitness, None, None
        
        cand_fit = func(candidates)
        if stopping_condition() or len(cand_fit) == 0:
            if len(cand_fit) > 0:
                valid = ~np.isnan(cand_fit)
                if np.any(valid):
                    bi = np.nanargmin(cand_fit)
                    return population, fitness, candidates[bi], cand_fit[bi]
            return population, fitness, None, None
        
        if len(cand_fit) < len(candidates):
            cand_fit = np.append(cand_fit, np.full(len(candidates) - len(cand_fit), np.nan))
        
        # Find best candidate
        valid_mask = ~np.isnan(cand_fit)
        if not np.any(valid_mask):
            return population, fitness, None, None
        
        best_cand_idx = np.nanargmin(cand_fit)
        best_cand_f = cand_fit[best_cand_idx]
        best_cand_x = candidates[best_cand_idx]
        
        # Replace worst elite if improvement found
        worst_elite_idx = elite_idx[-1]
        if best_cand_f < fitness[worst_elite_idx]:
            population[worst_elite_idx] = best_cand_x
            fitness[worst_elite_idx] = best_cand_f
        
        # Also try replacing other elites with shrink results if better
        shrink_fits = cand_fit[4:]
        for i, si in enumerate(elite_idx[1:]):
            if i < len(shrink_fits) and not np.isnan(shrink_fits[i]):
                if shrink_fits[i] < fitness[si]:
                    population[si] = shrunk[i]
                    fitness[si] = shrink_fits[i]
        
        # Return overall best found
        overall_best_idx = np.nanargmin(fitness)
        return population, fitness, population[overall_best_idx], fitness[overall_best_idx]

    def _get_best(self, population, fitness):
        """Return best fitness and corresponding solution."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return np.inf, population[0]
        idx = np.nanargmin(fitness)
        return fitness[idx], population[idx].copy()

    def _update_best_from_eval(self, trials, trial_fitness, best_x, best_f):
        """Helper to check if any trial improves on best after final eval."""
        valid = ~np.isnan(trial_fitness)
        if np.any(valid):
            idx = np.nanargmin(trial_fitness)
            if trial_fitness[idx] < best_f:
                return trial_fitness[idx], trials[idx].copy()
        return best_f, best_x

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid) ** 2, axis=1))
        return np.mean(distances)
```