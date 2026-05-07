import numpy as np


class HybridDESimplex:
    """
    A hybrid optimizer combining Differential Evolution with simplex-based
    local refinement. The DE component uses adaptive F and CR with
    current-to-pbest mutation. Periodically, a simplex contraction step
    (inspired by Nelder-Mead) is applied to the best subset of the population
    to accelerate local convergence.
    
    Key differences from prior proposals:
    - Combines global DE search with batched simplex moves (reflection, contraction)
    - Uses archive of successful parameters for adaptation (SHADE-like memory)
    - Simplex refinement operates on elite subpopulation in batch mode
    - Opposition-based reinitialization for stagnation escape
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np = min(max(6 * dim, 40), 300)
        self.elite_size = max(dim + 1, 10)
        self.memory_size = 10
        self.p_best_rate = 0.15
        self.simplex_interval = 5
        self.stagnation_limit = 15

    def __call__(self, func, stopping_condition):
        population, fitness = self._initialize_population(func)
        if stopping_condition():
            return self._get_best(population, fitness)

        f_memory, cr_memory = self._initialize_memory()
        memory_idx = 0
        gen_without_improvement = 0
        best_fitness_ever = np.nanmin(fitness)

        gen = 0
        while not stopping_condition():
            # DE mutation and crossover
            f_vals, cr_vals = self._sample_parameters(f_memory, cr_memory)
            mutants = self._mutate_batch(population, fitness, f_vals)
            trials = self._crossover_batch(population, mutants, cr_vals)
            trials = self._clip_bounds(trials)

            trial_fitness = func(trials)
            if stopping_condition():
                population, fitness = self._merge_partial(
                    population, fitness, trials, trial_fitness
                )
                break

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]

            # Selection and parameter memory update
            population, fitness, success_f, success_cr = self._select_survivors_batch(
                population, fitness, trials, trial_fitness, f_vals, cr_vals
            )

            memory_idx = self._update_memory(
                f_memory, cr_memory, memory_idx, success_f, success_cr
            )

            # Track stagnation
            current_best = np.nanmin(fitness)
            if current_best < best_fitness_ever - 1e-12:
                best_fitness_ever = current_best
                gen_without_improvement = 0
            else:
                gen_without_improvement += 1

            # Simplex-based local refinement on elites
            if gen % self.simplex_interval == 0 and not stopping_condition():
                population, fitness = self._simplex_refinement_batch(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break

            # Opposition-based restart if stagnant
            if gen_without_improvement >= self.stagnation_limit:
                population, fitness = self._opposition_restart(
                    population, fitness, func, stopping_condition
                )
                gen_without_improvement = 0
                if stopping_condition():
                    break
                best_fitness_ever = np.nanmin(fitness)

            gen += 1

        return self._get_best(population, fitness)

    def _initialize_population(self, func):
        population = np.random.uniform(
            self.lb, self.ub, size=(self.np, self.dim)
        )
        fitness = func(population)
        return population, fitness

    def _initialize_memory(self):
        f_memory = np.full(self.memory_size, 0.5)
        cr_memory = np.full(self.memory_size, 0.5)
        return f_memory, cr_memory

    def _sample_parameters(self, f_memory, cr_memory):
        indices = np.random.randint(0, self.memory_size, size=self.np)
        mu_f = f_memory[indices]
        mu_cr = cr_memory[indices]

        f_vals = self._cauchy_sample(mu_f, 0.1)
        f_vals = np.clip(f_vals, 0.05, 1.0)

        cr_vals = np.random.normal(mu_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0.0, 1.0)
        return f_vals, cr_vals

    def _cauchy_sample(self, loc, scale):
        """Batch Cauchy sampling with regeneration of non-positive values."""
        samples = loc + scale * np.tan(np.pi * (np.random.random(len(loc)) - 0.5))
        # Regenerate non-positive
        mask = samples <= 0
        max_tries = 10
        for _ in range(max_tries):
            if not np.any(mask):
                break
            n_bad = np.sum(mask)
            samples[mask] = loc[mask] + scale * np.tan(
                np.pi * (np.random.random(n_bad) - 0.5)
            )
            mask = samples <= 0
        samples[samples <= 0] = 0.05
        return samples

    def _mutate_batch(self, population, fitness, f_vals):
        """Current-to-pbest/1 mutation applied in batch."""
        n = len(population)
        p_count = max(2, int(np.round(self.p_best_rate * n)))
        sorted_indices = np.argsort(fitness)
        pbest_indices = sorted_indices[:p_count]

        # Select random pbest for each individual
        chosen_pbest = pbest_indices[np.random.randint(0, p_count, size=n)]

        # Two distinct random indices for each individual
        r1 = np.random.randint(0, n, size=n)
        r2 = np.random.randint(0, n, size=n)
        for i in range(n):
            while r1[i] == i:
                r1[i] = np.random.randint(0, n)
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, n)

        f_col = f_vals[:, np.newaxis]
        mutants = (
            population
            + f_col * (population[chosen_pbest] - population)
            + f_col * (population[r1] - population[r2])
        )
        return mutants

    def _crossover_batch(self, population, mutants, cr_vals):
        """Binomial crossover applied in batch."""
        n, d = population.shape
        cr_matrix = cr_vals[:, np.newaxis] * np.ones((1, d))
        rand_matrix = np.random.random((n, d))
        j_rand = np.random.randint(0, d, size=n)

        mask = rand_matrix < cr_matrix
        # Ensure at least one dimension from mutant
        rows = np.arange(n)
        mask[rows, j_rand] = True

        trials = np.where(mask, mutants, population)
        return trials

    def _clip_bounds(self, x):
        """Clip solutions to search bounds."""
        return np.clip(x, self.lb, self.ub)

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, f_vals, cr_vals):
        """Greedy selection: keep trial if at least as good."""
        valid_len = min(len(fitness), len(trial_fitness))
        improved = np.zeros(len(population), dtype=bool)
        success_f = []
        success_cr = []

        for i in range(valid_len):
            tf = trial_fitness[i]
            if np.isnan(tf):
                continue
            if tf <= fitness[i]:
                if tf < fitness[i]:
                    improved[i] = True
                    success_f.append(f_vals[i])
                    success_cr.append(cr_vals[i])
                population[i] = trials[i]
                fitness[i] = tf

        return population, fitness, np.array(success_f), np.array(success_cr)

    def _update_memory(self, f_memory, cr_memory, memory_idx, success_f, success_cr):
        """Update SHADE-style parameter memory with Lehmer mean."""
        if len(success_f) == 0:
            return memory_idx

        # Lehmer mean for F
        f_lehmer = np.sum(success_f ** 2) / (np.sum(success_f) + 1e-30)
        # Weighted mean for CR
        cr_mean = np.mean(success_cr)

        f_memory[memory_idx] = f_lehmer
        cr_memory[memory_idx] = cr_mean
        memory_idx = (memory_idx + 1) % len(f_memory)
        return memory_idx

    def _simplex_refinement_batch(self, population, fitness, func, stopping_condition):
        """
        Batched simplex-inspired refinement on the elite subset.
        Performs reflection and contraction moves simultaneously for all
        elite members relative to the centroid.
        """
        sorted_idx = np.argsort(fitness)
        elite_count = min(self.elite_size, len(population))
        elite_idx = sorted_idx[:elite_count]
        worst_idx = sorted_idx[-1]

        elite = population[elite_idx]
        elite_fit = fitness[elite_idx]

        # Compute centroid of the best half of elites
        half = max(2, elite_count // 2)
        centroid = np.mean(elite[:half], axis=0)

        # Reflection of worst elite members through centroid
        reflection_candidates = self._reflect_batch(elite, centroid, alpha=1.0)
        reflection_candidates = self._clip_bounds(reflection_candidates)

        if stopping_condition():
            return population, fitness

        ref_fitness = func(reflection_candidates)
        if stopping_condition():
            return self._apply_simplex_results(
                population, fitness, elite_idx, reflection_candidates, ref_fitness
            )

        if len(ref_fitness) < len(reflection_candidates):
            reflection_candidates = reflection_candidates[:len(ref_fitness)]
            elite_idx = elite_idx[:len(ref_fitness)]
            elite_fit = elite_fit[:len(ref_fitness)]

        # Contraction towards centroid for those where reflection didn't help
        improved_mask = ref_fitness < elite_fit
        need_contraction = ~improved_mask & ~np.isnan(ref_fitness)

        if np.any(need_contraction):
            contract_idx = np.where(need_contraction)[0]
            contracted = self._contract_batch(
                elite[contract_idx] if len(contract_idx) <= len(elite) else elite,
                centroid, beta=0.5
            )
            contracted = self._clip_bounds(contracted)

            if not stopping_condition():
                con_fitness = func(contracted)
                if len(con_fitness) >= len(contract_idx):
                    for j, ci in enumerate(contract_idx):
                        if ci < len(elite_idx) and not np.isnan(con_fitness[j]):
                            if con_fitness[j] < elite_fit[ci]:
                                population[elite_idx[ci]] = contracted[j]
                                fitness[elite_idx[ci]] = con_fitness[j]

        # Apply reflection improvements
        for i in range(len(elite_idx)):
            if i < len(ref_fitness) and improved_mask[i] and not np.isnan(ref_fitness[i]):
                population[elite_idx[i]] = reflection_candidates[i]
                fitness[elite_idx[i]] = ref_fitness[i]

        return population, fitness

    def _reflect_batch(self, points, centroid, alpha):
        """Reflect points through centroid: centroid + alpha * (centroid - point)."""
        return centroid[np.newaxis, :] + alpha * (centroid[np.newaxis, :] - points)

    def _contract_batch(self, points, centroid, beta):
        """Contract points towards centroid: centroid + beta * (point - centroid)."""
        return centroid[np.newaxis, :] + beta * (points - centroid[np.newaxis, :])

    def _apply_simplex_results(self, population, fitness, elite_idx, candidates, cand_fitness):
        """Apply partial simplex results when stopping early."""
        valid = min(len(elite_idx), len(cand_fitness))
        for i in range(valid):
            if not np.isnan(cand_fitness[i]) and cand_fitness[i] < fitness[elite_idx[i]]:
                population[elite_idx[i]] = candidates[i]
                fitness[elite_idx[i]] = cand_fitness[i]
        return population, fitness

    def _opposition_restart(self, population, fitness, func, stopping_condition):
        """
        Restart half the population using opposition-based learning.
        Keep the best half, replace worst half with opposition points.
        """
        n = len(population)
        sorted_idx = np.argsort(fitness)
        keep_count = n // 2
        replace_count = n - keep_count

        # Opposition of the worst individuals through bounds midpoint
        worst_idx = sorted_idx[keep_count:]
        worst_pop = population[worst_idx]
        opposition = (self.lb + self.ub) - worst_pop

        # Add some randomness to opposition
        noise = np.random.uniform(-5.0, 5.0, size=opposition.shape)
        opposition = opposition + noise
        opposition = self._clip_bounds(opposition)

        if stopping_condition():
            return population, fitness

        opp_fitness = func(opposition)
        if stopping_condition():
            valid = min(len(opp_fitness), replace_count)
            for i in range(valid):
                if not np.isnan(opp_fitness[i]) and opp_fitness[i] < fitness[worst_idx[i]]:
                    population[worst_idx[i]] = opposition[i]
                    fitness[worst_idx[i]] = opp_fitness[i]
            return population, fitness

        valid = min(len(opp_fitness), replace_count)
        for i in range(valid):
            if not np.isnan(opp_fitness[i]):
                # Greedy: keep better of original and opposition
                if opp_fitness[i] < fitness[worst_idx[i]]:
                    population[worst_idx[i]] = opposition[i]
                    fitness[worst_idx[i]] = opp_fitness[i]

        return population, fitness

    def _merge_partial(self, population, fitness, trials, trial_fitness):
        """Merge partial evaluation results into population."""
        valid = min(len(fitness), len(trial_fitness))
        for i in range(valid):
            if not np.isnan(trial_fitness[i]) and trial_fitness[i] < fitness[i]:
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]
        return population, fitness

    def _get_best(self, population, fitness):
        """Return the best fitness and corresponding solution."""
        valid_mask = ~np.isnan(fitness)
        if not np.any(valid_mask):
            return float('inf'), population[0]
        best_idx = np.nanargmin(fitness)
        return float(fitness[best_idx]), population[best_idx].copy()
