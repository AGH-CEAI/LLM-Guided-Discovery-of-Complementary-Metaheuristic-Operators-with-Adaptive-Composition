

```python
import numpy as np


class AdaptiveMultiStrategyDE:
    """
    Adaptive Multi-Strategy Differential Evolution with Opposition-Based Learning,
    Rank-Based Mutation, and Cauchy-Gaussian Hybrid Perturbation.
    
    Key novelties:
    - Three mutation strategies compete: rank-weighted/best-2/current-to-pbest
    - Strategy selection probabilities adapt based on success history
    - F and CR parameters drawn from Cauchy and Gaussian distributions respectively,
      with adaptive location parameters (SHADE-like memory)
    - Opposition-based population reinitialization when diversity collapses
    - Fitness-rank-weighted donor selection (not uniform random)
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 50), 300)
        self.memory_size = 10
        self.p_best_ratio = 0.15
        self.archive_max = self.np_size
        self.num_strategies = 3
        self.min_pop = max(4, dim)

    def __call__(self, func, stopping_condition):
        pop, fitness = self._initialize_population(func)
        if stopping_condition():
            return self._best_result(pop, fitness)

        archive = np.empty((0, self.dim))
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        strategy_probs = np.ones(self.num_strategies) / self.num_strategies
        strategy_success = np.zeros(self.num_strategies)
        strategy_total = np.zeros(self.num_strategies)
        generation = 0
        stagnation_counter = 0
        prev_best = np.min(fitness)

        while not stopping_condition():
            # Generate parameters for each individual
            f_vals, cr_vals = self._generate_parameters_batch(memory_f, memory_cr)

            # Assign strategies to individuals
            strategies = self._assign_strategies_batch(strategy_probs)

            # Build trial vectors using assigned strategies
            trials = self._build_trials_batch(
                pop, fitness, archive, f_vals, cr_vals, strategies
            )

            # Clip to bounds
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                pop, fitness = self._safe_merge(pop, fitness, trials, trial_fitness)
                break

            # Handle truncated results
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                f_vals = f_vals[:len(trial_fitness)]
                cr_vals = cr_vals[:len(trial_fitness)]
                strategies = strategies[:len(trial_fitness)]

            # Selection and archive update
            pop, fitness, archive, success_f, success_cr, succ_strats, fail_strats = (
                self._select_and_archive(
                    pop, fitness, trials, trial_fitness, archive, f_vals, cr_vals, strategies
                )
            )

            # Update parameter memories
            memory_f, memory_cr, memory_idx = self._update_parameter_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr
            )

            # Update strategy probabilities
            strategy_probs, strategy_success, strategy_total = (
                self._update_strategy_probs(
                    strategy_probs, strategy_success, strategy_total,
                    succ_strats, fail_strats
                )
            )

            # Check stagnation and apply diversity injection
            current_best = np.min(fitness)
            stagnation_counter, prev_best = self._check_stagnation(
                current_best, prev_best, stagnation_counter
            )

            if stagnation_counter > 25:
                pop, fitness, stagnation_counter = self._opposition_based_restart(
                    pop, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break

            generation += 1

        return self._best_result(pop, fitness)

    def _initialize_population(self, func):
        """Create initial population uniformly in bounds and evaluate."""
        pop = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(pop)
        # Handle potential truncation
        if len(fitness) < len(pop):
            pop = pop[:len(fitness)]
        return pop, fitness

    def _best_result(self, pop, fitness):
        """Extract the best solution found, handling NaN values."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return np.inf, pop[0]
        idx = np.nanargmin(fitness)
        return float(fitness[idx]), pop[idx].copy()

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values for all individuals using memory-based distributions."""
        n = self.np_size
        # Pick random memory indices for each individual
        ri = np.random.randint(0, len(memory_f), size=n)

        # F from Cauchy distribution centered at memory values
        f_vals = self._cauchy_sample(memory_f[ri], 0.1, n)
        f_vals = np.clip(f_vals, 0.05, 1.5)
        # Truncate above 1
        f_vals = np.minimum(f_vals, 1.0)

        # CR from Gaussian distribution centered at memory values
        cr_vals = np.random.normal(memory_cr[ri], 0.1)
        cr_vals = np.clip(cr_vals, 0.0, 1.0)

        return f_vals, cr_vals

    def _cauchy_sample(self, loc, scale, n):
        """Sample from Cauchy distribution for each element using its own location."""
        u = np.random.uniform(size=n)
        samples = loc + scale * np.tan(np.pi * (u - 0.5))
        # Resample negative values
        neg_mask = samples <= 0
        max_retries = 10
        for _ in range(max_retries):
            if not np.any(neg_mask):
                break
            count = np.sum(neg_mask)
            u_new = np.random.uniform(size=count)
            new_samples = loc[neg_mask] + scale * np.tan(np.pi * (u_new - 0.5))
            samples[neg_mask] = new_samples
            neg_mask = samples <= 0
        samples[samples <= 0] = 0.05
        return samples

    def _assign_strategies_batch(self, strategy_probs):
        """Assign a mutation strategy to each individual based on current probabilities."""
        return np.random.choice(
            self.num_strategies, size=self.np_size, p=strategy_probs
        )

    def _build_trials_batch(self, pop, fitness, archive, f_vals, cr_vals, strategies):
        """Construct trial vectors for all individuals based on their assigned strategies."""
        n, d = pop.shape
        donors = np.empty_like(pop)

        mask0 = strategies == 0
        mask1 = strategies == 1
        mask2 = strategies == 2

        if np.any(mask0):
            donors[mask0] = self._mutation_rank_weighted(
                pop, fitness, f_vals, np.where(mask0)[0]
            )
        if np.any(mask1):
            donors[mask1] = self._mutation_best_2(
                pop, fitness, f_vals, np.where(mask1)[0]
            )
        if np.any(mask2):
            donors[mask2] = self._mutation_current_to_pbest(
                pop, fitness, archive, f_vals, np.where(mask2)[0]
            )

        trials = self._binomial_crossover_batch(pop, donors, cr_vals)
        return trials

    def _mutation_rank_weighted(self, pop, fitness, f_vals, indices):
        """Rank-weighted mutation: donor vectors selected with probability proportional to fitness rank."""
        n, d = pop.shape
        k = len(indices)
        # Compute rank weights (best gets highest weight)
        ranks = np.argsort(np.argsort(fitness))  # rank 0 = best
        weights = (n - ranks).astype(float)
        weights /= weights.sum()

        donors = np.empty((k, d))
        for batch_start in range(0, k, k):  # single batch
            # Select r1 weighted by fitness rank
            r1 = self._weighted_random_choice(weights, k)
            r2 = np.random.randint(0, n, size=k)
            # Ensure r1 != r2 != target
            for i in range(k):
                target = indices[i]
                while r1[i] == target:
                    r1[i] = self._weighted_random_choice(weights, 1)[0]
                while r2[i] == target or r2[i] == r1[i]:
                    r2[i] = np.random.randint(0, n)

            f_k = f_vals[indices].reshape(-1, 1)
            donors = pop[r1] + f_k * (pop[r1] - pop[r2])
        return donors

    def _weighted_random_choice(self, weights, size):
        """Vectorized weighted random selection returning indices."""
        return np.random.choice(len(weights), size=size, p=weights)

    def _mutation_best_2(self, pop, fitness, f_vals, indices):
        """DE/best/2 mutation strategy."""
        n, d = pop.shape
        k = len(indices)
        best_idx = np.argmin(fitness)

        r = np.random.randint(0, n, size=(k, 4))
        # Ensure all random indices are distinct and not equal to target or best
        for i in range(k):
            used = {indices[i], best_idx}
            for j in range(4):
                while r[i, j] in used:
                    r[i, j] = np.random.randint(0, n)
                used.add(r[i, j])

        f_k = f_vals[indices].reshape(-1, 1)
        donors = (pop[best_idx] +
                  f_k * (pop[r[:, 0]] - pop[r[:, 1]]) +
                  f_k * (pop[r[:, 2]] - pop[r[:, 3]]))
        return donors

    def _mutation_current_to_pbest(self, pop, fitness, archive, f_vals, indices):
        """DE/current-to-pbest/1 with archive."""
        n, d = pop.shape
        k = len(indices)

        # Select p-best individuals
        p = max(2, int(self.p_best_ratio * n))
        sorted_idx = np.argsort(fitness)
        pbest_pool = sorted_idx[:p]
        pbest_chosen = pbest_pool[np.random.randint(0, p, size=k)]

        # Select r1 from population
        r1 = np.random.randint(0, n, size=k)
        for i in range(k):
            while r1[i] == indices[i]:
                r1[i] = np.random.randint(0, n)

        # Select r2 from population + archive
        combined = pop
        if len(archive) > 0:
            combined = np.vstack([pop, archive])
        nc = len(combined)
        r2 = np.random.randint(0, nc, size=k)
        for i in range(k):
            while r2[i] == indices[i] or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, nc)

        f_k = f_vals[indices].reshape(-1, 1)
        donors = (pop[indices] +
                  f_k * (pop[pbest_chosen] - pop[indices]) +
                  f_k * (pop[r1] - combined[r2]))
        return donors

    def _binomial_crossover_batch(self, pop, donors, cr_vals):
        """Binomial crossover applied to entire population at once."""
        n, d = pop.shape
        cr_matrix = cr_vals.reshape(-1, 1) * np.ones((1, d))
        rand_matrix = np.random.uniform(size=(n, d))
        # Ensure at least one dimension from donor
        j_rand = np.random.randint(0, d, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, donors, pop)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search space bounds using bounce-back."""
        # Simple clipping approach with midpoint reflection for out-of-bounds
        below = trials < self.lb
        above = trials > self.ub
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_and_archive(self, pop, fitness, trials, trial_fitness,
                            archive, f_vals, cr_vals, strategies):
        """Greedy selection: keep better of parent/trial. Track successful parameters."""
        n = min(len(pop), len(trials), len(trial_fitness))
        pop = pop[:n]
        fitness = fitness[:n]
        trials = trials[:n]
        trial_fitness = trial_fitness[:n]

        # Identify improvements (handle NaN in trial_fitness)
        valid_trial = ~np.isnan(trial_fitness)
        improved = valid_trial & (trial_fitness <= fitness)

        # Record successful parameters
        success_f = f_vals[:n][improved]
        success_cr = cr_vals[:n][improved]
        succ_strats = strategies[:n][improved]
        fail_strats = strategies[:n][~improved]

        # Add replaced parents to archive
        if np.any(improved):
            new_archive_members = pop[improved].copy()
            archive = np.vstack([archive, new_archive_members]) if len(archive) > 0 else new_archive_members.copy()
            # Trim archive
            if len(archive) > self.archive_max:
                keep = np.random.choice(len(archive), self.archive_max, replace=False)
                archive = archive[keep]

        # Replace
        pop[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]

        return pop, fitness, archive, success_f, success_cr, succ_strats, fail_strats

    def _update_parameter_memory(self, memory_f, memory_cr, memory_idx,
                                 success_f, success_cr):
        """Update SHADE-style parameter memory with Lehmer mean for F and arithmetic mean for CR."""
        if len(success_f) == 0:
            return memory_f, memory_cr, memory_idx

        # Lehmer mean for F
        sum_f2 = np.sum(success_f ** 2)
        sum_f = np.sum(success_f)
        if sum_f > 0:
            lehmer_f = sum_f2 / sum_f
        else:
            lehmer_f = 0.5
        memory_f[memory_idx] = lehmer_f

        # Weighted arithmetic mean for CR
        mean_cr = np.mean(success_cr) if len(success_cr) > 0 else 0.5
        memory_cr[memory_idx] = mean_cr

        memory_idx = (memory_idx + 1) % len(memory_f)
        return memory_f, memory_cr, memory_idx

    def _update_strategy_probs(self, strategy_probs, strategy_success,
                               strategy_total, succ_strats, fail_strats):
        """Update strategy selection probabilities based on recent success rates."""
        decay = 0.9
        strategy_success *= decay
        strategy_total *= decay

        for s in range(self.num_strategies):
            s_count = np.sum(succ_strats == s)
            f_count = np.sum(fail_strats == s)
            strategy_success[s] += s_count
            strategy_total[s] += s_count + f_count

        # Compute new probabilities with minimum threshold
        min_prob = 0.05
        total_successes = strategy_success.sum()
        if total_successes > 0:
            raw_probs = strategy_success / (strategy_total + 1e-12)
            raw_probs = np.maximum(raw_probs, min_prob)
            strategy_probs = raw_probs / raw_probs.sum()
        else:
            strategy_probs = np.ones(self.num_strategies) / self.num_strategies

        return strategy_probs, strategy_success, strategy_total

    def _check_stagnation(self, current_best, prev_best, stagnation_counter):
        """Detect if the population has stagnated (no improvement in best fitness)."""
        if current_best < prev_best - 1e-12:
            stagnation_counter = 0
            prev_best = current_best
        else:
            stagnation_counter += 1
        return stagnation_counter, prev_best

    def _opposition_based_restart(self, pop, fitness, func, stopping_condition):
        """Reinitialize half the population using opposition-based learning."""
        n, d = pop.shape
        best_idx = np.argmin(fitness)

        # Keep the best half, regenerate the worst half using opposition
        sorted_idx = np.argsort(fitness)
        keep = max(4, n // 2)
        replace_idx = sorted_idx[keep:]

        # Opposition of worst individuals relative to bounds midpoint
        opp = self.lb + self.ub - pop[replace_idx]
        # Add some randomization
        noise = np.random.uniform(-10, 10, size=opp.shape)
        opp = opp + noise
        opp = np.clip(opp, self.lb, self.ub)

        if stopping_condition():
            return pop, fitness, 0

        opp_fitness = func(opp)
        if len(opp_fitness) < len(opp):
            opp = opp[:len(opp_fitness)]
            replace_idx = replace_idx[:len(opp_fitness)]

        valid = ~np.isnan(opp_fitness)
        pop[replace_idx[valid]] = opp[valid]
        fitness[replace_idx[valid]] = opp_fitness[valid]

        return pop, fitness, 0

    def _safe_merge(self, pop, fitness, trials, trial_fitness):
        """Safely merge trial results into population at the end of the run."""
        n = min(len(pop), len(trials), len(trial_fitness))
        if n == 0:
            return pop, fitness
        valid = ~np.isnan(trial_fitness[:n])
        improved = valid & (trial_fitness[:n] <= fitness[:n])
        pop[:n][improved] = trials[:n][improved]
        fitness[:n][improved] = trial_fitness[:n][improved]
        return pop, fitness

    def _compute_diversity(self, pop):
        """Compute population diversity as mean pairwise distance to centroid."""
        centroid = np.mean(pop, axis=0)
        dists = np.sqrt(np.sum((pop - centroid) ** 2, axis=1))
        return np.mean(dists)
```