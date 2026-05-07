import numpy as np


class AdaptiveMultiStrategyDE:
    """
    Differential Evolution with adaptive multi-strategy mutation selection,
    success-history based parameter adaptation (SHADE-like), and
    opposition-based learning restarts.
    
    Unique features:
    - Four mutation strategies compete: DE/current-to-pbest/1, DE/rand-to-best/2,
      DE/midpoint-target/1 (novel), and DE/centroid-difference/1 (novel).
    - Strategy selection probabilities adapt based on success rates using
      a sliding window.
    - Rank-based opposition learning restart when population stagnates.
    - Cauchy/Normal hybrid parameter adaptation with memory.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.NP = min(max(6 * dim, 30), 300)
        self.p_best_ratio = 0.15
        self.memory_size = 8
        self.archive_max = self.NP
        self.num_strategies = 4
        self.stagnation_limit = 25
        self.min_pop_size = max(10, dim)

    def __call__(self, func, stopping_condition):
        pop, fitness = self._initialize_population(func)
        if stopping_condition():
            return self._best_result(pop, fitness)

        memory_F, memory_CR = self._initialize_memories()
        strategy_probs = np.ones(self.num_strategies) / self.num_strategies
        archive = np.empty((0, self.dim))
        success_history = [[] for _ in range(self.num_strategies)]
        mem_idx = 0
        stagnation_counter = 0
        best_ever_f = np.nanmin(fitness)
        best_ever_x = pop[np.nanargmin(fitness)].copy()

        while not stopping_condition():
            # Generate parameters and strategy assignments
            strategies = self._select_strategies(strategy_probs)
            F_vals, CR_vals = self._generate_parameters(memory_F, memory_CR)

            # Build trial vectors using batch mutation + crossover
            mutants = self._mutate_batch(pop, fitness, archive, strategies, F_vals)
            trials = self._crossover_batch(pop, mutants, CR_vals)
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                pop, fitness, best_ever_f, best_ever_x = self._update_best_tracking(
                    pop, fitness, trials, trial_fitness, best_ever_f, best_ever_x
                )
                break

            # Handle truncated evaluations
            valid = self._get_valid_mask(trial_fitness)
            if np.sum(valid) == 0:
                break

            # Selection and archive update
            pop, fitness, archive, succ_F, succ_CR, succ_strats = self._select_survivors_batch(
                pop, fitness, trials, trial_fitness, archive, F_vals, CR_vals, strategies, valid
            )

            # Track best ever
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < best_ever_f:
                best_ever_f = fitness[current_best_idx]
                best_ever_x = pop[current_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Update memories
            memory_F, memory_CR, mem_idx = self._update_parameter_memory(
                memory_F, memory_CR, mem_idx, succ_F, succ_CR
            )

            # Update strategy probabilities
            strategy_probs = self._update_strategy_probs(
                strategy_probs, succ_strats, success_history
            )

            # Restart if stagnant
            if stagnation_counter >= self._adaptive_stagnation_limit(fitness):
                pop, fitness, archive, stagnation_counter = self._restart_population(
                    func, pop, fitness, best_ever_x, stopping_condition
                )
                if stopping_condition():
                    break
                memory_F, memory_CR = self._initialize_memories()
                strategy_probs = np.ones(self.num_strategies) / self.num_strategies
                mem_idx = 0

                cur_best_idx = np.nanargmin(fitness)
                if fitness[cur_best_idx] < best_ever_f:
                    best_ever_f = fitness[cur_best_idx]
                    best_ever_x = pop[cur_best_idx].copy()

        return best_ever_f, best_ever_x

    def _initialize_population(self, func):
        """Create initial random population and evaluate."""
        pop = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        fitness = func(pop)
        return pop, fitness

    def _initialize_memories(self):
        """Initialize SHADE-style parameter memories."""
        memory_F = np.full(self.memory_size, 0.5)
        memory_CR = np.full(self.memory_size, 0.5)
        return memory_F, memory_CR

    def _select_strategies(self, strategy_probs):
        """Assign a mutation strategy to each individual based on adaptive probabilities."""
        probs = np.maximum(strategy_probs, 0.05)
        probs /= probs.sum()
        strategies = np.random.choice(self.num_strategies, size=self.NP, p=probs)
        return strategies

    def _generate_parameters(self, memory_F, memory_CR):
        """Generate F and CR values from SHADE memory using Cauchy/Normal distributions."""
        ri = np.random.randint(0, self.memory_size, size=self.NP)
        # F from Cauchy
        F_vals = self._cauchy_sample(memory_F[ri], 0.1, self.NP)
        F_vals = np.clip(F_vals, 0.05, 1.5)
        # Truncate F > 1
        F_vals = np.minimum(F_vals, 1.0)
        # CR from Normal
        CR_vals = np.random.normal(memory_CR[ri], 0.1)
        CR_vals = np.clip(CR_vals, 0.0, 1.0)
        return F_vals, CR_vals

    def _cauchy_sample(self, loc, scale, size):
        """Sample from Cauchy distribution, regenerating non-positive values."""
        samples = loc + scale * np.tan(np.pi * (np.random.uniform(size=size) - 0.5))
        # Regenerate non-positive
        mask = samples <= 0
        max_tries = 10
        for _ in range(max_tries):
            if not np.any(mask):
                break
            n_bad = np.sum(mask)
            samples[mask] = loc[mask] if np.ndim(loc) > 0 else loc
            samples[mask] += scale * np.tan(np.pi * (np.random.uniform(size=n_bad) - 0.5))
            mask = samples <= 0
        samples[samples <= 0] = 0.05
        return samples

    def _mutate_batch(self, pop, fitness, archive, strategies, F_vals):
        """Apply four different mutation strategies in a batched manner."""
        NP, dim = pop.shape
        mutants = np.empty_like(pop)
        F = F_vals[:, np.newaxis]  # (NP, 1)

        sorted_indices = np.argsort(fitness)
        p_best_count = max(2, int(self.p_best_ratio * NP))

        # Pre-compute common elements
        best_idx = sorted_indices[0]
        x_best = pop[best_idx]

        # Combined pool for archive-augmented selection
        if len(archive) > 0:
            combined = np.vstack([pop, archive])
        else:
            combined = pop

        for s in range(self.num_strategies):
            mask = strategies == s
            n_s = np.sum(mask)
            if n_s == 0:
                continue

            idx_s = np.where(mask)[0]
            targets = pop[idx_s]
            F_s = F[idx_s]

            if s == 0:
                # DE/current-to-pbest/1
                mutants[idx_s] = self._mutation_current_to_pbest(
                    pop, targets, F_s, sorted_indices, p_best_count, combined, idx_s, n_s
                )
            elif s == 1:
                # DE/rand-to-best/2
                mutants[idx_s] = self._mutation_rand_to_best_2(
                    pop, x_best, F_s, idx_s, n_s
                )
            elif s == 2:
                # DE/midpoint-target/1 (novel)
                mutants[idx_s] = self._mutation_midpoint_target(
                    pop, targets, F_s, sorted_indices, p_best_count, idx_s, n_s
                )
            elif s == 3:
                # DE/centroid-difference/1 (novel)
                mutants[idx_s] = self._mutation_centroid_difference(
                    pop, fitness, targets, F_s, idx_s, n_s
                )

        return mutants

    def _mutation_current_to_pbest(self, pop, targets, F_s, sorted_indices, p_best_count, combined, idx_s, n_s):
        """DE/current-to-pbest/1 with archive."""
        NP = len(pop)
        pbest_select = sorted_indices[np.random.randint(0, p_best_count, size=n_s)]
        x_pbest = pop[pbest_select]

        r1 = self._random_indices_excluding(NP, idx_s, n_s)
        r2 = self._random_indices_excluding(len(combined), idx_s, n_s, exclude2=r1)

        return targets + F_s * (x_pbest - targets) + F_s * (pop[r1] - combined[r2])

    def _mutation_rand_to_best_2(self, pop, x_best, F_s, idx_s, n_s):
        """DE/rand-to-best/2: base is random, guided toward best with two differences."""
        NP = len(pop)
        r0 = self._random_indices_excluding(NP, idx_s, n_s)
        r1 = self._random_indices_excluding(NP, idx_s, n_s, exclude2=r0)
        r2 = self._random_indices_excluding(NP, idx_s, n_s, exclude2=r1)
        r3 = self._random_indices_excluding(NP, idx_s, n_s, exclude2=r2)

        base = pop[r0]
        return base + F_s * (x_best - base) + F_s * (pop[r1] - pop[r2]) + 0.5 * F_s * (pop[r2] - pop[r3])

    def _mutation_midpoint_target(self, pop, targets, F_s, sorted_indices, p_best_count, idx_s, n_s):
        """Novel: midpoint between target and a pbest individual, plus a difference vector."""
        NP = len(pop)
        pbest_select = sorted_indices[np.random.randint(0, p_best_count, size=n_s)]
        x_pbest = pop[pbest_select]

        midpoint = 0.5 * (targets + x_pbest)
        r1 = self._random_indices_excluding(NP, idx_s, n_s)
        r2 = self._random_indices_excluding(NP, idx_s, n_s, exclude2=r1)

        return midpoint + F_s * (pop[r1] - pop[r2])

    def _mutation_centroid_difference(self, pop, fitness, targets, F_s, idx_s, n_s):
        """Novel: uses centroid of top-k versus centroid of bottom-k as direction."""
        NP = len(pop)
        k = max(3, NP // 5)
        sorted_idx = np.argsort(fitness)
        top_centroid = np.mean(pop[sorted_idx[:k]], axis=0)
        bottom_centroid = np.mean(pop[sorted_idx[-k:]], axis=0)
        direction = top_centroid - bottom_centroid  # (dim,)

        r1 = self._random_indices_excluding(NP, idx_s, n_s)
        noise = pop[r1] - targets

        return targets + F_s * direction[np.newaxis, :] + 0.5 * F_s * noise

    def _random_indices_excluding(self, pool_size, exclude1, n, exclude2=None):
        """Generate random indices avoiding self-index, with optional second exclusion."""
        indices = np.random.randint(0, pool_size, size=n)
        for _ in range(5):
            bad = indices == exclude1
            if exclude2 is not None:
                bad |= (indices == exclude2)
            if not np.any(bad):
                break
            indices[bad] = np.random.randint(0, pool_size, size=np.sum(bad))
        return indices

    def _crossover_batch(self, pop, mutants, CR_vals):
        """Binomial crossover applied to entire population at once."""
        NP, dim = pop.shape
        CR = CR_vals[:, np.newaxis]
        rand_matrix = np.random.uniform(size=(NP, dim))
        j_rand = np.random.randint(0, dim, size=NP)

        mask = rand_matrix < CR
        # Ensure at least one dimension from mutant
        mask[np.arange(NP), j_rand] = True

        trials = np.where(mask, mutants, pop)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds using bounce-back."""
        NP, dim = trials.shape
        # For out-of-bounds, use midpoint between bound and corresponding parent
        below = trials < self.lb
        above = trials > self.ub
        # Simple clip (fast and robust)
        trials = np.clip(trials, self.lb, self.ub)
        # Add small random perturbation to clipped values to avoid boundary accumulation
        perturb_below = np.random.uniform(0, 1, size=(NP, dim)) * below
        perturb_above = np.random.uniform(0, 1, size=(NP, dim)) * above
        trials += perturb_below - perturb_above
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _get_valid_mask(self, trial_fitness):
        """Return boolean mask of valid (non-NaN) fitness evaluations."""
        return ~np.isnan(trial_fitness)

    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness, archive, F_vals, CR_vals, strategies, valid):
        """Greedy selection: keep trial if it improves or equals the target."""
        improved = valid & (trial_fitness <= fitness)

        # Record successful parameters
        succ_F = F_vals[improved]
        succ_CR = CR_vals[improved]
        succ_strats = strategies[improved]

        # Update archive with replaced parents
        replaced_parents = pop[improved]
        if len(replaced_parents) > 0:
            archive = np.vstack([archive, replaced_parents]) if len(archive) > 0 else replaced_parents.copy()
            if len(archive) > self.archive_max:
                keep = np.random.choice(len(archive), self.archive_max, replace=False)
                archive = archive[keep]

        # Replace
        pop[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]

        return pop, fitness, archive, succ_F, succ_CR, succ_strats

    def _update_best_tracking(self, pop, fitness, trials, trial_fitness, best_ever_f, best_ever_x):
        """Update best tracking after a final evaluation batch."""
        valid = ~np.isnan(trial_fitness)
        improved = valid & (trial_fitness <= fitness)
        pop[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]

        valid_fitness = fitness[~np.isnan(fitness)]
        if len(valid_fitness) > 0:
            best_idx = np.nanargmin(fitness)
            if fitness[best_idx] < best_ever_f:
                best_ever_f = fitness[best_idx]
                best_ever_x = pop[best_idx].copy()

        return pop, fitness, best_ever_f, best_ever_x

    def _update_parameter_memory(self, memory_F, memory_CR, mem_idx, succ_F, succ_CR):
        """SHADE-style weighted Lehmer mean update of parameter memories."""
        if len(succ_F) == 0:
            return memory_F, memory_CR, mem_idx

        # Weighted Lehmer mean for F
        weights = np.ones(len(succ_F)) / len(succ_F)  # equal weights
        lehmer_F = np.sum(weights * succ_F ** 2) / (np.sum(weights * succ_F) + 1e-30)
        mean_CR = np.sum(weights * succ_CR) / (np.sum(weights) + 1e-30)

        memory_F[mem_idx] = lehmer_F
        memory_CR[mem_idx] = mean_CR
        mem_idx = (mem_idx + 1) % self.memory_size

        return memory_F, memory_CR, mem_idx

    def _update_strategy_probs(self, strategy_probs, succ_strats, success_history):
        """Update mutation strategy selection probabilities based on recent success."""
        if len(succ_strats) == 0:
            return strategy_probs

        # Update sliding window of successes
        for s in range(self.num_strategies):
            count = np.sum(succ_strats == s)
            success_history[s].append(count)
            # Keep only last 50 generations
            if len(success_history[s]) > 50:
                success_history[s] = success_history[s][-50:]

        # Compute new probabilities from cumulative success counts
        totals = np.array([sum(success_history[s]) for s in range(self.num_strategies)], dtype=float)
        total_sum = totals.sum()
        if total_sum > 0:
            new_probs = totals / total_sum
        else:
            new_probs = np.ones(self.num_strategies) / self.num_strategies

        # Exponential moving average with current probs for stability
        alpha = 0.3
        strategy_probs = alpha * new_probs + (1.0 - alpha) * strategy_probs
        strategy_probs = np.maximum(strategy_probs, 0.05)
        strategy_probs /= strategy_probs.sum()

        return strategy_probs

    def _adaptive_stagnation_limit(self, fitness):
        """Compute stagnation limit based on population diversity."""
        diversity = self._compute_diversity_fitness(fitness)
        if diversity < 1e-10:
            return max(5, self.stagnation_limit // 3)
        return self.stagnation_limit

    def _compute_diversity_fitness(self, fitness):
        """Compute fitness diversity as standard deviation of valid fitness values."""
        valid = fitness[~np.isnan(fitness)]
        if len(valid) < 2:
            return 0.0
        return np.std(valid)

    def _restart_population(self, func, pop, fitness, best_ever_x, stopping_condition):
        """Opposition-based learning restart: keep elite, regenerate rest."""
        NP, dim = pop.shape
        sorted_idx = np.argsort(fitness)

        # Keep top 20% elite
        n_elite = max(2, NP // 5)
        elite = pop[sorted_idx[:n_elite]].copy()
        elite_fit = fitness[sorted_idx[:n_elite]].copy()

        # Generate opposition-based points from elite
        n_opp = min(n_elite, NP - n_elite)
        center = np.mean(elite, axis=0)
        opp_points = 2.0 * center - elite[:n_opp]
        opp_points = self._clip_to_bounds(opp_points)

        # Fill remaining with random + perturbation around best
        n_random = NP - n_elite - n_opp
        if n_random > 0:
            n_near = n_random // 2
            n_far = n_random - n_near
            # Near best
            near_best = best_ever_x + np.random.normal(0, 10, size=(n_near, dim))
            near_best = self._clip_to_bounds(near_best)
            # Far random
            far_random = np.random.uniform(self.lb, self.ub, (n_far, dim))
            new_random = np.vstack([near_best, far_random]) if n_near > 0 else far_random
        else:
            new_random = np.empty((0, dim))

        # Assemble new population
        new_pop_parts = [elite, opp_points]
        if len(new_random) > 0:
            new_pop_parts.append(new_random)
        new_pop = np.vstack(new_pop_parts)[:NP]
        new_pop = self._clip_to_bounds(new_pop)

        # Evaluate non-elite parts
        n_to_eval = NP - n_elite
        if n_to_eval > 0 and not stopping_condition():
            eval_fitness = func(new_pop[n_elite:])
            if stopping_condition():
                valid_count = np.sum(~np.isnan(eval_fitness))
                new_fitness = np.full(NP, np.inf)
                new_fitness[:n_elite] = elite_fit
                new_fitness[n_elite:n_elite + len(eval_fitness)] = eval_fitness
                return new_pop, new_fitness, np.empty((0, dim)), 0

            new_fitness = np.concatenate([elite_fit, eval_fitness])
        else:
            new_fitness = np.full(NP, np.inf)
            new_fitness[:n_elite] = elite_fit

        archive = np.empty((0, dim))
        return new_pop, new_fitness, archive, 0

    def _best_result(self, pop, fitness):
        """Return the best solution found so far."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return np.inf, pop[0].copy()
        best_idx = np.nanargmin(fitness)
        return fitness[best_idx], pop[best_idx].copy()
