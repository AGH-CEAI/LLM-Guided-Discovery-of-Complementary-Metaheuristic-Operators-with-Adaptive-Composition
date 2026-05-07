import numpy as np


class AdaptiveVortexDE:
    """
    Adaptive Vortex Differential Evolution (AVDE)
    
    A DE variant featuring:
    - Vortex mutation: individuals spiral around the best solution with adaptive radius
    - Rank-based archive for mutation donors with memory of past elite solutions
    - Dual-pool parameter adaptation (successful F/CR tracked separately for
      exploitation vs exploration mutations)
    - Opposition-based population restart on stagnation detection
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 30), 600)
        self.archive_size = self.np_size
        self.F_pool_explore = []
        self.F_pool_exploit = []
        self.CR_pool_explore = []
        self.CR_pool_exploit = []
        self.F_mean_explore = 0.7
        self.F_mean_exploit = 0.5
        self.CR_mean_explore = 0.5
        self.CR_mean_exploit = 0.9
        self.stagnation_counter = 0
        self.stagnation_limit = max(30, dim)
        self.vortex_angle = 0.0
        self.vortex_speed = 0.1
        self.generation = 0

    def __call__(self, func, stopping_condition):
        population, fitness = self._initialize_population(func, stopping_condition)
        if stopping_condition():
            return self._get_best(population, fitness)

        archive = self._initialize_archive(population, fitness)
        best_f, best_x = self._get_best(population, fitness)
        prev_best_f = best_f

        while not stopping_condition():
            self.generation += 1
            
            # Decide mutation strategy per individual
            strategy_mask = self._assign_strategies(fitness)
            
            # Generate adaptive parameters
            F_vals, CR_vals = self._generate_parameters(strategy_mask)
            
            # Create mutant vectors using vortex-enhanced mutation
            mutants = self._mutate_batch(population, fitness, archive, F_vals, strategy_mask, best_x)
            
            # Crossover
            trials = self._crossover_batch(population, mutants, CR_vals)
            
            # Clip to bounds
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trials
            trial_fitness = func(trials)
            if stopping_condition():
                valid = self._get_valid_mask(trial_fitness)
                if np.any(valid):
                    population, fitness = self._partial_select(
                        population, fitness, trials, trial_fitness, valid, 
                        F_vals, CR_vals, strategy_mask
                    )
                    cb_f, cb_x = self._get_best(population, fitness)
                    if cb_f < best_f:
                        best_f, best_x = cb_f, cb_x.copy()
                break
            
            valid = self._get_valid_mask(trial_fitness)
            if not np.all(valid):
                population, fitness = self._partial_select(
                    population, fitness, trials, trial_fitness, valid,
                    F_vals, CR_vals, strategy_mask
                )
                cb_f, cb_x = self._get_best(population, fitness)
                if cb_f < best_f:
                    best_f, best_x = cb_f, cb_x.copy()
                break
            
            # Selection and archive update
            success_mask = trial_fitness < fitness
            archive = self._update_archive(archive, population, success_mask)
            self._update_parameter_memory(F_vals, CR_vals, strategy_mask, 
                                          success_mask, fitness, trial_fitness)
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            cb_f, cb_x = self._get_best(population, fitness)
            if cb_f < best_f:
                best_f, best_x = cb_f, cb_x.copy()
            
            # Stagnation detection and restart
            if self._detect_stagnation(best_f, prev_best_f):
                population, fitness = self._restart_if_stagnant(
                    population, fitness, best_x, func, stopping_condition
                )
                if stopping_condition():
                    cb_f, cb_x = self._get_best(population, fitness)
                    if cb_f < best_f:
                        best_f, best_x = cb_f, cb_x.copy()
                    break
                cb_f, cb_x = self._get_best(population, fitness)
                if cb_f < best_f:
                    best_f, best_x = cb_f, cb_x.copy()
            
            prev_best_f = best_f
            self._advance_vortex()

        return best_f, best_x

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        valid = self._get_valid_mask(fitness)
        if not np.all(valid):
            fitness = np.where(valid, fitness, np.inf)
        return population, fitness

    def _initialize_archive(self, population, fitness):
        """Initialize archive with copies of the initial population."""
        sorted_idx = np.argsort(fitness)
        archive_idx = sorted_idx[:self.archive_size]
        archive = population[archive_idx].copy()
        return archive

    def _get_best(self, population, fitness):
        """Return the best fitness and corresponding solution."""
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return np.inf, population[0].copy()
        idx = np.nanargmin(fitness)
        return fitness[idx], population[idx].copy()

    def _get_valid_mask(self, fitness):
        """Return boolean mask of valid (non-NaN) fitness values."""
        return np.isfinite(fitness)

    def _assign_strategies(self, fitness):
        """Assign exploration (0) or exploitation (1) strategy per individual.
        
        Top-ranked individuals get exploitation, bottom get exploration.
        Uses a smooth probability transition based on rank.
        """
        ranks = np.argsort(np.argsort(fitness)).astype(float)
        # Probability of exploitation decreases with rank (worse individuals explore more)
        prob_exploit = 1.0 - ranks / (len(fitness) - 1 + 1e-30)
        # Add generation-based bias: early = explore, late = exploit
        gen_factor = min(self.generation / (10 * self.dim + 1), 1.0)
        prob_exploit = prob_exploit * 0.5 + gen_factor * 0.5
        strategy_mask = (np.random.rand(len(fitness)) < prob_exploit).astype(int)
        return strategy_mask

    def _generate_parameters(self, strategy_mask):
        """Generate F and CR values per individual using Cauchy/Normal distributions."""
        n = len(strategy_mask)
        F_vals = np.empty(n)
        CR_vals = np.empty(n)
        
        # Exploration individuals
        explore_mask = strategy_mask == 0
        n_explore = np.sum(explore_mask)
        if n_explore > 0:
            F_vals[explore_mask] = self._cauchy_sample(self.F_mean_explore, 0.1, n_explore)
            CR_vals[explore_mask] = np.clip(
                np.random.normal(self.CR_mean_explore, 0.1, n_explore), 0, 1
            )
        
        # Exploitation individuals
        exploit_mask = strategy_mask == 1
        n_exploit = np.sum(exploit_mask)
        if n_exploit > 0:
            F_vals[exploit_mask] = self._cauchy_sample(self.F_mean_exploit, 0.1, n_exploit)
            CR_vals[exploit_mask] = np.clip(
                np.random.normal(self.CR_mean_exploit, 0.1, n_exploit), 0, 1
            )
        
        F_vals = np.clip(F_vals, 0.05, 1.5)
        return F_vals, CR_vals

    def _cauchy_sample(self, loc, scale, size):
        """Sample from Cauchy distribution, regenerating values <= 0."""
        samples = loc + scale * np.tan(np.pi * (np.random.rand(size) - 0.5))
        bad = samples <= 0
        while np.any(bad):
            n_bad = np.sum(bad)
            samples[bad] = loc + scale * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
            bad = samples <= 0
        return np.minimum(samples, 2.0)

    def _mutate_batch(self, population, fitness, archive, F_vals, strategy_mask, best_x):
        """Create mutant vectors using vortex-enhanced DE mutations.
        
        Exploration: DE/rand/1 with vortex perturbation
        Exploitation: DE/current-to-pbest/1 with archive
        """
        n = len(population)
        mutants = np.empty_like(population)
        
        # Compute exploitation mutations (current-to-pbest/1 with archive)
        exploit_mask = strategy_mask == 1
        if np.any(exploit_mask):
            mutants[exploit_mask] = self._mutate_current_to_pbest(
                population, fitness, archive, F_vals, exploit_mask
            )
        
        # Compute exploration mutations (rand/1 + vortex)
        explore_mask = strategy_mask == 0
        if np.any(explore_mask):
            mutants[explore_mask] = self._mutate_vortex_rand(
                population, F_vals, explore_mask, best_x
            )
        
        return mutants

    def _mutate_current_to_pbest(self, population, fitness, archive, F_vals, mask):
        """DE/current-to-pbest/1 mutation with external archive."""
        indices = np.where(mask)[0]
        n_mut = len(indices)
        
        # Select pbest (top p fraction, p in [0.05, 0.2])
        p = max(0.05, 0.2 - 0.15 * min(self.generation / (20 * self.dim), 1.0))
        n_pbest = max(1, int(p * len(population)))
        pbest_pool = np.argsort(fitness)[:n_pbest]
        pbest_idx = pbest_pool[np.random.randint(0, n_pbest, n_mut)]
        
        # Select r1 from population (different from current)
        r1 = self._random_indices_excluding(len(population), indices, n_mut)
        
        # Select r2 from population + archive
        combined = np.vstack([population, archive])
        r2 = np.random.randint(0, len(combined), n_mut)
        
        F = F_vals[indices, np.newaxis]
        current = population[indices]
        mutants = current + F * (population[pbest_idx] - current) + F * (population[r1] - combined[r2])
        return mutants

    def _mutate_vortex_rand(self, population, F_vals, mask, best_x):
        """DE/rand/1 mutation enhanced with a vortex spiral perturbation."""
        indices = np.where(mask)[0]
        n_mut = len(indices)
        
        # Standard rand/1 base
        r0 = self._random_indices_excluding(len(population), indices, n_mut)
        r1 = self._random_indices_excluding(len(population), indices, n_mut)
        r2 = self._random_indices_excluding(len(population), indices, n_mut)
        
        # Ensure r0, r1, r2 are all different (best effort)
        for _ in range(3):
            collide = (r1 == r0) | (r2 == r0) | (r2 == r1)
            if not np.any(collide):
                break
            r1[collide] = np.random.randint(0, len(population), np.sum(collide))
            r2[collide] = np.random.randint(0, len(population), np.sum(collide))
        
        F = F_vals[indices, np.newaxis]
        base_mutant = population[r0] + F * (population[r1] - population[r2])
        
        # Vortex perturbation: spiral offset around best
        vortex_perturbation = self._compute_vortex_perturbation(
            population[indices], best_x, n_mut
        )
        
        # Blend: small vortex component
        alpha = 0.15 * np.exp(-0.01 * self.generation)
        mutants = base_mutant + alpha * vortex_perturbation
        return mutants

    def _compute_vortex_perturbation(self, current, center, n):
        """Compute a spiral/vortex displacement around a center point.
        
        Uses rotation in random 2D subspaces to create spiral motion.
        """
        diff = current - center[np.newaxis, :]
        norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-30
        
        # Create random orthogonal perturbation via random rotation
        random_dirs = np.random.randn(n, self.dim)
        # Make orthogonal to diff
        proj = np.sum(random_dirs * diff, axis=1, keepdims=True) / (norms ** 2)
        random_dirs = random_dirs - proj * diff
        random_norms = np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-30
        random_dirs = random_dirs / random_norms
        
        # Spiral: radial contraction + tangential rotation
        angle = self.vortex_angle
        radial = -0.3 * diff  # pull toward center
        tangential = norms * random_dirs * np.sin(angle)
        
        perturbation = radial + tangential
        return perturbation

    def _advance_vortex(self):
        """Advance the vortex angle for the next generation."""
        self.vortex_angle += self.vortex_speed
        if self.vortex_angle > 2 * np.pi:
            self.vortex_angle -= 2 * np.pi
        # Slowly increase rotation speed
        self.vortex_speed = min(self.vortex_speed * 1.005, 0.5)

    def _random_indices_excluding(self, pool_size, exclude_indices, n):
        """Generate n random indices from [0, pool_size) trying to avoid exclude_indices."""
        result = np.random.randint(0, pool_size, n)
        collide = result == exclude_indices
        attempts = 0
        while np.any(collide) and attempts < 5:
            result[collide] = np.random.randint(0, pool_size, np.sum(collide))
            collide = result == exclude_indices
            attempts += 1
        return result

    def _crossover_batch(self, population, mutants, CR_vals):
        """Binomial crossover applied to the full population batch."""
        n, d = population.shape
        CR = CR_vals[:, np.newaxis]
        
        # Crossover mask
        rand_matrix = np.random.rand(n, d)
        cross_mask = rand_matrix < CR
        
        # Ensure at least one dimension is from mutant (j_rand)
        j_rand = np.random.randint(0, d, n)
        cross_mask[np.arange(n), j_rand] = True
        
        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to the search bounds. Uses bounce-back for out-of-bounds."""
        below = trials < self.lb
        above = trials > self.ub
        
        # Bounce-back: midpoint between bound and the corresponding parent dimension
        # For simplicity, just clip
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection: keep trial if it's better, else keep parent."""
        improved = trial_fitness < fitness
        improved = improved & np.isfinite(trial_fitness)
        population = np.where(improved[:, np.newaxis], trials, population)
        fitness = np.where(improved, trial_fitness, fitness)
        return population, fitness

    def _partial_select(self, population, fitness, trials, trial_fitness, valid,
                        F_vals, CR_vals, strategy_mask):
        """Handle partial evaluation results where some fitnesses may be NaN/Inf."""
        improved = valid & (trial_fitness < fitness)
        population = np.where(improved[:, np.newaxis], trials, population)
        fitness = np.where(improved, trial_fitness, fitness)
        # Also update parameter memory for valid successes
        self._update_parameter_memory(F_vals, CR_vals, strategy_mask,
                                      improved, fitness, trial_fitness)
        return population, fitness

    def _update_archive(self, archive, population, success_mask):
        """Add successful parent vectors to the archive, randomly replacing if full."""
        successful_parents = population[success_mask]
        if len(successful_parents) == 0:
            return archive
        
        new_archive = np.vstack([archive, successful_parents])
        if len(new_archive) > self.archive_size:
            keep_idx = np.random.choice(len(new_archive), self.archive_size, replace=False)
            new_archive = new_archive[keep_idx]
        return new_archive

    def _update_parameter_memory(self, F_vals, CR_vals, strategy_mask, 
                                  success_mask, old_fitness, new_fitness):
        """Update the parameter adaptation memory using Lehmer mean of successful parameters."""
        if not np.any(success_mask):
            return
        
        # Weight by fitness improvement
        improvements = np.abs(old_fitness[success_mask] - new_fitness[success_mask])
        improvements = np.where(np.isfinite(improvements), improvements, 0)
        total_imp = np.sum(improvements) + 1e-30
        weights = improvements / total_imp
        
        succ_F = F_vals[success_mask]
        succ_CR = CR_vals[success_mask]
        succ_strat = strategy_mask[success_mask]
        
        # Update explore pool
        explore_succ = succ_strat == 0
        if np.any(explore_succ):
            w_e = weights[explore_succ]
            w_e = w_e / (np.sum(w_e) + 1e-30)
            self.F_mean_explore = self._weighted_lehmer_mean(
                succ_F[explore_succ], w_e, self.F_mean_explore
            )
            self.CR_mean_explore = self._weighted_arithmetic_mean(
                succ_CR[explore_succ], w_e, self.CR_mean_explore
            )
        
        # Update exploit pool
        exploit_succ = succ_strat == 1
        if np.any(exploit_succ):
            w_x = weights[exploit_succ]
            w_x = w_x / (np.sum(w_x) + 1e-30)
            self.F_mean_exploit = self._weighted_lehmer_mean(
                succ_F[exploit_succ], w_x, self.F_mean_exploit
            )
            self.CR_mean_exploit = self._weighted_arithmetic_mean(
                succ_CR[exploit_succ], w_x, self.CR_mean_exploit
            )

    def _weighted_lehmer_mean(self, values, weights, current_mean):
        """Compute weighted Lehmer mean and blend with current mean (c=0.8)."""
        numerator = np.sum(weights * values ** 2)
        denominator = np.sum(weights * values) + 1e-30
        lehmer = numerator / denominator
        c = 0.8
        return (1 - c) * current_mean + c * np.clip(lehmer, 0.05, 1.5)

    def _weighted_arithmetic_mean(self, values, weights, current_mean):
        """Compute weighted arithmetic mean and blend with current mean (c=0.7)."""
        wm = np.sum(weights * values)
        c = 0.7
        return np.clip((1 - c) * current_mean + c * wm, 0.0, 1.0)

    def _detect_stagnation(self, best_f, prev_best_f):
        """Detect if the search has stagnated based on best fitness improvement."""
        rel_improvement = abs(prev_best_f - best_f) / (abs(prev_best_f) + 1e-30)
        if rel_improvement < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 2)
        return self.stagnation_counter >= self.stagnation_limit

    def _restart_if_stagnant(self, population, fitness, best_x, func, stopping_condition):
        """Perform opposition-based partial restart, preserving elite individuals."""
        self.stagnation_counter = 0
        n = len(population)
        
        # Keep top 20% elite
        n_elite = max(2, n // 5)
        elite_idx = np.argsort(fitness)[:n_elite]
        
        # Generate opposition-based population for the rest
        n_new = n - n_elite
        center = (self.lb + self.ub) / 2.0
        
        # Mix of opposition and random restart
        n_opposition = n_new // 2
        n_random = n_new - n_opposition
        
        # Opposition: reflect around center with jitter
        base_idx = np.random.choice(n, n_opposition, replace=True)
        opposition = 2 * center - population[base_idx]
        jitter = np.random.uniform(-5, 5, opposition.shape)
        opposition = opposition + jitter
        
        # Random restart around best with large sigma
        sigma = 30.0 * max(0.1, 1.0 - self.generation / (50 * self.dim))
        random_pop = best_x[np.newaxis, :] + sigma * np.random.randn(n_random, self.dim)
        
        # Combine
        new_pop = np.vstack([population[elite_idx], opposition, random_pop])
        new_pop = np.clip(new_pop, self.lb, self.ub)
        
        if stopping_condition():
            new_fitness = np.full(n, np.inf)
            new_fitness[:n_elite] = fitness[elite_idx]
            return new_pop, new_fitness
        
        new_fitness = func(new_pop)
        valid = self._get_valid_mask(new_fitness)
        new_fitness = np.where(valid, new_fitness, np.inf)
        
        # Reset vortex parameters
        self.vortex_angle = 0.0
        self.vortex_speed = 0.1
        
        return new_pop, new_fitness

    def _compute_diversity(self, population):
        """Compute population diversity as average distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid) ** 2, axis=1))
        return np.mean(distances)
