import numpy as np


class AdaptiveArchiveDEA:
    """
    Differential Evolution with Adaptive Archive and Fitness-Weighted Composite Mutation.
    
    Key innovations:
    - Fitness-weighted composite mutation: weights difference vectors by their
      historical fitness improvement contributions
    - Bounded archive of diverse elite solutions that expands search reach
    - Adaptive step-size with momentum-like accumulation
    - Dynamic ring topology with adaptive neighborhood radius
    - Stagnation detection with diversity-preserving restart
    """
    
    def __init__(self, pop_size=None, dim=30, bounds=(-100, 100), seed=None):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size if pop_size else max(50, 10 * dim)
        self.pop_size = min(self.pop_size, 500)  # cap runaway size
        self.rng = np.random.default_rng(seed)
        
        # Mutable state (reset on each run)
        self.population = None
        self.fitness = None
        self.mutant = None
        self.trial = None
        self.archive = None
        self.step_size = None
        self.crossover_rate = None
        self.momentum = None
        self.gen = None
        self.stagnation_counter = None
        self.best_fitness_history = None
        self.neighbor_radius = None
        
    def __call__(self, func, stopping_condition):
        self._reset_state()
        self._initialize_population()
        self._initialize_parameters()
        self._evaluate(func)
        
        while not stopping_condition():
            self._record_fitness_history()
            self._select_parents()
            self._mutate()
            self._crossover()
            self._clip_to_bounds()
            self._evaluate_trial(func)
            self._select_survivors()
            self._update_archive()
            self._adapt_step_size()
            self._adapt_crossover_rate()
            self._adapt_topology()
            self._check_stagnation(func)
            self.gen += 1
            
        return self._return_best()
    
    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    
    def _reset_state(self):
        """Reset all mutable state to initial values."""
        self.population = None
        self.fitness = None
        self.mutant = None
        self.trial = None
        self.archive = np.array([], dtype=np.float64).reshape(0, self.dim)
        self.step_size = 0.5
        self.crossover_rate = 0.5
        self.momentum = np.zeros(self.dim)
        self.gen = 0
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.neighbor_radius = 3
    
    def _initialize_population(self):
        """Initialize population with random candidates in bounds."""
        low, high = self.bounds
        self.population = self.rng.uniform(low, high, (self.pop_size, self.dim))
        self._clip_to_bounds()
    
    def _initialize_parameters(self):
        """Initialize adaptive parameters."""
        self.step_size = 0.5 + self.rng.uniform(-0.1, 0.1)
        self.crossover_rate = 0.5 + self.rng.uniform(-0.1, 0.1)
        self.momentum = np.zeros(self.dim)
    
    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    
    def _clip_to_bounds(self):
        """Clip all candidates to problem bounds."""
        low, high = self.bounds
        self.population = np.clip(self.population, low, high)
        if self.mutant is not None:
            self.mutant = np.clip(self.mutant, low, high)
        if self.trial is not None:
            self.trial = np.clip(self.trial, low, high)
    
    def _evaluate(self, func):
        """Evaluate fitness for entire population."""
        self.fitness = func(self.population)
        self._sanitize_fitness()
    
    def _evaluate_trial(self, func):
        """Evaluate fitness for trial vector only."""
        fitness = func(self.trial.reshape(1, -1))[0]
        if not np.isfinite(fitness):
            fitness = 1e20
        self.trial_fitness = fitness
    
    def _sanitize_fitness(self):
        """Replace NaN/Inf with large penalty value."""
        mask = ~np.isfinite(self.fitness)
        self.fitness[mask] = 1e20
    
    # ------------------------------------------------------------------
    # Variation operators
    # ------------------------------------------------------------------
    
    def _select_parents(self):
        """Select indices for mutation parents using tournament on ring."""
        n = self.pop_size
        r = self.neighbor_radius
        
        # Build ring neighbor list for each individual
        self.parent_indices = np.zeros((n, 4), dtype=np.intp)
        
        for i in range(n):
            # Ring neighbors with wrap-around
            candidates = [(i - 2) % n, (i - 1) % n, (i + 1) % n, (i + 2) % n]
            
            # Shuffle candidates deterministically using seed
            self.rng.shuffle(candidates)
            
            # Tournament selection based on fitness
            selected = sorted(candidates, key=lambda idx: self.fitness[idx])[:3]
            self.parent_indices[i] = selected
    
    def _compute_fitness_weights(self, idx1, idx2):
        """
        Compute fitness-based weight for a difference vector.
        Returns weight in [0.5, 1.5] based on relative fitness of source points.
        """
        f1 = self.fitness[idx1]
        f2 = self.fitness[idx2]
        f_best = np.min(self.fitness)
        f_worst = np.max(self.fitness)
        
        # Avoid division by zero
        f_range = max(f_worst - f_best, 1e-10)
        
        # Weight based on average normalized fitness of the two individuals
        avg_norm = 0.5 * ((f1 - f_best) + (f2 - f_best)) / f_range
        weight = 1.0 + (0.5 - avg_norm)
        
        return np.clip(weight, 0.5, 1.5)
    
    def _mutate(self):
        """
        Adaptive archive composite mutation:
        mutant = x_base + F1*(x_a - x_b) + F2*(archive_x - x_c)
        where F1 and F2 are adapted and difference vectors are fitness-weighted.
        """
        n = self.pop_size
        self.mutant = np.empty((n, self.dim))
        
        # Archive sampling: if archive empty, use random population members
        if len(self.archive) < 3:
            archive_sample = self.population[self.rng.integers(0, n, size=3)]
        else:
            archive_sample = self.archive[self.rng.integers(0, len(self.archive), size=3)]
        
        for i in range(n):
            # Base vector: best of current and two ring neighbors
            base_idx = self.parent_indices[i, 0]
            x_base = self.population[base_idx]
            
            # First difference vector: ring neighbors (fitness-weighted)
            idx_a = self.parent_indices[i, 1]
            idx_b = self.parent_indices[i, 2]
            diff1 = self.population[idx_a] - self.population[idx_b]
            w1 = self._compute_fitness_weights(idx_a, idx_b)
            
            # Second difference vector: archive vs random population
            idx_c = self.rng.integers(0, n)
            diff2 = archive_sample[0] - self.population[idx_c]
            w2 = self.rng.uniform(0.5, 1.5)  # Random weight for archive diff
            
            # Combine with step-size and momentum
            self.momentum = 0.8 * self.momentum + 0.2 * diff1
            step1 = self.step_size * w1 * self.momentum
            step2 = 0.5 * self.step_size * w2 * diff2
            
            self.mutant[i] = x_base + step1 + step2
    
    def _crossover(self):
        """
        Binomial crossover with adaptive rate.
        Each dimension has independent crossover probability.
        """
        n, d = self.pop_size, self.dim
        self.trial = np.empty((n, d))
        
        # Ensure at least one dimension comes from mutant
        keep_dims = self.rng.integers(0, d, size=n)
        
        for i in range(n):
            # Per-dimension crossover mask
            mask = self.rng.random(d) < self.crossover_rate
            mask[keep_dims[i]] = True
            
            self.trial[i] = np.where(mask, self.mutant[i], self.population[i])
    
    def _select_survivors(self):
        """One-to-one greedy selection between trial and population."""
        better = self.trial_fitness < self.fitness
        self.population[better] = self.trial[better]
        self.fitness[better] = self.trial_fitness[better]
    
    # ------------------------------------------------------------------
    # Archive management
    # ------------------------------------------------------------------
    
    def _update_archive(self):
        """Add improved solutions to bounded archive, maintaining diversity."""
        # Add best individuals not already in archive
        best_k = min(5, self.pop_size // 10)
        best_indices = np.argpartition(self.fitness, best_k)[:best_k]
        
        candidates = self.population[best_indices]
        
        # Diversity check: only add if sufficiently different from existing
        for cand in candidates:
            if self._is_diverse_enough(cand):
                if len(self.archive) < self.pop_size:
                    self.archive = np.vstack([self.archive, cand])
                else:
                    # Replace random if archive full
                    replace_idx = self.rng.integers(0, len(self.archive))
                    self.archive[replace_idx] = cand
    
    def _is_diverse_enough(self, candidate, threshold=0.1):
        """Check if candidate is sufficiently different from archive."""
        if len(self.archive) == 0:
            return True
        
        # Compute minimum distance to archive
        distances = np.linalg.norm(self.archive - candidate, axis=1)
        min_dist = np.min(distances)
        
        # Normalize by problem dimension and range
        range_size = self.bounds[1] - self.bounds[0]
        normalized_dist = min_dist / (range_size * np.sqrt(self.dim))
        
        return normalized_dist > threshold
    
    # ------------------------------------------------------------------
    # Adaptation mechanisms
    # ------------------------------------------------------------------
    
    def _adapt_step_size(self):
        """
        Adapt step-size based on recent improvement rate.
        Uses exponential moving average with bounded adjustment.
        """
        if len(self.best_fitness_history) < 5:
            return
        
        recent = self.best_fitness_history[-5:]
        if len(recent) < 2:
            return
        
        # Compute relative improvement
        f_init = recent[0]
        f_curr = recent[-1]
        improvement = (f_init - f_curr) / (abs(f_init) + 1e-10)
        
        # Adjust step-size: increase on good progress, decrease otherwise
        if improvement > 0.01:
            factor = 1.1  # Exploratory
        elif improvement > 0:
            factor = 1.0  # Stable
        else:
            factor = 0.9  # Exploitative
        
        self.step_size = np.clip(self.step_size * factor, 0.01, 2.0)
    
    def _adapt_crossover_rate(self):
        """
        Adapt crossover rate based on population diversity.
        High diversity -> higher crossover (recombination power)
        Low diversity -> lower crossover (preserves good solutions)
        """
        diversity = self._compute_diversity()
        
        # Target crossover rate inversely related to diversity
        target_cr = 0.9 - 0.6 * np.clip(diversity, 0, 1)
        
        # Smooth adaptation
        self.crossover_rate = 0.9 * self.crossover_rate + 0.1 * target_cr
        self.crossover_rate = np.clip(self.crossover_rate, 0.1, 0.95)
    
    def _adapt_topology(self):
        """
        Adapt neighborhood radius based on stagnation.
        Increases radius when stuck to escape local optima.
        """
        if self.stagnation_counter > 10:
            self.neighbor_radius = min(self.neighbor_radius + 1, self.pop_size // 4)
        elif self.stagnation_counter == 0 and self.gen > 20:
            self.neighbor_radius = max(2, self.neighbor_radius - 1)
    
    # ------------------------------------------------------------------
    # Monitoring and control
    # ------------------------------------------------------------------
    
    def _compute_diversity(self):
        """
        Compute normalized population diversity as average relative
        distance to centroid.
        """
        if self.population is None or len(self.population) < 2:
            return 0.0
        
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        avg_dist = np.mean(distances)
        
        range_size = self.bounds[1] - self.bounds[0]
        normalized = avg_dist / (range_size * np.sqrt(self.dim))
        
        return np.clip(normalized, 0, 1)
    
    def _record_fitness_history(self):
        """Record best fitness for stagnation detection."""
        best_f = np.min(self.fitness)
        self.best_fitness_history.append(best_f)
        
        # Keep history bounded
        if len(self.best_fitness_history) > 50:
            self.best_fitness_history = self.best_fitness_history[-50:]
    
    def _check_stagnation(self, func):
        """
        Check for stagnation and trigger diversity-preserving restart if needed.
        """
        if len(self.best_fitness_history) < 10:
            return
        
        # Detect if best hasn't improved recently
        recent_best = min(self.best_fitness_history[-10:])
        history_best = min(self.best_fitness_history)
        
        if recent_best == history_best:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Restart if severely stagnant
        if self.stagnation_counter >= 20:
            self._restart_if_stagnant(func)
    
    def _restart_if_stagnant(self, func):
        """
        Diversity-preserving restart: keep best solutions, reinitialize rest.
        """
        # Preserve best individuals
        n_preserve = max(2, self.pop_size // 5)
        best_indices = np.argpartition(self.fitness, n_preserve)[:n_preserve]
        preserved = self.population[best_indices].copy()
        preserved_fitness = self.fitness[best_indices].copy()
        
        # Reinitialize others with diversity injection
        low, high = self.bounds
        new_count = self.pop_size - n_preserve
        
        # Partially random, partially from Gaussian perturbation of best
        new_random = self.rng.uniform(low, high, (new_count // 2, self.dim))
        
        best_one = self.population[np.argmin(self.fitness)]
        perturbation_scale = 0.3 * (high - low)
        new_perturbed = best_one + self.rng.normal(
            0, perturbation_scale, (new_count - new_count // 2, self.dim)
        )
        
        # Combine and clip
        new_individuals = np.vstack([new_random, new_perturbed])
        new_individuals = np.clip(new_individuals, low, high)
        
        # Reconstruct population
        self.population[:n_preserve] = preserved
        self.population[n_preserve:] = new_individuals
        
        # Re-evaluate
        self._evaluate(func)
        
        # Reset stagnation counter and adaptive parameters
        self.stagnation_counter = 0
        self.step_size = 0.5 + self.rng.uniform(-0.1, 0.1)
        self.crossover_rate = 0.5 + self.rng.uniform(-0.1, 0.1)
        self.momentum = np.zeros(self.dim)
        self.neighbor_radius = 3
    
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    
    def _return_best(self):
        """Return best fitness and corresponding solution."""
        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], self.population[best_idx].copy()
