import numpy as np


class HybridDESimplexOptimizer:
    """
    Hybrid optimizer combining Differential Evolution with adaptive parameters,
    simplex-inspired local search moves, and intelligent restart mechanisms.
    
    Key features:
    - Multi-strategy DE mutation (current-to-pbest, rand/1, rand/2)
    - Simplex-like contraction/expansion moves for local refinement
    - Adaptive F and CR via success-history (SHADE-like)
    - Population diversity monitoring with restart on stagnation
    - Elite archive for maintaining good solutions across restarts
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 60), 300)
        self.archive_size = self.np_size
        self.p_best_rate = 0.15
        self.memory_size = 20
        self.simplex_prob = 0.15
        self.restart_threshold = 1e-12
        self.stagnation_limit = 50
        self.f_opt = np.inf
        self.x_opt = None

    def _initialize_population(self):
        """Create a random population uniformly in bounds."""
        pop = np.random.uniform(self.lb, self.ub, size=(self.np_size, self.dim))
        return pop

    def _initialize_memory(self):
        """Initialize success-history memory for F and CR adaptation."""
        self.memory_f = np.full(self.memory_size, 0.5)
        self.memory_cr = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

    def _initialize_archive(self):
        """Initialize the external archive for storing replaced solutions."""
        self.archive = np.empty((0, self.dim))

    def _initialize_stagnation_counter(self):
        """Reset stagnation tracking variables."""
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf

    def _clip_to_bounds(self, pop):
        """Clip population to search bounds."""
        return np.clip(pop, self.lb, self.ub)

    def _generate_adaptive_params(self, n):
        """Generate F and CR values from success-history memory (SHADE-style)."""
        indices = np.random.randint(0, self.memory_size, size=n)
        mu_f = self.memory_f[indices]
        mu_cr = self.memory_cr[indices]
        
        # Cauchy distribution for F
        f_vals = mu_f + 0.1 * np.random.standard_cauchy(n)
        f_vals = np.clip(f_vals, 0.05, 1.0)
        # Truncate negative values by regenerating
        neg_mask = f_vals <= 0
        while np.any(neg_mask):
            f_vals[neg_mask] = mu_f[neg_mask] + 0.1 * np.random.standard_cauchy(np.sum(neg_mask))
            f_vals = np.clip(f_vals, 0.05, 1.0)
            neg_mask = f_vals <= 0
        
        # Normal distribution for CR
        cr_vals = np.random.normal(mu_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0.0, 1.0)
        
        return f_vals, cr_vals

    def _select_pbest_indices(self, fitness, n):
        """Select random indices from the p-best portion of the population."""
        p = max(2, int(self.p_best_rate * len(fitness)))
        sorted_indices = np.argsort(fitness)[:p]
        chosen = sorted_indices[np.random.randint(0, p, size=n)]
        return chosen

    def _mutate_batch(self, pop, fitness, f_vals):
        """
        Generate mutant vectors using current-to-pbest/1 strategy with archive.
        v_i = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2_or_archive)
        """
        n = len(pop)
        # pbest indices
        pbest_idx = self._select_pbest_indices(fitness, n)
        
        # r1 != i
        r1 = np.random.randint(0, n, size=n)
        for i in range(n):
            while r1[i] == i:
                r1[i] = np.random.randint(0, n)
        
        # r2 from pop + archive, != i, != r1
        combined = np.vstack([pop, self.archive]) if len(self.archive) > 0 else pop.copy()
        n_combined = len(combined)
        r2 = np.random.randint(0, n_combined, size=n)
        for i in range(n):
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, n_combined)
        
        f_col = f_vals[:, np.newaxis]
        mutants = pop + f_col * (pop[pbest_idx] - pop) + f_col * (pop[r1] - combined[r2])
        return mutants

    def _crossover_batch(self, pop, mutants, cr_vals):
        """Binomial crossover between population and mutants."""
        n, d = pop.shape
        rand_matrix = np.random.rand(n, d)
        j_rand = np.random.randint(0, d, size=n)
        
        # Create crossover mask
        mask = rand_matrix < cr_vals[:, np.newaxis]
        # Ensure at least one dimension from mutant
        for i in range(n):
            mask[i, j_rand[i]] = True
        
        trials = np.where(mask, mutants, pop)
        return trials

    def _simplex_moves_batch(self, pop, fitness):
        """
        Simplex-inspired moves: reflection/contraction toward best solutions.
        Applied to a random subset of the population.
        """
        n = len(pop)
        n_simplex = max(1, int(self.simplex_prob * n))
        indices = np.random.choice(n, size=n_simplex, replace=False)
        
        # Find centroid of top solutions
        k = max(3, int(0.1 * n))
        top_k = np.argsort(fitness)[:k]
        centroid = np.mean(pop[top_k], axis=0)
        
        simplex_trials = pop.copy()
        # Reflection toward centroid with random contraction
        alpha = np.random.uniform(0.3, 1.5, size=(n_simplex, 1))
        simplex_trials[indices] = centroid + alpha * (centroid - pop[indices])
        
        return simplex_trials, indices

    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        """Greedy selection: keep the better of parent vs trial for each index."""
        valid = ~np.isnan(trial_fitness)
        improved = np.zeros(len(pop), dtype=bool)
        
        mask = valid & (trial_fitness < fitness)
        improved[mask] = True
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        new_pop[mask] = trials[mask]
        new_fitness[mask] = trial_fitness[mask]
        
        return new_pop, new_fitness, improved

    def _update_archive(self, pop, improved_mask):
        """Add replaced parents to archive, trim to max size."""
        replaced = pop[improved_mask]
        if len(replaced) > 0:
            self.archive = np.vstack([self.archive, replaced]) if len(self.archive) > 0 else replaced.copy()
            if len(self.archive) > self.archive_size:
                keep = np.random.choice(len(self.archive), self.archive_size, replace=False)
                self.archive = self.archive[keep]

    def _update_memory(self, f_vals, cr_vals, improved_mask, fitness_diff):
        """Update success-history memory with weighted Lehmer mean (SHADE-style)."""
        if not np.any(improved_mask):
            return
        
        s_f = f_vals[improved_mask]
        s_cr = cr_vals[improved_mask]
        s_diff = fitness_diff[improved_mask]
        
        if len(s_f) == 0 or np.sum(s_diff) == 0:
            return
        
        weights = s_diff / np.sum(s_diff)
        
        # Weighted Lehmer mean for F
        denom = np.sum(weights * s_f)
        if denom > 0:
            new_f = np.sum(weights * s_f ** 2) / denom
        else:
            new_f = self.memory_f[self.memory_idx]
        
        # Weighted mean for CR
        new_cr = np.sum(weights * s_cr)
        
        self.memory_f[self.memory_idx] = np.clip(new_f, 0.05, 1.0)
        self.memory_cr[self.memory_idx] = np.clip(new_cr, 0.0, 1.0)
        self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def _update_best(self, pop, fitness):
        """Track the global best solution found so far."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_idx = np.where(valid)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.f_opt:
            self.f_opt = fitness[best_idx]
            self.x_opt = pop[best_idx].copy()

    def _compute_diversity(self, pop):
        """Compute population diversity as average distance from centroid."""
        centroid = np.mean(pop, axis=0)
        diffs = pop - centroid
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _detect_stagnation(self, fitness):
        """Check if the population has stagnated based on fitness improvement."""
        current_best = np.nanmin(fitness)
        if np.isnan(current_best):
            return False
        
        improvement = self.prev_best_fitness - current_best
        if improvement < self.restart_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.prev_best_fitness = current_best
        return self.stagnation_counter >= self.stagnation_limit

    def _restart(self, pop, fitness):
        """
        Restart the population while preserving elite solutions.
        Uses a mix of random individuals and perturbations around the best.
        """
        n = len(pop)
        n_elite = max(2, int(0.1 * n))
        elite_idx = np.argsort(fitness)[:n_elite]
        elite = pop[elite_idx].copy()
        
        # New random portion
        n_random = n - n_elite - n_elite  # elite + perturbations + random
        n_perturbed = min(n_elite, n - n_elite)
        n_random = max(0, n - n_elite - n_perturbed)
        
        # Perturbed copies of elites
        scale = max(5.0, self._compute_diversity(pop) * 2.0)
        perturbed = elite[:n_perturbed] + np.random.normal(0, scale, size=(n_perturbed, self.dim))
        
        # Random new individuals
        random_pop = np.random.uniform(self.lb, self.ub, size=(n_random, self.dim))
        
        new_pop = np.vstack([elite, perturbed, random_pop])[:n]
        new_pop = self._clip_to_bounds(new_pop)
        
        # Reset adaptation
        self._initialize_memory()
        self._initialize_archive()
        self._initialize_stagnation_counter()
        
        return new_pop

    def _local_search_batch(self, pop, fitness):
        """
        Apply local search to top solutions using simplex-like contraction.
        Creates trial solutions by averaging pairs of good solutions with perturbation.
        """
        n = len(pop)
        k = max(3, int(0.15 * n))
        top_idx = np.argsort(fitness)[:k]
        top_pop = pop[top_idx]
        
        # Generate pairs and average them with small perturbation
        idx1 = np.random.choice(k, size=k)
        idx2 = np.random.choice(k, size=k)
        
        scale = np.std(top_pop, axis=0) * 0.5 + 1e-10
        noise = np.random.normal(0, 1, size=(k, self.dim)) * scale
        
        local_trials = 0.5 * (top_pop[idx1] + top_pop[idx2]) + noise
        local_trials = self._clip_to_bounds(local_trials)
        
        return local_trials, top_idx

    def _handle_truncated_result(self, trial_fitness, n_expected):
        """Handle case where func returns fewer values than expected."""
        if len(trial_fitness) < n_expected:
            padded = np.full(n_expected, np.nan)
            padded[:len(trial_fitness)] = trial_fitness
            return padded, True
        return trial_fitness, False

    def _reduce_population_size(self, pop, fitness, min_size=None):
        """Optionally reduce population size for intensification phase."""
        if min_size is None:
            min_size = max(6, int(0.5 * self.np_size))
        
        if len(pop) <= min_size:
            return pop, fitness
        
        new_size = max(min_size, len(pop) - 2)
        sorted_idx = np.argsort(fitness)[:new_size]
        return pop[sorted_idx], fitness[sorted_idx]

    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating all components."""
        self.f_opt = np.inf
        self.x_opt = None
        
        # Initialize
        pop = self._initialize_population()
        self._initialize_memory()
        self._initialize_archive()
        self._initialize_stagnation_counter()
        
        # Initial evaluation
        fitness = func(pop)
        if stopping_condition():
            self._update_best(pop, fitness)
            return self.f_opt, self.x_opt
        
        fitness, truncated = self._handle_truncated_result(fitness, len(pop))
        if truncated:
            self._update_best(pop, fitness)
            return self.f_opt, self.x_opt
        
        self._update_best(pop, fitness)
        generation = 0
        
        while not stopping_condition():
            n = len(pop)
            
            # Generate adaptive parameters
            f_vals, cr_vals = self._generate_adaptive_params(n)
            
            # DE mutation and crossover
            mutants = self._mutate_batch(pop, fitness, f_vals)
            mutants = self._clip_to_bounds(mutants)
            trials = self._crossover_batch(pop, mutants, cr_vals)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trials
            trial_fitness = func(trials)
            if stopping_condition():
                trial_fitness, _ = self._handle_truncated_result(trial_fitness, n)
                self._update_best(trials, trial_fitness)
                break
            trial_fitness, truncated = self._handle_truncated_result(trial_fitness, n)
            if truncated:
                self._update_best(trials, trial_fitness)
                break
            
            # Selection
            old_pop = pop.copy()
            old_fitness = fitness.copy()
            pop, fitness, improved = self._select_survivors_batch(pop, fitness, trials, trial_fitness)
            
            # Compute fitness differences for memory update
            fitness_diff = np.maximum(0, old_fitness - fitness)
            
            # Update archive and memory
            self._update_archive(old_pop, improved)
            self._update_memory(f_vals, cr_vals, improved, fitness_diff)
            self._update_best(pop, fitness)
            
            # Periodic local search
            if generation % 5 == 0 and not stopping_condition():
                local_trials, local_idx = self._local_search_batch(pop, fitness)
                local_fitness = func(local_trials)
                if stopping_condition():
                    local_fitness, _ = self._handle_truncated_result(local_fitness, len(local_trials))
                    self._update_best(local_trials, local_fitness)
                    break
                local_fitness, truncated = self._handle_truncated_result(local_fitness, len(local_trials))
                if truncated:
                    self._update_best(local_trials, local_fitness)
                    break
                
                # Replace if improved
                k = len(local_idx)
                valid_local = ~np.isnan(local_fitness[:k])
                for j in range(k):
                    if valid_local[j] and local_fitness[j] < fitness[local_idx[j]]:
                        pop[local_idx[j]] = local_trials[j]
                        fitness[local_idx[j]] = local_fitness[j]
                
                self._update_best(pop, fitness)
            
            # Simplex moves periodically
            if generation % 8 == 3 and not stopping_condition():
                simplex_trials, simplex_idx = self._simplex_moves_batch(pop, fitness)
                s_trials = simplex_trials[simplex_idx]
                s_fitness = func(s_trials)
                if stopping_condition():
                    s_fitness, _ = self._handle_truncated_result(s_fitness, len(s_trials))
                    self._update_best(s_trials, s_fitness)
                    break
                s_fitness, truncated = self._handle_truncated_result(s_fitness, len(s_trials))
                if truncated:
                    self._update_best(s_trials, s_fitness)
                    break
                
                for j, idx in enumerate(simplex_idx):
                    if not np.isnan(s_fitness[j]) and s_fitness[j] < fitness[idx]:
                        pop[idx] = s_trials[j]
                        fitness[idx] = s_fitness[j]
                
                self._update_best(pop, fitness)
            
            # Check stagnation and restart if needed
            if self._detect_stagnation(fitness):
                pop = self._restart(pop, fitness)
                fitness = func(pop)
                if stopping_condition():
                    fitness, _ = self._handle_truncated_result(fitness, len(pop))
                    self._update_best(pop, fitness)
                    break
                fitness, truncated = self._handle_truncated_result(fitness, len(pop))
                if truncated:
                    self._update_best(pop, fitness)
                    break
                self._update_best(pop, fitness)
            
            generation += 1
        
        return self.f_opt, self.x_opt
