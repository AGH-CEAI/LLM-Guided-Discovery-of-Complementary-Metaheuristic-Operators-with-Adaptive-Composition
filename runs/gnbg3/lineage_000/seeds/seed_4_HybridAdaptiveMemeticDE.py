import numpy as np


class HybridAdaptiveMemeticDE:
    """
    Hybrid Adaptive Memetic Differential Evolution with Gradient-Approximated
    Local Search (HAMDE-GALS).
    
    Key components:
    - DE/best/2 mutation with adaptive scaling factor (CR adaptation via success history)
    - Sliding window archive of superior solutions
    - Gradient-approximated local search triggered stochastically on top individuals
    - Adaptive population sizing with restart on stagnation
    - Quasi-opposition-based initialization for better coverage
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population sizing: 6*dim provides good balance
        self.NP = min(max(6 * dim, 20), 300)
        
        # Adaptive parameters
        self.F = 0.5  # Initial scaling factor
        self.CR = 0.5  # Initial crossover rate
        self.p_best = 0.1  # Probability of using best for mutation
        
        # Archive for storing good solutions
        self.archive_size = self.NP
        self.archive = None
        
        # Local search parameters
        self.local_search_freq = 0.05  # Probability of triggering local search per generation
        self.local_search_rate = 0.1  # Step size for gradient approximation
        self.local_search_iters = 3  # Max local search iterations per trigger
        
        # Diversity tracking
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 50
        
        # Success history for adaptation
        self.success_cr = []
        self.success_f = []
        self.avg_f = 0.6
        self.avg_cr = 0.5
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population_quasi_opposition(self, func):
        """Initialize population using quasi-opposition sampling for better coverage."""
        # Generate random population
        population = np.random.uniform(
            self.lower, self.upper, size=(self.NP, self.dim)
        )
        
        # Compute opposition points
        midpoint = (self.lower + self.upper) / 2
        opposition = midpoint * 2 - population
        opposition = np.clip(opposition, self.lower, self.upper)
        
        # Evaluate both and keep the better half from combined pool
        combined = np.vstack([population, opposition])
        fitness = func(combined)
        
        # Select NP best from combined
        best_indices = np.argsort(fitness)[:self.NP]
        population = combined[best_indices]
        fitness = fitness[best_indices]
        
        # Initialize archive with best solutions
        self.archive = population[:min(10, self.NP)].copy()
        
        return population, fitness
    
    def _update_archive_batch(self, better_solutions):
        """Update the archive with newly discovered superior solutions."""
        if self.archive is None:
            self.archive = better_solutions.copy()
            return
        
        # Combine and keep best unique solutions up to archive_size
        combined = np.vstack([self.archive, better_solutions])
        
        # Remove duplicates approximately
        unique_mask = self._find_unique_rows(combined)
        combined = combined[unique_mask]
        
        # Keep best archive_size solutions
        if len(combined) > self.archive_size:
            archive_fitness = func(combined)
            best_idx = np.argsort(archive_fitness)[:self.archive_size]
            self.archive = combined[best_idx]
        else:
            self.archive = combined
    
    def _find_unique_rows(self, arr, tolerance=1e-5):
        """Find indices of approximately unique rows."""
        n = len(arr)
        unique = np.ones(n, dtype=bool)
        for i in range(n):
            if unique[i]:
                diffs = np.linalg.norm(arr[i] - arr[i+1:], axis=1)
                unique[i+1:][diffs < tolerance] = False
        return unique
    
    def _mutate_batch(self, population, fitness):
        """Generate mutant vectors using DE/best/2 with archive enhancement."""
        # Select indices for mutation
        idx_best = np.argmin(fitness)
        best_vector = population[idx_best]
        
        # Prepare candidate pool (population + archive)
        if self.archive is not None and len(self.archive) >= 4:
            candidate_pool = np.vstack([population, self.archive])
        else:
            candidate_pool = population
        
        n_candidates = len(candidate_pool)
        
        # Select 4 distinct random indices from candidate pool
        perm = np.random.permutation(n_candidates)[:4]
        r1, r2, r3, r4 = candidate_pool[perm[0]], candidate_pool[perm[1]], \
                         candidate_pool[perm[2]], candidate_pool[perm[3]]
        
        # Adaptive F with heavy-tailed distribution
        F_adapted = self.F * np.random.standard_cauchy()
        F_adapted = max(0.0, min(2.0, F_adapted))  # Bound F
        
        # DE/best/2 mutation with archive contribution
        mutant = best_vector + F_adapted * (r1 - r2) + F_adapted * (r3 - r4)
        
        return mutant
    
    def _crossover_batch(self, target, mutant):
        """Binomial crossover with adaptive CR."""
        # Generate random CR for this trial
        cr_trial = np.random.normal(self.avg_cr, 0.1)
        cr_trial = max(0.0, min(1.0, cr_trial))
        
        # Binomial crossover
        cross_mask = np.random.random(self.dim) < cr_trial
        
        # Ensure at least one dimension is different
        if not np.any(cross_mask):
            cross_mask[np.random.randint(self.dim)] = True
        
        trial = np.where(cross_mask, mutant, target)
        return trial
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        # Sample-based computation for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diffs = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        
        # Average of upper triangle
        n = len(sample)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        avg_distance = distances[mask].mean() if distances[mask].size > 0 else 0.0
        
        return avg_distance / (self.upper - self.lower)  # Normalized
    
    def _gradient_approximate_batch(self, x, func):
        """Compute gradient approximation using forward differences for a batch."""
        grad = np.zeros_like(x)
        f_base = func(x)
        
        # Handle truncated output
        if len(f_base) < len(x):
            return grad
        
        for d in range(self.dim):
            # Perturb each dimension
            perturbed = x.copy()
            perturbed[:, d] += self.local_search_rate
            perturbed = np.clip(perturbed, self.lower, self.upper)
            
            f_perturbed = func(perturbed)
            
            # Handle truncated output
            if len(f_perturbed) >= len(x):
                grad[:, d] = (f_perturbed - f_base) / self.local_search_rate
        
        return grad
    
    def _local_search_batch(self, population, fitness, func):
        """Perform local search on top-performing individuals."""
        # Select top candidates for local search
        n_top = max(1, int(self.NP * 0.1))
        top_indices = np.argsort(fitness)[:n_top]
        
        local_candidates = population[top_indices].copy()
        
        for _ in range(self.local_search_iters):
            # Compute gradient approximation
            grad = self._gradient_approximate_batch(local_candidates, func)
            
            # Gradient descent step with momentum
            step = -self.local_search_rate * grad
            new_candidates = local_candidates + step
            new_candidates = self._clip_to_bounds(new_candidates)
            
            # Evaluate
            new_fitness = func(new_candidates)
            
            # Handle truncated output
            if len(new_fitness) < len(new_candidates):
                break
            
            # Accept improvements
            improved_mask = new_fitness < fitness[top_indices]
            local_candidates[improved_mask] = new_candidates[improved_mask]
        
        return local_candidates, top_indices
    
    def _adapt_parameters(self, succeeded):
        """Adapt CR and F based on success history."""
        if succeeded:
            self.success_cr.append(self.avg_cr)
            self.success_f.append(self.avg_f)
        
        # Keep sliding window of last 10 successes
        if len(self.success_cr) > 10:
            self.success_cr = self.success_cr[-10:]
            self.success_f = self.success_f[-10:]
        
        # Update averages
        if self.success_cr:
            self.avg_cr = 0.5 * self.avg_cr + 0.5 * np.mean(self.success_cr)
            self.avg_f = 0.5 * self.avg_f + 0.5 * np.mean(self.success_f)
        
        # Bound parameters
        self.avg_cr = max(0.1, min(0.9, self.avg_cr))
        self.avg_f = max(0.1, min(1.5, self.avg_f))
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors using greedy (μ+λ) selection."""
        # Combine parents and offspring
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        
        # Select NP best
        survival_indices = np.argsort(combined_fit)[:self.NP]
        new_population = combined_pop[survival_indices]
        new_fitness = combined_fit[survival_indices]
        
        return new_population, new_fitness
    
    def _restart_if_stagnant(self, population, fitness, func):
        """Reinitialize population partially if stagnation detected."""
        diversity = self._compute_diversity(population)
        
        if diversity < self.diversity_threshold or self.stagnation_counter > self.max_stagnation:
            # Reinitialize 70% of population
            n_reinit = int(0.7 * self.NP)
            reinit_indices = np.random.choice(self.NP, n_reinit, replace=False)
            
            new_individuals = np.random.uniform(
                self.lower, self.upper, size=(n_reinit, self.dim)
            )
            
            # Inject some best solutions unchanged
            population[reinit_indices] = new_individuals
            fitness = func(population)
            
            self.stagnation_counter = 0
            self.archive = population[:min(10, self.NP)].copy()
        
        return population, fitness
    
    def _check_termination_conditions(self, fitness, trial_fitness, trials, stopping_condition):
        """Check if optimization should terminate."""
        # Check budget exhaustion
        if len(trial_fitness) < len(trials):
            return True
        
        # Check user stopping condition
        if stopping_condition():
            return True
        
        return False
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize with quasi-opposition
        population, fitness = self._initialize_population_quasi_opposition(func)
        
        # Track best
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()
        
        gen = 0
        prev_best = f_opt
        
        while not stopping_condition():
            # Check termination after population initialization
            if self._check_termination_conditions(fitness, fitness, population, stopping_condition):
                break
            
            # Build trial batch (one trial per individual for fair comparison)
            trials = np.zeros((self.NP, self.dim))
            trial_fitness = np.zeros(self.NP)
            
            for i in range(self.NP):
                mutant = self._mutate_batch(population, fitness)
                trials[i] = self._crossover_batch(population[i], mutant)
            
            # Clip to bounds before evaluation
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trial batch
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < self.NP:
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                
                if valid_trials == 0:
                    break
            
            # Check stopping after trial evaluation
            if stopping_condition():
                break
            
            # Track improvements for adaptation
            improvements = trial_fitness < fitness[:len(trial_fitness)]
            
            # Update archive with improved solutions
            if np.any(improvements):
                improved_solutions = trials[improvements]
                self._update_archive_batch(improved_solutions)
            
            # Select survivors
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Adapt parameters based on success
            self._adapt_parameters(np.any(improvements))
            
            # Update best
            best_idx = np.argmin(fitness)
            current_best = fitness[best_idx]
            
            if current_best < f_opt:
                f_opt = current_best
                x_opt = population[best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Trigger local search stochastically
            if np.random.random() < self.local_search_freq and gen > 5:
                local_candidates, top_indices = self._local_search_batch(
                    population, fitness, func
                )
                
                # Merge local search results
                for idx, (candidate, top_idx) in enumerate(zip(local_candidates, top_indices)):
                    if func(candidate.reshape(1, -1))[0] < fitness[top_idx]:
                        population[top_idx] = candidate
            
            # Check for stagnation and restart if needed
            population, fitness = self._restart_if_stagnant(population, fitness, func)
            
            # Update best after all modifications
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < f_opt:
                f_opt = fitness[best_idx]
                x_opt = population[best_idx].copy()
            
            gen += 1
        
        return f_opt, x_opt
