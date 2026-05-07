import numpy as np


class AdaptiveMemeticDifferentialEvolution:
    """
    Hybrid DE + Memetic (local search) optimizer with adaptive covariance-guided mutation.
    Key features:
    - DE/rand/1 mutation with adaptive scale
    - CMA-ES-inspired covariance ellipsoid for directional mutation
    - Nelder-Mead local search on archive of best individuals
    - Adaptive population reinitialization on stagnation
    - Geometric semantic crossover for offspring diversity
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population size: 6*dim is a good balance
        self.NP = min(max(6 * dim, 30), 300)
        
        # Adaptive parameters
        self.F = 0.5  # Differential weight
        self.CR = 0.7  # Crossover rate
        self.fs = np.ones(self.NP)  # Per-individual scale factors
        self.crs = np.ones(self.NP) * self.CR  # Per-individual CR
        
        # Covariance adaptation state
        self.cov_matrix = np.eye(dim)
        self.cov_learning_rate = 0.01
        self.p_c = np.zeros(dim)
        self.sigma = 0.5
        self.sigma_adapt_rate = 0.1
        
        # Local search state
        self.local_search_frequency = 5  # Apply LS every N generations
        self.local_search_budget = 0.1  # Fraction of budget for LS
        self.nm_alpha = 1.0
        self.nm_gamma = 2.0
        self.nm_rho = 0.5
        self.nm_sigma = 0.5
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 20
        self.success_history = []
        self.archive_size = min(10, self.NP // 3)
        
        # Internal state
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.best_fitness = np.inf
        self.generation = 0
        self.func_eval_count = 0
        self.total_func_calls = 0
        
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.NP, self.dim)
        )
        self.population = np.clip(self.population, self.lower, self.upper)
        
    def _evaluate_batch(self, pop):
        """Evaluate entire population batch."""
        fitness = self.func(pop)
        self.func_eval_count += len(fitness)
        return fitness
    
    def _update_best(self, fitness):
        """Track global best and best index."""
        current_best_idx = np.argmin(fitness)
        current_best = fitness[current_best_idx]
        
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            self.best_idx = current_best_idx
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
    def _mutate_batch(self, indices):
        """
        Generate mutant vectors using DE/rand/1 with covariance guidance.
        Returns mutant vectors of shape (len(indices), dim).
        """
        n_trials = len(indices)
        pop = self.population[indices]
        
        # Select 3 distinct random indices for rand/1 mutation
        r1 = np.random.randint(0, len(indices), size=n_trials)
        r2 = np.random.randint(0, len(indices), size=n_trials)
        r3 = np.random.randint(0, len(indices), size=n_trials)
        
        # Ensure distinctness
        for i in range(n_trials):
            while r2[i] == r1[i]:
                r2[i] = np.random.randint(0, len(indices))
            while r3[i] == r1[i] or r3[i] == r2[i]:
                r3[i] = np.random.randint(0, len(indices))
        
        # Base vector
        base = pop[r1]
        
        # Difference vectors
        diff1 = pop[r2] - pop[r3]
        
        # Apply covariance-guided scaling occasionally
        if self.generation > 5 and np.random.rand() < 0.3:
            # Sample from covariance ellipsoid
            L = np.linalg.cholesky(self.cov_matrix + 1e-8 * np.eye(self.dim))
            z = np.random.randn(n_trials, self.dim)
            cov_perturbation = (z @ L.T) * self.sigma * 0.1
            diff1 = diff1 + cov_perturbation
        
        # Scale factor with per-individual adaptation
        F_batch = self.fs[indices] if len(indices) == len(self.fs) else np.tile(self.fs, n_trials // self.NP + 1)[:n_trials]
        
        mutants = base + F_batch[:, np.newaxis] * diff1
        return mutants
    
    def _crossover_batch(self, targets, mutants, cr_batch=None):
        """
        Binomial crossover between targets and mutants.
        Returns trial vectors of shape (n_trials, dim).
        """
        n_trials = mutants.shape[0]
        
        if cr_batch is None:
            cr_batch = self.crs[:n_trials] if n_trials <= len(self.crs) else np.tile(self.crs, n_trials // len(self.crs) + 1)[:n_trials]
        
        # Random crossover mask
        mask = np.random.rand(n_trials, self.dim) < cr_batch[:, np.newaxis]
        
        # Ensure at least one dimension comes from mutant
        enforce_idx = np.random.randint(0, self.dim, size=n_trials)
        mask[np.arange(n_trials), enforce_idx] = True
        
        trials = np.where(mask, mutants, targets)
        return trials
    
    def _clip_to_bounds(self, pop):
        """Clip population to search bounds."""
        return np.clip(pop, self.lower, self.upper)
    
    def _apply_nelder_mead_batch(self, candidates, candidate_fitness, max_evals):
        """
        Apply Nelder-Mead local search to top candidates.
        Returns refined candidates and fitness.
        """
        n_ls = min(3, len(candidates))
        if n_ls == 0:
            return candidates, candidate_fitness
        
        top_indices = np.argsort(candidate_fitness)[:n_ls]
        refined = np.copy(candidates[top_indices])
        refined_fit = np.copy(candidate_fitness[top_indices])
        
        ls_evals = 0
        max_ls_evals = int(max_evals)
        
        for i in range(n_ls):
            if ls_evals >= max_ls_evals:
                break
                
            # Initialize simplex around candidate
            x_best = refined[i]
            simplex = np.tile(x_best, (self.dim + 1, 1))
            
            # Random simplex initialization
            for j in range(self.dim + 1):
                if j == 0:
                    continue
                simplex[j] = x_best + np.random.randn(self.dim) * self.nm_sigma * 0.1
            simplex = np.clip(simplex, self.lower, self.upper)
            
            # Evaluate simplex
            simplex_fit = self.func(simplex)
            ls_evals += len(simplex_fit)
            
            # Nelder-Mead iterations
            for _ in range(20):  # Max 20 NM iterations per candidate
                if ls_evals >= max_ls_evals:
                    break
                    
                # Sort by fitness
                order = np.argsort(simplex_fit)
                simplex = simplex[order]
                simplex_fit = simplex_fit[order]
                
                # Compute centroid (excluding worst)
                x_o = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_o + self.nm_alpha * (x_o - simplex[-1])
                x_r = np.clip(x_r, self.lower, self.upper)
                fit_r = self.func(x_r.reshape(1, -1))[0]
                ls_evals += 1
                
                if simplex_fit[0] <= fit_r < simplex_fit[-2]:
                    simplex[-1] = x_r
                    simplex_fit[-1] = fit_r
                elif fit_r < simplex_fit[0]:
                    # Expansion
                    x_e = x_o + self.nm_gamma * (x_r - x_o)
                    x_e = np.clip(x_e, self.lower, self.upper)
                    fit_e = self.func(x_e.reshape(1, -1))[0]
                    ls_evals += 1
                    
                    if fit_e < fit_r:
                        simplex[-1] = x_e
                        simplex_fit[-1] = fit_e
                    else:
                        simplex[-1] = x_r
                        simplex_fit[-1] = fit_r
                else:
                    # Contraction
                    x_c = x_o + self.nm_rho * (simplex[-1] - x_o)
                    x_c = np.clip(x_c, self.lower, self.upper)
                    fit_c = self.func(x_c.reshape(1, -1))[0]
                    ls_evals += 1
                    
                    if fit_c < simplex_fit[-1]:
                        simplex[-1] = x_c
                        simplex_fit[-1] = fit_c
                    else:
                        # Shrink
                        simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                        if ls_evals < max_ls_evals:
                            shrink_evals = min(self.dim, max_ls_evals - ls_evals)
                            for j in range(1, 1 + shrink_evals):
                                simplex[j] = np.clip(simplex[j], self.lower, self.upper)
                            if shrink_evals > 0:
                                shrink_fit = self.func(simplex[1:1 + shrink_evals])
                                simplex_fit[1:1 + shrink_evals] = shrink_fit
                                ls_evals += len(shrink_fit)
            
            # Update best
            if simplex_fit[0] < refined_fit[i]:
                refined[i] = simplex[0]
                refined_fit[i] = simplex_fit[0]
        
        return refined, refined_fit
    
    def _adapt_step_size(self, trial_fitness, target_fitness):
        """
        Adapt scale factors based on success rates.
        Uses a simple success-history based adaptation.
        """
        improvements = trial_fitness < target_fitness
        
        if len(self.success_history) > 5:
            self.success_history.pop(0)
        self.success_history.append(improvements.mean())
        
        # Adaptation rate
        if len(self.success_history) >= 5:
            success_rate = np.mean(self.success_history[-5:])
            
            if success_rate > 0.2:
                # Increase F slightly if success rate is good
                self.F = min(1.0, self.F * 1.1)
            elif success_rate < 0.1:
                # Decrease F if success rate is poor
                self.F = max(0.1, self.F * 0.9)
        
        # Per-individual F adaptation
        for i in range(min(len(improvements), len(self.fs))):
            if improvements[i]:
                self.fs[i] = min(1.5, self.fs[i] * 1.1)
            else:
                self.fs[i] = max(0.1, self.fs[i] * 0.95)
        
        # Per-individual CR adaptation
        for i in range(min(len(improvements), len(self.crs))):
            if improvements[i]:
                self.crs[i] = min(1.0, self.crs[i] * 1.05)
            else:
                self.crs[i] = max(0.0, self.crs[i] * 0.95)
    
    def _adapt_covariance(self, improvements, trial_pop, target_pop):
        """
        Adapt covariance matrix using evolution strategies principles.
        Updates the mutation ellipsoid based on successful directions.
        """
        if not improvements.any():
            return
            
        # Successful directions
        successful = trial_pop[improvements] - target_pop[improvements]
        
        if len(successful) > 0:
            # Update covariance based on successful mutation directions
            cov_update = np.zeros_like(self.cov_matrix)
            for d in successful:
                cov_update += np.outer(d, d)
            cov_update /= max(1, len(successful))
            
            # CMA-ES style update
            self.cov_matrix = (1 - self.cov_learning_rate) * self.cov_matrix + \
                             self.cov_learning_rate * cov_update
            
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(self.cov_matrix)
            if np.any(eigvals <= 0):
                self.cov_matrix += (abs(min(eigvals)) + 1e-8) * np.eye(self.dim)
            
            # Adapt sigma
            if len(successful) > 0:
                avg_norm = np.mean(np.linalg.norm(successful, axis=1))
                if avg_norm < self.sigma * 0.5:
                    self.sigma *= 0.9
                elif avg_norm > self.sigma * 2:
                    self.sigma *= 1.1
                self.sigma = np.clip(self.sigma, 0.01, 2.0)
    
    def _compute_diversity(self):
        """Compute population diversity using average pairwise distance."""
        if self.population is None or len(self.population) < 2:
            return 1.0
        
        # Sample for efficiency
        sample_size = min(50, len(self.population))
        indices = np.random.choice(len(self.population), sample_size, replace=False)
        sample = self.population[indices]
        
        # Pairwise distances
        diffs = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        
        # Exclude diagonal
        mask = ~np.eye(sample_size, dtype=bool)
        avg_distance = distances[mask].mean()
        
        return avg_distance / (self.upper - self.lower)
    
    def _restart_if_stagnant(self):
        """Reinitialize part of population if stagnant."""
        if self.stagnation_counter < self.max_stagnation:
            return False
            
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold or self.stagnation_counter >= 2 * self.max_stagnation:
            # Partial reinitialization - replace worst half
            n_replace = self.NP // 2
            replace_indices = np.argsort(self.fitness)[-n_replace:]
            
            new_part = np.random.uniform(
                self.lower, self.upper, size=(n_replace, self.dim)
            )
            
            # Inject some diversity from best
            best_idx = np.argmin(self.fitness)
            for i in range(n_replace):
                new_part[i] = self.population[best_idx] + np.random.randn(self.dim) * 10.0
            
            self.population[replace_indices] = np.clip(new_part, self.lower, self.upper)
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            self.stagnation_counter = 0
            
            # Reset adaptation parameters
            self.F = 0.5
            self.sigma = 0.5
            self.cov_matrix = np.eye(self.dim) * (self.sigma ** 2)
            
            return True
        return False
    
    def _select_survivors_batch(self, targets, target_fit, trials, trial_fit):
        """
        Select survivors using elitist (mu + lambda) strategy.
        Returns new population and fitness.
        """
        # Combine population and trials
        combined = np.vstack([targets, trials])
        combined_fit = np.concatenate([target_fit, trial_fit])
        
        # Select best NP individuals
        select_indices = np.argsort(combined_fit)[:self.NP]
        
        new_pop = combined[select_indices]
        new_fit = combined_fit[select_indices]
        
        return new_pop, new_fit
    
    def _generate_archive(self, fitness, top_k):
        """Generate archive of top-k individuals for local search."""
        top_indices = np.argsort(fitness)[:top_k]
        return self.population[top_indices], fitness[top_indices]
    
    def __call__(self, func, stopping_condition):
        """
        Main optimization loop.
        Returns (f_opt, x_opt).
        """
        self.func = func
        self.generation = 0
        self.func_eval_count = 0
        self.best_fitness = np.inf
        self.stagnation_counter = 0
        self.success_history = []
        
        # Initialize
        self._initialize_population()
        self.fitness = self._evaluate_batch(self.population)
        
        # Initial best
        self._update_best(self.fitness)
        
        # Main loop
        while not stopping_condition():
            self.generation += 1
            
            # Check budget after population eval
            if len(self.func(self.population[:1])) == 0:
                break
            
            # Generate trials using batched mutation and crossover
            all_indices = np.arange(self.NP)
            mutants = self._mutate_batch(all_indices)
            mutants = self._clip_to_bounds(mutants)
            
            trials = self._crossover_batch(self.population, mutants)
            trials = self._clip_to_bounds(trials)
            
            # Check budget before trial evaluation
            if len(trials) == 0 or stopping_condition():
                break
                
            # Evaluate trials
            trial_fit = self._evaluate_batch(trials)
            
            # Handle budget exhaustion
            if len(trial_fit) < len(trials):
                trials = trials[:len(trial_fit)]
                trial_fit = trial_fit[:len(trial_fit)]
                if len(trial_fit) == 0:
                    break
                    
            if stopping_condition():
                break
            
            # Track improvements for adaptation
            improvements = trial_fit < self.fitness
            
            # Adapt parameters
            self._adapt_step_size(trial_fit, self.fitness)
            self._adapt_covariance(improvements, trials, self.population)
            
            # Select survivors
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fit
            )
            
            # Update best
            self._update_best(self.fitness)
            
            # Apply local search periodically
            if self.generation % self.local_search_frequency == 0:
                if self._compute_diversity() > 0.05:
                    archive, archive_fit = self._generate_archive(self.fitness, self.archive_size)
                    
                    # Estimate local search budget
                    ls_budget = max(10, self.func_eval_count * 0.01)
                    
                    refined, refined_fit = self._apply_nelder_mead_batch(
                        archive, archive_fit, ls_budget
                    )
                    
                    # Integrate refined individuals
                    for i in range(len(refined)):
                        # Find worst individual to potentially replace
                        worst_idx = np.argmax(self.fitness)
                        if refined_fit[i] < self.fitness[worst_idx]:
                            self.population[worst_idx] = refined[i]
                            self.fitness[worst_idx] = refined_fit[i]
                    
                    self._update_best(self.fitness)
            
            # Check for stagnation and restart if needed
            self._restart_if_stagnant()
            
            # Periodic covariance reset to prevent degeneration
            if self.generation % 50 == 0:
                self.cov_matrix = np.eye(self.dim) * (self.sigma ** 2)
        
        # Return best
        best_local_idx = np.argmin(self.fitness)
        return self.fitness[best_local_idx], self.population[best_local_idx].copy()
