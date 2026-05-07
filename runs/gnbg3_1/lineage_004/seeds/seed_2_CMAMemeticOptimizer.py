import numpy as np


class CMAMemeticOptimizer:
    """
    Covariance Matrix Adaptation Memetic Optimizer.
    Combines CMA-ES style covariance adaptation with memetic local search
    and a ring-based population topology for information sharing.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(200, max(4 * dim, 8 * dim))  # Population size
        self.bounds = np.array([-100.0, 100.0])
        self.lb = self.bounds[0]
        self.ub = self.bounds[1]
        
        # Step size and covariance parameters
        self.step_size = 0.5 * (self.ub - self.lb) / np.sqrt(self.dim)
        self.initial_step = self.step_size
        
        # CMA-ES style parameters
        self.p_c = np.zeros(dim)  # Evolution path for covariance
        self.p_s = np.zeros(dim)  # Evolution path for step size
        self.cov = np.eye(dim)    # Covariance matrix
        
        # Adaptation rates
        self.c_c = 2.0 / (dim + 2.0)
        self.c_s = 2.0 / (dim + 2.0)
        self.c_1 = 1.0 / (dim ** 2 + 6.0)
        self.c_mu = 1.0 / (dim ** 2 + 6.0)
        self.c_succ = 0.11 if dim > 15 else 0.25
        
        # Memetic parameters
        self.local_search_prob = 0.15
        self.local_search_radius = 0.1 * self.step_size
        
        # Diversity tracking
        self.convergence_threshold = 1e-8
        self.stagnation_counter = 0
        self.max_stagnation = 50
        self.best_fitness_history = []
        
        # Population state
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.f_opt = np.inf
        self.x_opt = None
        
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling for better coverage."""
        self.population = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        # LHS-style perturbation for better spread
        for d in range(self.dim):
            points = np.linspace(self.lb, self.ub, self.NP + 1)
            offsets = np.random.uniform(0, (self.ub - self.lb) / self.NP, self.NP)
            self.population[:, d] = points[:-1] + offsets
        
        self.population = np.clip(self.population, self.lb, self.ub)
        self.fitness = func(self.population)
        
        self.best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_idx]
        self.x_opt = self.population[self.best_idx].copy()
        
    def _initialize_covariance_matrix(self):
        """Initialize covariance matrix to diagonal with appropriate scaling."""
        self.cov = np.eye(self.dim)
        self.step_size = self.initial_step
        self.p_c = np.zeros(self.dim)
        self.p_s = np.zeros(self.dim)
        
    def _compute_diversity(self):
        """Compute population diversity using average pairwise distance."""
        if self.population is None or len(self.population) < 2:
            return 0.0
        
        mean_pop = np.mean(self.population, axis=0)
        variance = np.mean(np.sum((self.population - mean_pop) ** 2, axis=1))
        return np.sqrt(variance)
    
    def _sample_trials_batch(self, population, n_trials=None):
        """Sample trial individuals using current covariance matrix and step size."""
        if n_trials is None:
            n_trials = self.NP
            
        # Decompose covariance matrix using Cholesky
        L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(self.dim))
        
        # Generate random directions
        z = np.random.randn(n_trials, self.dim)
        
        # Apply covariance transformation
        y = z @ L.T
        
        # Scale by step size and select mean from distribution
        # Use weighted combination of best individual and population mean
        weights = np.ones(self.NP) / self.NP
        weights[self.best_idx] = 0.3
        weights = weights / weights.sum()
        mean_vector = population.T @ weights
        
        # Create trial population
        trials = mean_vector + self.step_size * y
        
        # Add ring-topology influence (neighbor information)
        ring_influence = np.random.rand(n_trials, 1) * 0.1 * self.step_size
        neighbor_idx = (np.arange(self.NP) + 1) % self.NP
        trials += ring_influence * (population[neighbor_idx[:n_trials]] - mean_vector)
        
        return np.clip(trials, self.lb, self.ub)
    
    def _mutate_batch(self, population, fitness):
        """Apply differential mutation using ring topology with covariance scaling."""
        n = len(population)
        
        # Get indices for ring topology
        idx = np.arange(n)
        idx_r1 = (idx + 1) % n
        idx_r2 = (idx + 2) % n
        idx_r3 = (idx + 3) % n
        
        # Select base individuals (prefer good ones)
        base_probs = np.exp(-fitness / (np.mean(fitness) + 1e-10))
        base_probs /= base_probs.sum()
        base_idx = np.random.choice(n, size=n, p=base_probs)
        
        # Differential vectors with covariance scaling
        L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(self.dim))
        scale = self.step_size / np.sqrt(self.dim)
        
        diffs = []
        for i in range(n):
            d1 = population[idx_r1[i]] - population[idx_r2[i]]
            d2 = population[idx_r3[i]] - population[idx_r1[i]]
            diff = d1 + 0.5 * d2
            diffs.append(diff)
        diffs = np.array(diffs)
        
        # Scale differential vectors
        mutants = population[base_idx] + scale * diffs
        
        # Apply local search to a subset (memetic component)
        local_search_mask = np.random.rand(n) < self.local_search_prob
        if local_search_mask.any():
            ls_radius = self.local_search_radius * np.random.randn(local_search_mask.sum(), self.dim)
            mutants[local_search_mask] += ls_radius[local_search_mask.sum() > 0 if local_search_mask.sum() > 0 else True]
        
        return np.clip(mutants, self.lb, self.ub)
    
    def _crossover_batch(self, target_pop, mutant_pop, cr=0.7):
        """Binomial crossover with dimension-aware adaptation."""
        n = len(target_pop)
        
        # Adaptive crossover rate based on diversity
        diversity = self._compute_diversity()
        adaptive_cr = np.clip(cr + 0.1 * (1 - diversity / (self.ub - self.lb)), 0.3, 0.95)
        
        # Random crossover mask
        mask = np.random.rand(n, self.dim) < adaptive_cr
        
        # Ensure at least one dimension from mutant
        enforce_dim = np.random.randint(0, self.dim, size=n)
        mask[np.arange(n), enforce_dim] = True
        
        trials = np.where(mask, mutant_pop, target_pop)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors using (μ/μ_I, λ) style selection with weighted combination."""
        n = len(population)
        
        # Combine parent and trial populations
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        
        # Compute weights based on fitness (lower is better)
        sorted_idx = np.argsort(combined_fit)
        
        # Weighted combination of best individuals
        weights = np.zeros(len(combined_fit))
        mu_eff = min(n, 10)
        for i in range(min(mu_eff, len(sorted_idx))):
            weights[sorted_idx[i]] = 1.0 / mu_eff
        
        # Select new population using weighted sampling
        weight_probs = np.abs(weights) / np.sum(np.abs(weights))
        selected_idx = np.random.choice(len(combined_pop), size=n, p=weight_probs, replace=True)
        
        new_population = combined_pop[selected_idx]
        new_fitness = combined_fit[selected_idx]
        
        # Track improvements
        current_best = np.min(new_fitness)
        if current_best < self.f_opt:
            improvement = self.f_opt - current_best
            self.f_opt = current_best
            self.best_idx = np.argmin(new_fitness)
            self.x_opt = new_population[self.best_idx].copy()
            self.stagnation_counter = 0
            return new_population, new_fitness, improvement
        
        self.stagnation_counter += 1
        return new_population, new_fitness, 0.0
    
    def _adapt_step_size(self, improvement):
        """Adapt step size based on improvement history."""
        self.best_fitness_history.append(self.f_opt)
        
        if len(self.best_fitness_history) > 5:
            recent_improvements = np.diff(self.best_fitness_history[-6:])
            avg_improvement = np.mean(np.abs(recent_improvements))
            
            # Adapt step size: increase if making progress, decrease if stagnating
            if improvement > 1e-10:
                self.step_size *= 1.05
            elif self.stagnation_counter > 3:
                self.step_size *= 0.9
                
            # Bound step size
            max_step = (self.ub - self.lb) * 0.5
            min_step = 1e-6
            self.step_size = np.clip(self.step_size, min_step, max_step)
            
    def _adapt_covariance(self, population, fitness):
        """CMA-ES style covariance adaptation with weighted recombination."""
        # Compute weighted mean
        weights = np.exp(-fitness / (np.mean(fitness) + 1e-10))
        weights /= weights.sum()
        mean = population.T @ weights
        
        # Update evolution paths
        y = mean - self.x_opt
        z = y / (self.step_size + 1e-10)
        
        # Cumulation for covariance
        self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c)) * z
        
        # Update covariance matrix
        rank_one = np.outer(self.p_c, self.p_c)
        
        # Rank-μ update using weighted individuals
        centered_pop = population - self.x_opt
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(len(population)):
            w = weights[i] if fitness[i] < np.median(fitness) else -weights[i] * 0.5
            y_i = centered_pop[i] / (self.step_size + 1e-10)
            rank_mu += w * np.outer(y_i, y_i)
        
        # Combine updates
        self.cov = (1 - self.c_1 - self.c_mu) * self.cov + self.c_1 * rank_one + self.c_mu * rank_mu
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.cov))
        if min_eig < 1e-10:
            self.cov += (1e-9 - min_eig) * np.eye(self.dim)
        
    def _restart_if_stagnant(self):
        """Restart population if stagnating to escape local optima."""
        if self.stagnation_counter >= self.max_stagnation:
            # Keep best individual
            best = self.x_opt.copy()
            
            # Reinitialize population around best
            self.population = np.random.uniform(
                best - 0.5 * self.step_size,
                best + 0.5 * self.step_size,
                (self.NP, self.dim)
            )
            self.population = np.clip(self.population, self.lb, self.ub)
            self.population[0] = best
            
            # Reset covariance
            self._initialize_covariance_matrix()
            self.stagnation_counter = 0
            return True
        return False
    
    def _update_local_search_radius(self):
        """Adapt local search radius based on progress."""
        if self.stagnation_counter > 10:
            self.local_search_radius *= 1.2
        elif self.stagnation_counter == 0:
            self.local_search_radius *= 0.95
        
        max_ls_radius = self.step_size * 2.0
        min_ls_radius = 1e-6
        self.local_search_radius = np.clip(self.local_search_radius, min_ls_radius, max_ls_radius)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        self._initialize_population(func)
        self._initialize_covariance_matrix()
        
        generation = 0
        
        while not stopping_condition():
            # Generate trial population using CMA sampling
            trials_cma = self._sample_trials_batch(self.population)
            
            # Generate trial population using differential mutation
            mutants = self._mutate_batch(self.population, self.fitness)
            
            # Crossover for both strategies
            trials_de = self._crossover_batch(self.population, mutants)
            
            # Combine strategies: 70% CMA, 30% DE
            n_cma = int(0.7 * self.NP)
            trials = np.vstack([trials_cma[:n_cma], trials_de[n_cma:]])
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle truncated/NaN returns
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if stopping_condition():
                    break
                
            # Check for all NaN
            if np.all(np.isnan(trial_fitness)):
                break
            
            # Selection
            self.population, self.fitness, improvement = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            # Adaptations
            self._adapt_step_size(improvement)
            self._adapt_covariance(self.population, self.fitness)
            self._update_local_search_radius()
            
            # Check for stagnation and restart
            self._restart_if_stagnant()
            
            generation += 1
            
            # Safety check
            if generation > 10000:
                break
        
        return self.f_opt, self.x_opt
