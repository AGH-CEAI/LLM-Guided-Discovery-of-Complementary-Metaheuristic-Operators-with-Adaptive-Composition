import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    A population-based optimizer that combines CMA-ES-style covariance adaptation
    with a diverse population. Uses an archive of elite solutions to stabilize
    covariance updates and includes adaptive step-size control.
    
    Key differences from MirrorSpectralDE:
    - Covariance matrix adaptation (not spectral DE operators)
    - Archive-based guidance for mutations
    - Evolution path for step-size adaptation
    - Rank-based covariance update
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(10 * dim, 500)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # CMA-ES style parameters
        self.popsize = self.NP
        self.mu = max(1, self.NP // 4)  # Number of parents for recombination
        self.weights = self._compute_logistic_weights()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.lr_cov = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.lr_step = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.damp_step = 1.0 + max(0, (3.0 * dim ** - 0.5 + 1.0) - 1.0)
        
        # Initialize distribution parameters
        self.mean = None
        self.cov = None
        self.step_size = None
        self.evolution_path = None
        self.cov_path = None
        
        # Archive of good solutions for guidance
        self.archive = None
        self.archive_max_size = self.NP * 3
        
        # Best tracking
        self.best_fitness = np.inf
        self.best_solution = None
        
        # Stagnation detection
        self.stagnation_counter = 0
        self.stagnation_threshold = 50 + 10 * dim
        
    def _compute_logistic_weights(self):
        """Compute superlinear weights for weighted recombination."""
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = np.maximum(weights, 0.0)
        weights = weights / np.sum(weights)
        return weights
    
    def _initialize_distribution(self):
        """Initialize the search distribution at center of search space."""
        center = (self.lower + self.upper) / 2.0
        self.mean = np.zeros(self.dim) + center
        
        # Initial covariance: scaled identity matrix
        initial_variance = ((self.upper - self.lower) / 6.0) ** 2
        self.cov = np.eye(self.dim) * initial_variance
        
        # Initial step size
        self.step_size = (self.upper - self.lower) / 3.0
        
        # Evolution paths for adaptation
        self.evolution_path = np.zeros(self.dim)
        self.cov_path = np.zeros(self.dim)
    
    def _init_archive(self):
        """Initialize empty archive for storing good solutions."""
        self.archive = np.zeros((0, self.dim))
    
    def _sample_population_batch(self):
        """Sample entire population from current multivariate normal distribution."""
        samples = np.random.multivariate_normal(
            self.mean, self.cov, size=self.NP
        )
        return np.clip(samples, self.lower, self.upper)
    
    def _mutate_batch(self, population):
        """
        Generate batch of mutated candidates using covariance-guided perturbation.
        Combines population members with archive-guided directions.
        """
        # Sample indices for base individuals
        indices = np.random.randint(0, self.NP, size=self.NP)
        bases = population[indices]
        
        # Generate random directions from current distribution
        directions = np.random.multivariate_normal(
            np.zeros(self.dim), self.cov, size=self.NP
        )
        
        # Add archive guidance if available
        if len(self.archive) > self.mu:
            archive_indices = np.random.randint(0, len(self.archive), size=self.NP)
            archive_guides = self.archive[archive_indices]
            guide_weight = 0.3 * min(1.0, len(self.archive) / self.archive_max_size)
            
            # Blend archive guidance into directions
            archive_diffs = archive_guides - bases
            directions = (1 - guide_weight) * directions + guide_weight * archive_diffs
        
        # Apply scaled perturbation
        mutants = bases + self.step_size * 0.5 * directions
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """
        Perform simulated binary crossover (SBX) between population and mutants.
        Returns trial candidates.
        """
        trials = np.empty_like(population)
        cross_rate = 0.5
        
        for i in range(self.NP):
            if np.random.random() < cross_rate:
                trials[i] = mutants[i]
            else:
                trials[i] = population[i]
        
        return trials
    
    def _clip_to_bounds(self, candidates):
        """Ensure all candidates are within search bounds."""
        return np.clip(candidates, self.lower, self.upper)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Select survivors based on fitness comparison.
        Returns new population and fitness values.
        """
        better_mask = trial_fitness < fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        return new_population, new_fitness
    
    def _update_distribution(self, population, fitness, trials, trial_fitness):
        """
        Update mean, step size, and covariance matrix based on successful mutations.
        Uses weighted recombination and rank-based covariance update.
        """
        # Identify successful mutations
        improvements = trial_fitness < fitness
        num_improved = np.sum(improvements)
        
        if num_improved == 0:
            self.stagnation_counter += 1
            return
        
        self.stagnation_counter = 0
        
        # Compute weighted mean shift
        selected = trials[improvements]
        selected_fitness = trial_fitness[improvements]
        
        if len(selected) >= self.mu:
            # Sort by fitness and compute weighted mean
            sorted_indices = np.argsort(selected_fitness)
            sorted_selected = selected[sorted_indices]
            
            # Weighted recombination of selected individuals
            selected_means = np.sum(sorted_selected[:self.mu] * self.weights[:, np.newaxis], axis=0)
            
            # Update mean
            old_mean = self.mean.copy()
            self.mean = selected_means
            
            # Compute shift for covariance update
            mean_shift = self.mean - old_mean
        else:
            # Use all improvements with equal weight
            mean_shift = np.mean(selected, axis=0) - self.mean
            self.mean = np.mean(selected, axis=0)
        
        # Update evolution path for step size
        self.evolution_path = (
            (1 - 1.0 / self.damp_step) * self.evolution_path +
            np.sqrt(2.0 / self.damp_step) * mean_shift / self.step_size
        )
        
        # Adapt step size using evolution path length
        path_norm = np.linalg.norm(self.evolution_path)
        expected_norm = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
        self.step_size *= np.exp((path_norm / expected_norm - 1.0) * self.lr_step / self.damp_step)
        
        # Ensure step size is within reasonable bounds
        self.step_size = np.clip(self.step_size, 1e-10, (self.upper - self.lower) / 2.0)
        
        # Rank-based covariance update
        if num_improved >= self.mu:
            # Use improvement vectors for covariance update
            sorted_indices = np.argsort(trial_fitness[improvements])
            improved_sorted = trials[improvements][sorted_indices]
            
            # Compute weighted covariance update
            diff_matrix = improved_sorted[:self.mu] - old_mean
            weighted_cov = np.zeros((self.dim, self.dim))
            for k in range(min(self.mu, len(diff_matrix))):
                v = diff_matrix[k]
                weighted_cov += self.weights[k] * np.outer(v, v)
            
            # Add rank-1 update from mean shift
            rank_one_update = np.outer(mean_shift, mean_shift)
            
            # Update covariance with damping
            self.cov = (
                (1 - self.lr_cov) * self.cov +
                self.lr_cov * rank_one_update / self.step_size ** 2 +
                1e-8 * np.eye(self.dim)
            )
            
            # Ensure covariance remains positive definite
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def _update_archive(self, population, fitness):
        """
        Update the archive with best solutions found.
        Maintains diversity by keeping solutions from different regions.
        """
        # Add current population to archive
        new_archive = np.vstack([self.archive, population])
        new_fitness = np.concatenate([
            np.full(len(self.archive), -np.inf),  # Archive fitness unknown
            fitness
        ])
        
        # Keep only best solutions up to max size
        if len(new_archive) > self.archive_max_size:
            # Select based on fitness
            keep_indices = np.argsort(new_fitness)[:self.archive_max_size]
            self.archive = new_archive[keep_indices]
        else:
            self.archive = new_archive
    
    def _check_stagnation(self):
        """Check if the search has stagnated."""
        return self.stagnation_counter > self.stagnation_threshold
    
    def _restart_optimizer(self):
        """Restart the optimizer when stagnation is detected."""
        self._initialize_distribution()
        self.archive = np.zeros((0, self.dim))
        self.stagnation_counter = 0
        self.step_size = (self.upper - self.lower) / 3.0
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Parameters:
            func: callable that accepts (N, dim) array and returns (N,) fitness
            stopping_condition: callable that returns True when to stop
            
        Returns:
            (f_opt, x_opt): best fitness and corresponding solution
        """
        dim = self.dim
        
        # Initialize
        self._initialize_distribution()
        self._init_archive()
        
        # Initial population
        population = self._sample_population_batch()
        fitness = func(population)
        
        # Handle budget exhaustion at start
        if len(fitness) < self.NP:
            valid_mask = ~np.isnan(fitness)
            population = population[valid_mask]
            fitness = fitness[valid_mask]
            if len(fitness) == 0:
                return self.best_fitness, self.best_solution
        
        # Track best
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = population[best_idx].copy()
        
        # Main evolution loop
        while not stopping_condition():
            # Generate trial candidates
            mutants = self._mutate_batch(population)
            trials = self._crossover_batch(population, mutants)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < self.NP:
                valid_mask = ~np.isnan(trial_fitness)[:self.NP]
                trials = trials[valid_mask]
                trial_fitness = trial_fitness[valid_mask]
                if len(trial_fitness) == 0:
                    break
                
                # Evaluate only valid trials against corresponding population members
                pop_subset = population[valid_mask]
                fit_subset = fitness[valid_mask]
                pop_subset, fit_subset = self._select_survivors_batch(
                    pop_subset, fit_subset, trials, trial_fitness
                )
                population[valid_mask] = pop_subset
                fitness[valid_mask] = fit_subset
            else:
                # Normal selection
                population, fitness = self._select_survivors_batch(
                    population, fitness, trials, trial_fitness
                )
            
            # Check stopping condition after selection
            if stopping_condition():
                break
            
            # Update distribution parameters
            self._update_distribution(population, fitness, trials, trial_fitness)
            
            # Update archive
            self._update_archive(population, fitness)
            
            # Check for stagnation and restart if needed
            if self._check_stagnation():
                self._restart_optimizer()
                population = self._sample_population_batch()
                fitness = func(population)
                if len(fitness) < self.NP:
                    break
            
            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = population[best_idx].copy()
                self.stagnation_counter = 0  # Reset on improvement
        
        return self.best_fitness, self.best_solution
