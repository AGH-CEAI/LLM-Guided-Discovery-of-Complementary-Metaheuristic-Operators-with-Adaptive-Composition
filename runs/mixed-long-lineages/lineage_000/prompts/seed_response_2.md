```python
import numpy as np

class AdaptiveCovarianceMemeticOptimizer:
    """
    A population-based optimizer combining simplified CMA-ES covariance adaptation
    with elitist selection and adaptive step-size control.
    
    Key components:
    - Covariance matrix adaptation (rank-1 + rank-mu updates)
    - Path cumulation for step-size control
    - Adaptive recombination weights
    - Diversity-based restarts
    """
    
    def __init__(self, dim, NP=None, initial_sigma=1.0):
        self.dim = dim
        self.NP = NP if NP is not None else min(max(4 * dim, 64), 300)
        self.initial_sigma = initial_sigma
        
        # Search bounds
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # CMA-ES specific parameters (simplified)
        self.learning_rate_cov = 2.0 / (dim**2 + 6.0)
        self.learning_rate_mean = 1.0 / 7.0
        self.cumulation_steps = 4.0 / (dim + 4.0)
        self.cumulation_sigma = 4.0 / (dim + 4.0)
        self.damping_sigma = 1.0 + max(0, (self.cumulation_sigma - 1) * dim**0.5)
        
        # Evolution paths
        self.pc = None  # For covariance
        self.ps = None  # For step-size
        
        # Covariance matrix
        self.C = None
        self.B = None  # Eigenvector basis
        self.D = None  # Eigenvalue scales
        
        # Step size
        self.sigma = initial_sigma
        
        # Mean vector
        self.mean = None
        self.mean_old = None
        
        # Recombination weights (negative weights allow worse solutions)
        n_half = self.NP // 2
        raw_weights = np.concatenate([np.arange(1, n_half + 1),
                                       np.arange(-n_half, 0)])
        self.weights = raw_weights / np.sum(np.abs(raw_weights))
        self.mu_eff = (np.sum(self.weights[:n_half])**2) / np.sum(self.weights[:n_half]**2)
        
        # State tracking
        self.generation = 0
        self.f_opt = np.inf
        self.x_opt = None
        self.fitness_history = []
        self.population = None
        
    def _initialize_population(self):
        """Initialize mean vector, covariance matrix, and evolution paths."""
        # Mean at center of search space
        self.mean = np.zeros(self.dim)
        self.mean_old = self.mean.copy()
        
        # Initialize covariance as identity
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        
        # Reset evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # Reset step size
        self.sigma = self.initial_sigma
        
        # Sample initial population from N(mean, sigma^2 * C)
        self.population = self._sample_population_batch()
        
    def _sample_population_batch(self):
        """Sample NP candidates from multivariate normal distribution."""
        # Generate samples in eigenspace: z ~ N(0, I), then transform
        z = np.random.randn(self.NP, self.dim)
        # Apply covariance scaling: y = mean + sigma * B * D * z
        samples = self.mean + self.sigma * (self.B * self.D) @ z.T.T
        return samples
    
    def _clip_to_bounds(self, population):
        """Hard clipping to search bounds [-100, 100]."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population):
        """Evaluate fitness for entire population in one batch call."""
        return self.func(population)
    
    def _update_mean(self, selected_population, selected_weights):
        """Update mean vector using weighted recombination of selected individuals."""
        self.mean_old = self.mean.copy()
        self.mean = np.sum(selected_population * selected_weights[:, np.newaxis], axis=0)
    
    def _update_evolution_paths(self):
        """Update cumulation paths for covariance and step-size adaptation."""
        # Normalized mean difference
        y = (self.mean - self.mean_old) / max(self.sigma, 1e-10)
        y_c = np.linalg.solve(self.B * self.D, y)
        
        # Update covariance path (pc)
        self.pc = ((1 - self.cumulation_steps) * self.pc + 
                   np.sqrt(self.cumulation_steps * (2 - self.cumulation_steps)) * y)
        
        # Update step-size path (ps)
        self.ps = ((1 - self.cumulation_sigma) * self.ps + 
                   np.sqrt(self.cumulation_sigma * (2 - self.cumulation_sigma)) * y_c)
    
    def _update_covariance(self):
        """Simplified CMA-ES covariance update: rank-1 + rank-mu terms."""
        # Rank-1 update: (pc * pc^T)
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update: weighted sum of (yi * yi^T)
        # Use current population deviations from old mean
        yi = (self.population - self.mean_old) / max(self.sigma, 1e-10)
        rank_mu = yi.T @ np.diag(self.weights) @ yi
        
        # Combine with old covariance (forgetting factor)
        self.C = ((1 - self.learning_rate_cov) * self.C + 
                  self.learning_rate_cov * (rank_one + 
                  self.covariance_forgetting * rank_mu))
        
        # Ensure symmetry
        self.C = (self.C + self.C.T) / 2
        
    def _update_step_size(self):
        """Adapt step size using evolution path length."""
        # Expected norm of N(0, I) distribution
        chi_n = self.dim**0.5 * (1.0 - 1.0/(4.0*self.dim) + 1.0/(21.0*self.dim**2))
        
        # Update sigma based on ps length
        self.sigma *= np.exp((self.cumulation_sigma / self.damping_sigma) * 
                             (np.linalg.norm(self.ps) / chi_n - 1))
        
        # Clip sigma to prevent extremes
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _eigen_decompose_covariance(self):
        """Eigen-decompose covariance matrix for efficient sampling."""
        # Ensure positive definiteness
        self.C = (self.C + self.C.T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.C)
        
        # Clamp eigenvalues to prevent numerical issues
        eigvals = np.maximum(eigvals, 1e-10)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigvals)[::-1]
        self.D = np.sqrt(eigvals[idx])
        self.B = eigvecs[:, idx]
        
    def _select_survivors_batch(self, fitness, trial_fitness):
        """Elitist selection: keep parent if better, otherwise take trial."""
        better_mask = trial_fitness < fitness
        self.population[better_mask] = self.trial_population[better_mask]
        return np.minimum(fitness, trial_fitness)
    
    def _compute_diversity(self):
        """Compute average pairwise Euclidean distance in population."""
        sum_sq = np.sum(self.population**2, axis=1)
        dist_sq = (-2 * self.population @ self.population.T + 
                   sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :])
        # Avoid negative values due to numerical precision
        dist_sq = np.maximum(dist_sq, 0)
        return np.mean(np.sqrt(dist_sq))
    
    def _adapt_weights(self, improvement_ratio):
        """Adapt recombination weights based on recent performance."""
        if improvement_ratio > 0.2:
            # Good progress: emphasize best individuals
            self.weights = np.clip(self.weights * 1.2, -2.0, 5.0)
        elif improvement_ratio < 0.0:
            # Regressed: flatten weights
            self.weights = np.clip(self.weights * 0.8, -2.0, 5.0)
        
        # Renormalize weights
        sum_pos = np.sum(self.weights[self.weights > 0])
        sum_neg = np.sum(np.abs(self.weights[self.weights < 0]))
        if sum_pos > 0 and sum_neg > 0:
            pos_mask = self.weights > 0
            self.weights[pos_mask] /= sum_pos
            self.weights[~pos_mask] /= -sum_neg
            self.weights = self.weights / np.sum(np.abs(self.weights))
    
    def _restart_if_stagnant(self, fitness_history, current_fitness):
        """Check for stagnation and trigger restart if needed."""
        if len(fitness_history) < 15:
            return False
        
        recent_best = np.min(fitness_history[-10:])
        older_best = np.min(fitness_history[-15:-10])
        
        # Stagnation: no improvement over 10 generations
        if recent_best >= older_best - abs(older_best) * 1e-6:
            return True
        
        # Loss of diversity
        diversity = self._compute_diversity()
        if diversity < self.dim * 0.01:
            return True
        
        # Step size too small relative to bounds
        if self.sigma < 1e-8:
            return True
        
        return False
    
    def _handle_budget_exhaustion(self, fitness, trials):
        """Handle cases where func returns fewer values than requested."""
        if len(fitness) < len(trials):
            actual_size = len(fitness)
            self.population = self.population[:actual_size]
            self.trial_population = self.trial_population[:actual_size]
            self.NP = actual_size
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """
        Run optimization.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self.func = func
        self.generation = 0
        self.fitness_history = []
        self.covariance_forgetting = min(1.0, (2.0 / self.NP) * 
                                          (self.mu_eff - 1.5) / (self.dim + 10))
        
        # Initialize
        self._initialize_population()
        fitness = self._evaluate_batch(self.population)
        
        # Handle immediate budget exhaustion
        if self._handle_budget_exhaustion(fitness, self.population):
            best_idx = np.argmin(fitness)
            return fitness[best_idx], self.population[best_idx].copy()
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        self.fitness_history.append(self.f_opt)
        
        # Main evolutionary loop
        while not stopping_condition():
            self.generation += 1
            
            # Select top half for recombination
            n_select = max(2, self.NP // 2)
            select_idx = np.argpartition(fitness, n_select)[:n_select]
            selected_pop = self.population[select_idx]
            selected_weights = self.weights[:n_select]
            
            # Update mean vector
            self._update_mean(selected_pop, selected_weights)
            
            # Update evolution paths
            self._update_evolution_paths()
            
            # Update covariance matrix
            self._update_covariance()
            
            # Eigen-decompose for efficient sampling
            self._eigen_decompose_covariance()
            
            # Update step size
            self._update_step_size()
            
            # Sample new population
            self.trial_population = self._sample_population_batch()
            self.trial_population = self._clip_to_bounds(self.trial_population)
            
            # Evaluate trial population
            trial_fitness = self._evaluate_batch(self.trial_population)
            
            # Check stopping condition immediately after evaluation
            if stopping_condition():
                break
            
            # Handle budget exhaustion
            if self._handle_budget_exhaustion(trial_fitness, self.trial_population):
                fitness = fitness[:len(trial_fitness)]
                break
            
            # Selection: elitist replacement
            prev_best_fitness = self.f_opt
            fitness = self._select_survivors_batch(fitness, trial_fitness)
            
            # Track best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = self.population[best_idx].copy()
            
            self.fitness_history.append(self.f_opt)
            
            # Adapt weights based on improvement
            improvement_ratio = (prev_best_fitness - self.f_opt) / max(abs(self.f_opt), 1e-10)
            self._adapt_weights(improvement_ratio)
            
            # Check for stagnation and restart if needed
            if self._restart_if_stagnant(self.fitness_history, self.f_opt):
                # Restart with fresh population but keep best solution
                best_fitness = self.f_opt.copy()
                best_solution = self.x_opt.copy()
                
                self._initialize_population()
                self.f_opt = best_fitness
                self.x_opt = best_solution
                
                fitness = self._evaluate_batch(self.population)
                
                # Handle budget exhaustion on restart
                if self._handle_budget_exhaustion(fitness, self.population):
                    break
                
                # Inject best solution into population
                inject_idx = np.argmin(fitness)
                self.population[inject_idx] = best_solution
                fitness[inject_idx] = best_fitness
                
                if stopping_condition():
                    break
            
            # Final stopping condition check
            if stopping_condition():
                break
        
        return self.f_opt, self.x_opt.copy()
```