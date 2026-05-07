```python
import numpy as np


class AdaptiveCovarianceEvolution:
    """
    Adaptive Covariance Evolution (ACE) optimizer.
    
    A CMA-ES-inspired population-based optimizer with:
    - Island-based population topology for diversity
    - Rank-based covariance adaptation
    - Adaptive step size control
    - Diversity-triggered restarts
    
    Key methods (all batched over population):
    - _sample_trials_batch: Generate trial population using CMA-style sampling
    - _adapt_covariance_variant_01: Update covariance from successful mutations
    - _adapt_step_size: Control search step size based on success rate
    - _compute_diversity: Monitor population spread for restart detection
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(10 * dim, 400)  # Population size, cap at 400
        self.bounds = np.array([-100.0, 100.0])
        
        # CMA-ES-like state variables
        self.mean = None
        self.cov = None
        self.step_size = None
        self.evolution_path = None
        self.conjugate_path = None
        
        # Adaptation parameters
        self.mu_eff = None  # Effective number of parents
        self.mu_cov = None  # Covariance update weight count
        self.cc = None      # Evolution path cumulation factor
        self.cs = None      # Step size cumulation factor
        self.c1 = None      # Rank-1 covariance update rate
        self.cmu = None     # Rank-mu covariance update rate
        self.damps = None   # Step size damping
        
        # Tracking variables
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        self.successful_mutations = 0
        self.total_mutations = 0
        self.stagnation_counter = 0
        
        # Island topology: split population into subpopulations
        self.n_islands = max(2, min(8, self.np // 20))
        self.island_sizes = None
        self.island_offsets = None
        
    def _initialize_parameters(self):
        """Initialize CMA-ES control parameters based on dimension."""
        # Population parameters
        self.mu_eff = self.np / 10.0
        self.mu_cov = self.np / 10.0
        
        # Cumulation factors
        self.cc = (4.0 + self.mu_eff / self.dim) / (self.dim + 4.0 + 2.0 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 5.0)
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt(self.mu_eff) - 1.0) + self.cs
        
        # Covariance update rates
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / 
                       ((self.dim + 2) ** 2 + 2.0 * self.mu_eff / 2.0))
        
    def _initialize_population(self):
        """Initialize population, mean, covariance, and step size."""
        self._initialize_parameters()
        
        # Initialize mean to center of search space
        self.mean = np.zeros(self.dim)
        
        # Initialize covariance as diagonal with large variance
        self.cov = np.eye(self.dim) * (self.bounds[1] - self.bounds[0]) ** 2 / 16.0
        
        # Initialize step size
        self.step_size = (self.bounds[1] - self.bounds[0]) / 5.0
        
        # Initialize evolution paths
        self.evolution_path = np.zeros(self.dim)
        self.conjugate_path = np.zeros(self.dim)
        
        # Setup island topology
        base_size = self.np // self.n_islands
        remainder = self.np % self.n_islands
        self.island_sizes = np.full(self.n_islands, base_size)
        self.island_sizes[:remainder] += 1
        self.island_offsets = np.zeros(self.n_islands, dtype=int)
        self.island_offsets[1:] = np.cumsum(self.island_sizes)[:-1]
        
        # Reset tracking
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        self.stagnation_counter = 0
        self.successful_mutations = 0
        self.total_mutations = 0
        
    def _sample_trials_batch(self, n_samples):
        """
        Sample trial population using multivariate normal distribution.
        Returns array of shape (n_samples, dim).
        """
        samples = np.random.multivariate_normal(self.mean, self.cov, size=n_samples)
        return np.clip(samples, self.bounds[0], self.bounds[1])
    
    def _mutate_batch(self, population, fitness):
        """
        Generate mutation vectors using covariance-guided directions.
        Returns trial population of same shape as input.
        """
        # Compute z-scores for each individual
        z_scores = (population - self.mean) @ np.linalg.inv(self.cov)
        
        # Add small random perturbation scaled by step size
        perturbation = self.step_size * np.random.randn(self.np, self.dim) * 0.1
        
        # Create trials with biased direction toward better individuals
        better_mask = fitness < np.median(fitness)
        if np.any(better_mask):
            better_mean = population[better_mask].mean(axis=0)
            direction_bias = 0.2 * (better_mean - population)
        else:
            direction_bias = 0.0
        
        trials = population + perturbation + direction_bias
        return np.clip(trials, self.bounds[0], self.bounds[1])
    
    def _crossover_batch(self, population, trials):
        """
        Perform binomial crossover between population and trials.
        Uses per-dimension crossover rate.
        """
        cr = 0.5 + 0.3 * np.random.rand(self.dim)
        cr_expanded = cr[np.newaxis, :]
        
        # Randomly select which parent contributes each gene
        mask = np.random.rand(self.np, self.dim) < cr_expanded
        
        # Ensure at least one gene from trial
        mask[:, np.random.randint(self.dim)] = True
        
        offspring = np.where(mask, trials, population)
        return np.clip(offspring, self.bounds[0], self.bounds[1])
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Select survivors using (mu, lambda) selection.
        Returns (new_population, new_fitness).
        """
        # Combine population and trials
        combined = np.vstack([population, trials])
        combined_fitness = np.concatenate([fitness, trial_fitness])
        
        # Select best NP individuals
        indices = np.argsort(combined_fitness)[:self.np]
        return combined[indices], combined_fitness[indices]
    
    def _adapt_step_size(self, improvement_ratio):
        """
        Adapt step size based on improvement ratio.
        Uses cumulative evolution path for step size control.
        """
        # Update conjugate evolution path
        self.conjugate_path = (1 - self.cs) * self.conjugate_path + \
                              np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * \
                              (self.mean - self.evolution_path) / self.step_size
        
        # Compute norm of conjugate path
        norm_path = np.linalg.norm(self.conjugate_path)
        
        # Update step size based on path length
        if norm_path / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation)) < 1.5:
            self.step_size *= 1.0 + self.cs * (improvement_ratio - 0.2) / (1.0 + improvement_ratio)
        else:
            self.step_size *= 1.0 - self.cs * 0.2
        
        # Apply damping
        self.step_size *= np.exp(self.cs / self.damps * (1 - norm_path / (np.sqrt(self.dim) * 1.5 + 0.1)))
        
        # Keep step size bounded
        range_size = self.bounds[1] - self.bounds[0]
        self.step_size = np.clip(self.step_size, range_size * 1e-6, range_size * 0.5)
        
    def _adapt_covariance_variant_01(self, population, old_mean, improvement_occurred):
        """
        Update covariance matrix using rank-based adaptation.
        Variant 01: Uses weighted sum of successful mutation directions.
        """
        # Update evolution path
        self.evolution_path = (1 - self.cc) * self.evolution_path + \
                              np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * \
                              (self.mean - old_mean) / self.step_size
        
        # Compute rank-1 update component
        rank_one = self.c1 * np.outer(self.evolution_path, self.evolution_path)
        
        # Compute rank-mu update component
        z_diff = (population - old_mean) / self.step_size
        weights = np.maximum(0, self.mu_cov - np.arange(self.np)) / self.mu_cov
        rank_mu = self.cmu * np.sum([w * np.outer(z, z) for w, z in zip(weights, z_diff)], axis=0)
        
        # Apply covariance update
        self.cov = (1 - self.c1 - self.cmu) * self.cov + rank_one + rank_mu
        
        # Ensure positive definiteness
        self.cov = 0.5 * (self.cov + self.cov.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Adapt covariance learning rate based on improvement
        if not improvement_occurred:
            self.c1 = min(self.c1 * 1.1, 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff))
            self.cmu = min(self.cmu * 1.1, 0.5)
    
    def _adapt_covariance(self, population):
        """
        Additional covariance adaptation using eigenvalue adjustment.
        Ensures reasonable conditioning of covariance matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        
        # Limit eigenvalue range
        cond_max = 1e6
        eigenvalues = np.clip(eigenvalues, eigenvalues.max() / cond_max, eigenvalues.max())
        
        # Adjust step size based on eigenvalue spread
        eig_spread = np.log(eigenvalues.max() / (eigenvalues.min() + 1e-20))
        if eig_spread > np.log(cond_max):
            self.step_size *= np.exp(0.1 * (eig_spread - np.log(cond_max)) / np.log(cond_max))
        
        # Rebuild covariance
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def _compute_diversity(self, population):
        """
        Compute population diversity using average pairwise distance.
        Returns scalar diversity measure.
        """
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Compute average squared distance to mean
        dist_to_mean = np.sum((sample - sample.mean(axis=0)) ** 2, axis=1)
        return np.mean(dist_to_mean)
    
    def _restart_if_stagnant(self, fitness, diversity):
        """
        Check for stagnation and trigger restart if needed.
        """
        current_best = fitness.min()
        improvement = self.best_fitness - current_best
        
        if improvement > 1e-10:
            self.best_fitness = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Restart conditions
        should_restart = (
            self.stagnation_counter > 50 + self.dim or
            diversity < 1e-8 or
            self.step_size < 1e-10
        )
        
        if should_restart:
            # Reset with spread initialization
            self.mean = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            self.cov = np.eye(self.dim) * ((self.bounds[1] - self.bounds[0]) ** 2 / 16.0)
            self.step_size = (self.bounds[1] - self.bounds[0]) / 3.0
            self.evolution_path = np.zeros(self.dim)
            self.conjugate_path = np.zeros(self.dim)
            self.stagnation_counter = 0
            return True
        return False
    
    def _update_best(self, population, fitness):
        """Update tracked best solution."""
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = population[min_idx].copy()
    
    def _handle_truncated_evaluation(self, trials, trial_fitness):
        """
        Handle case where func returns fewer values than expected.
        """
        n_returned = len(trial_fitness)
        n_expected = len(trials)
        
        if n_returned < n_expected:
            # Truncate to valid evaluations
            trials = trials[:n_returned]
            return trials, trial_fitness, True
        return trials, trial_fitness, False
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and solution found
        """
        # Initialize
        self._initialize_population()
        
        # Initial population
        population = self._sample_trials_batch(self.np)
        fitness = func(population)
        
        if len(fitness) < self.np:
            # Budget exhausted immediately
            best_idx = np.argmin(fitness) if len(fitness) > 0 else 0
            return fitness[best_idx] if len(fitness) > 0 else np.inf, population[best_idx]
        
        self._update_best(population, fitness)
        
        # Main loop
        while not stopping_condition():
            self.generation += 1
            old_mean = self.mean.copy()
            
            # Generate trials using CMA-style sampling
            trials = self._sample_trials_batch(self.np)
            
            # Apply mutation and crossover
            trials = self._mutate_batch(population, fitness)
            trials = self._crossover_batch(population, trials)
            
            # Clip to bounds
            trials = np.clip(trials, self.bounds[0], self.bounds[1])
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle truncated evaluation
            trials, trial_fitness, truncated = self._handle_truncated_evaluation(trials, trial_fitness)
            
            if len(trial_fitness) == 0:
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Track successful mutations
            improvements = trial_fitness < fitness
            self.successful_mutations += improvements.sum()
            self.total_mutations += len(trials)
            
            # Compute improvement ratio
            old_best = fitness.min()
            new_best = min(fitness.min(), trial_fitness.min())
            improvement_ratio = max(0.0, (old_best - new_best) / (abs(old_best) + 1e-10))
            improvement_occurred = new_best < old_best
            
            # Select survivors
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            
            # Update mean from selected population
            self.mean = population.mean(axis=0)
            
            # Adapt step size
            self._adapt_step_size(improvement_ratio)
            
            # Adapt covariance matrix
            self._adapt_covariance_variant_01(population, old_mean, improvement_occurred)
            self._adapt_covariance(population)
            
            # Update best solution
            self._update_best(population, fitness)
            
            # Check for restart
            diversity = self._compute_diversity(population)
            if self._restart_if_stagnant(fitness, diversity):
                population = self._sample_trials_batch(self.np)
                fitness = func(population)
                
                if len(fitness) < self.np:
                    best_idx = np.argmin(fitness) if len(fitness) > 0 else 0
                    return fitness[best_idx] if len(fitness) > 0 else np.inf, population[best_idx]
                
                self._update_best(population, fitness)
        
        return self.best_fitness, self.best_solution
```