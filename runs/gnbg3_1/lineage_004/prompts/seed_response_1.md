```python
import numpy as np


class AdaptiveCovarianceDE:
    """Differential Evolution with Covariance-Adaptive Mutation and CMA-ES-style parameter control.
    
    Key features:
    - Covariance matrix adaptation tracking dimension correlations
    - Multiple mutation strategies combined with adaptive weighting
    - CMA-ES inspired evolution paths for step-size and covariance adaptation
    - Stagnation detection with adaptive restarts
    - Fully vectorized batch operations for efficiency
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        # Population size: 4*dim to 8*dim, capped at 240 for dim=30
        self.np = min(max(4 * dim, 120), 240)
        
        # Search bounds
        self.lower = -100.0
        self.upper = 100.0
        
        # DE control parameters
        self.F = 0.7  # Initial scaling factor
        self.CR = 0.85  # Crossover rate
        
        # CMA-ES style adaptation parameters
        self.step_size = 1.0
        self.cov = np.eye(dim)
        self.p_s = np.zeros(dim)  # Evolution path for step size
        self.p_c = np.zeros(dim)  # Evolution path for covariance
        
        # Learning rates
        self.cs = None
        self.cc = None
        self.c1 = None
        self.cmu = None
        
        # Tracking
        self.success_history = []
        self.stagnation_counter = 0
        self.generation = 0
        self.best_fitness = np.inf
        self.best_solution = None
        
        # Diversity threshold for restart
        self.diversity_threshold = 0.05 * (self.upper - self.lower)
    
    def _initialize_parameters_cma(self):
        """Initialize CMA-ES style learning rates based on dimension."""
        # Cumulation for step size
        self.cs = (self.dim + 5.0) / 3.0
        self.cs = 1.0 / (self.cs + np.sqrt(self.cs * self.cs))
        
        # Cumulation for covariance
        self.cc = (4.0 + self.dim) / (self.dim + 4.0)
        self.cc = 2.0 / (self.dim + 4.0)
        
        # Rank-1 covariance update
        self.c1 = 2.0 / ((self.dim + 1.3) * (self.dim + 1.3))
        
        # Rank-mu covariance update
        self.cmu = 0.3 * (2.0 / (self.dim * self.dim + 6.0))
    
    def _initialize_population(self, func):
        """Initialize population uniformly in bounds and evaluate fitness."""
        # Latin Hypercube Sampling for better initial distribution
        population = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            samples = np.linspace(self.lower, self.upper, self.np)
            population[:, d] = np.random.permutation(samples)
            population[:, d] += np.random.uniform(-1, 1, self.np) * (self.upper - self.lower) / self.np
        
        # Clip to bounds
        population = np.clip(population, self.lower, self.upper)
        
        # Evaluate fitness
        fitness = func(population)
        
        # Handle budget exhaustion
        if len(fitness) < self.np:
            valid_len = len(fitness)
            fitness = np.resize(fitness, self.np)
            fitness[valid_len:] = np.inf
            population = population[:valid_len] if valid_len > 0 else population
        
        # Initialize covariance from population spread
        if len(population) > 1:
            centered = population - population.mean(axis=0)
            self.cov = np.cov(centered.T)
            # Ensure positive definite
            min_eig = np.min(np.linalg.eigvalsh(self.cov))
            if min_eig < 1e-8:
                self.cov += (1e-7 - min_eig) * np.eye(self.dim)
        
        # Initialize CMA parameters
        self._initialize_parameters_cma()
        
        # Track best
        best_idx = np.nanargmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = population[best_idx].copy()
        
        return population, fitness
    
    def _mutate_batch(self, population, fitness):
        """Generate mutant vectors using covariance-adaptive mutation.
        
        Combines three strategies:
        1. Covariance-sampled: uses current covariance matrix for correlated exploration
        2. Current-to-pbest: directed mutation toward good region
        3. Rand/1: classic DE randomness for diversity
        """
        np_pop = len(population)
        dim = self.dim
        
        # Strategy weights based on stagnation
        if self.stagnation_counter > 5:
            # More exploration when stuck
            w_cov, w_pbest, w_rand = 0.5, 0.3, 0.2
        elif self.generation < 10:
            # Initial phase: balance exploration and exploitation
            w_cov, w_pbest, w_rand = 0.3, 0.4, 0.3
        else:
            # Normal phase: focus on exploitation
            w_cov, w_pbest, w_rand = 0.2, 0.5, 0.3
        
        # Strategy 1: Covariance-sampled mutation
        # Sample from multivariate normal with current covariance
        try:
            L = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            L = np.eye(dim)
        
        noise = np.random.randn(np_pop, dim) @ L.T
        cov_mutations = self.step_size * noise
        
        # Strategy 2: Current-to-pbest mutation
        # Select top 20% as pbest candidates
        n_pbest = max(np_pop // 5, 1)
        pbest_indices = np.argpartition(fitness, n_pbest)[:n_pbest]
        pbest = population[pbest_indices]
        
        # For each target, select random pbest
        pbest_idx = np.random.randint(0, n_pbest, np_pop)
        pbest_selected = pbest[pbest_idx]
        
        # Random indices for mutation
        rand_idx = np.random.randint(0, np_pop, np_pop)
        rand_vectors = population[rand_idx]
        
        # Current-to-pbest with F
        F_pbest = np.random.uniform(0.3, 1.0, np_pop)
        pbest_mutations = F_pbest[:, None] * (pbest_selected - population) + \
                          np.random.uniform(0.3, 0.9, np_pop)[:, None] * (rand_vectors - population)
        
        # Strategy 3: Classic rand/1 mutation
        idx1 = np.random.randint(0, np_pop, np_pop)
        idx2 = np.random.randint(0, np_pop, np_pop)
        idx3 = np.random.randint(0, np_pop, np_pop)
        
        F_rand = np.random.uniform(0.5, 1.0, np_pop)
        rand_mutations = F_rand[:, None] * (population[idx1] - population[idx2]) + \
                         population[idx3]
        
        # Combine strategies
        mutations = w_cov * cov_mutations + w_pbest * pbest_mutations + w_rand * rand_mutations
        
        # Apply mutation to create donor vectors
        donor = population + mutations
        
        return donor
    
    def _crossover_batch(self, population, donor):
        """Binomial crossover between target and donor vectors."""
        np_pop, dim = population.shape
        
        # Random threshold for each individual and dimension
        r = np.random.randint(0, dim, np_pop)
        j_rand = np.zeros((np_pop, dim), dtype=bool)
        j_rand[np.arange(np_pop), r] = True
        
        # Crossover mask
        mask = np.random.random((np_pop, dim)) < self.CR
        mask = mask | j_rand
        
        # Generate trial vectors
        trial = np.where(mask, donor, population)
        
        return trial
    
    def _select_survivors_batch(self, population, fitness, trial, trial_fitness):
        """Select better individuals between target and trial vectors."""
        # Better fitness wins (minimization)
        improved = trial_fitness < fitness
        
        # Update population
        population = np.where(improved[:, None], trial, population)
        fitness = np.where(improved, trial_fitness, fitness)
        
        return population, fitness, improved
    
    def _adapt_step_size(self, n_improved, np_pop):
        """Adapt step size based on success rate (CMA-ES style)."""
        # Record success
        if n_improved > 0:
            self.success_history.append(1.0)
        else:
            self.success_history.append(0.0)
        
        # Keep only recent history
        if len(self.success_history) > self.np * 2:
            self.success_history = self.success_history[-self.np * 2:]
        
        # Compute success rate
        success_rate = np.mean(self.success_history[-min(50, len(self.success_history)):])
        p_s_target = 0.2  # Target success rate
        
        # Update evolution path for step size
        self.p_s = (1 - self.cs) * self.p_s + \
                   np.sqrt(self.cs * (2 - self.cs)) * p_s_target * \
                   np.sqrt(1.0 / max(success_rate, 0.01)) * \
                   np.sign(success_rate - p_s_target + 1e-8)
        
        # Update step size
        self.step_size *= np.exp(self.cs / 2.0 * (np.linalg.norm(self.p_s) ** 2 / self.dim - 1.0))
        
        # Bound step size
        self.step_size = np.clip(self.step_size, 1e-10, 10.0)
    
    def _adapt_covariance(self, improved_mask, population, trial):
        """Adapt covariance matrix using evolution paths and successful steps."""
        dim = self.dim
        
        # Update evolution path for covariance
        if np.linalg.norm(self.p_s) > 1.5 * np.sqrt(dim):
            self.p_c = (1 - self.cc) * self.p_c + \
                       np.sqrt(self.cc * (2 - self.cc)) * self.step_size * \
                       (self.mean - self.best_solution) / max(self.step_size, 1e-10)
        else:
            self.p_c = (1 - self.cc) * self.p_c
        
        # Rank-1 update of covariance
        self.cov = (1 - self.c1 - self.cmu) * self.cov + \
                   self.c1 * np.outer(self.p_c, self.p_c)
        
        # Rank-mu update using recent successful steps
        if improved_mask.sum() > 0:
            successful_steps = trial[improved_mask] - population[improved_mask]
            
            # Weight recent steps more
            n_steps = len(successful_steps)
            weights = np.exp(-np.arange(n_steps) * 0.1)
            weights /= weights.sum()
            
            # Weighted outer product update
            weighted_cov = np.zeros((dim, dim))
            for i, step in enumerate(successful_steps):
                weighted_cov += weights[i] * np.outer(step, step)
            
            self.cov += self.cmu * weighted_cov
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.cov))
        if min_eig < 1e-10:
            self.cov += (1e-9 - min_eig) * np.eye(dim)
    
    def _adapt_F_CR(self, n_improved, np_pop):
        """Adapt F and CR based on success rate."""
        success_rate = n_improved / np_pop if np_pop > 0 else 0
        
        # Adapt F: decrease if success rate is low, increase if high
        if success_rate > 0.3:
            self.F = min(1.0, self.F * 1.05)
        elif success_rate < 0.1:
            self.F = max(0.3, self.F * 0.95)
        
        # Adapt CR: decrease if diversity is low, increase if high
        diversity = self._compute_diversity(self.population)
        if diversity < self.diversity_threshold:
            self.CR = max(0.5, self.CR * 0.98)
        else:
            self.CR = min(0.95, self.CR * 1.01)
    
    def _compute_diversity(self, population):
        """Compute population diversity as average Euclidean distance from centroid."""
        if len(population) < 2:
            return 1.0
        
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances) / (self.upper - self.lower)
    
    def _restart_if_stagnant(self, population, fitness):
        """Detect stagnation and reinitialize part of the population."""
        # Update mean
        self.mean = np.mean(population, axis=0)
        
        # Compute diversity
        diversity = self._compute_diversity(population)
        
        # Increment stagnation counter if diversity is low
        if diversity < self.diversity_threshold * 2:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Restart if stagnant
        if self.stagnation_counter > max(50, self.dim * 3):
            self.stagnation_counter = 0
            
            # Replace worst 30% with new random individuals
            n_replace = max(self.np // 3, 1)
            ranks = np.argsort(fitness)
            replace_idx = ranks[-n_replace:]
            
            # Generate new individuals using current covariance
            try:
                L = np.linalg.cholesky(self.cov * 0.5)
                for idx in replace_idx:
                    noise = np.random.randn(self.dim) @ L.T
                    new_ind = self.mean + noise * self.step_size
                    new_ind = np.clip(new_ind, self.lower, self.upper)
                    population[idx] = new_ind
            except np.linalg.LinAlgError:
                # Fallback to uniform random
                population[replace_idx] = np.random.uniform(
                    self.lower, self.upper, (n_replace, self.dim)
                )
            
            # Reset some parameters
            self.step_size = 1.0
            self.p_s *= 0.5
            self.p_c *= 0.5
        
        return population
    
    def _handle_budget_exhaustion(self, trial, trial_fitness):
        """Handle case where func returns fewer values than expected."""
        n_requested = len(trial)
        n_received = len(trial_fitness)
        
        if n_received < n_requested:
            # Identify unevaluated trials
            valid_mask = np.ones(n_requested, dtype=bool)
            valid_mask[:n_received] = False
            
            # Replace unevaluated with targets to avoid incorrect selection
            trial[valid_mask] = self.population[valid_mask]
            
            # Truncate fitness
            if n_received > 0:
                trial_fitness = trial_fitness[:n_received]
            else:
                trial_fitness = np.array([np.inf])
        
        return trial, trial_fitness
    
    def _update_best(self, population, fitness, improved_mask):
        """Update best solution if improvements found."""
        if improved_mask.sum() > 0:
            improved_pop = population[improved_mask]
            improved_fit = fitness[improved_mask]
            
            best_improved_idx = np.nanargmin(improved_fit)
            if improved_fit[best_improved_idx] < self.best_fitness:
                best_global_idx = np.where(improved_mask)[0][best_improved_idx]
                self.best_fitness = fitness[best_global_idx]
                self.best_solution = population[best_global_idx].copy()
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population, fitness = self._initialize_population(func)
        self.population = population
        self.generation = 0
        
        # Main loop
        while not stopping_condition():
            self.generation += 1
            
            # Generate trial vectors (mutation + crossover)
            donor = self._mutate_batch(population, fitness)
            trial = self._crossover_batch(population, donor)
            
            # Clip trial to bounds
            trial = np.clip(trial, self.lower, self.upper)
            
            # Evaluate trial fitness
            trial_fitness = func(trial)
            
            # Handle budget exhaustion
            if stopping_condition():
                break
            trial, trial_fitness = self._handle_budget_exhaustion(trial, trial_fitness)
            
            # Check budget again
            if stopping_condition():
                break
            
            # Selection
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial, trial_fitness
            )
            
            self.population = population
            n_improved = np.sum(improved_mask)
            
            # Update best solution
            self._update_best(population, fitness, improved_mask)
            
            # Adapt parameters
            self._adapt_step_size(n_improved, len(population))
            self._adapt_covariance(improved_mask, trial[trial_fitness < fitness], trial[trial_fitness < fitness])
            self._adapt_F_CR(n_improved, len(population))
            
            # Restart if stagnant
            population = self._restart_if_stagnant(population, fitness)
            
            # Re-evaluate fitness if population was modified
            if self.stagnation_counter == 0 and self.generation > 1:
                fitness = func(population)
                fitness = np.resize(fitness, len(population))
                fitness[np.isnan(fitness)] = np.inf
        
        return self.best_fitness, self.best_solution
```