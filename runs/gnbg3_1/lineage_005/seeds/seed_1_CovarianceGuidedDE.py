import numpy as np


class CovarianceGuidedDE:
    """
    Covariance-Guided Differential Evolution (CGDE)
    
    A DE variant that maintains a covariance matrix to guide mutation
    directions, combined with adaptive F/CR parameters and periodic
    eigendecomposition for efficient sampling.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(5 * dim, 60), 300)  # Population size 5*dim, capped
        self.fitness_best = np.inf
        self.x_best = None
        self.generation = 0
        self.stagnation_counter = 0
        
        # Adaptive parameters
        self.f_scale = 0.5
        self.cr_prob = 0.5
        self.f_history = []
        self.cr_history = []
        
        # Covariance matrix (initialized as identity)
        self.cov_matrix = np.eye(dim)
        self.eigenvalues = np.ones(dim)
        self.eigenvectors = np.eye(dim)
        self.cov_update_counter = 0
        self.successful_mutations = []
        self.successful_fitness_deltas = []
        
        # Diversity tracking
        self.prev_diversity = np.inf
        
        # Bounds
        self.lb = -100.0
        self.ub = 100.0
        
    def _initialize_population(self, func):
        """Initialize random population within bounds."""
        pop = np.random.uniform(self.lb, self.ub, size=(self.np, self.dim))
        pop = self._clip_to_bounds(pop)
        fitness = self._evaluate_batch(pop, func)
        return pop, fitness
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lb, self.ub)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population batch."""
        fitness = func(population)
        if len(fitness) < len(population):
            return fitness[:len(population)]
        return fitness
    
    def _compute_pbest_indices(self, fitness, fraction=0.2):
        """Compute indices of top pbest individuals."""
        n_best = max(1, int(self.np * fraction))
        return np.argpartition(fitness, n_best - 1)[:n_best]
    
    def _mutate_batch_current_to_pbest(self, population, fitness):
        """DE/current-to-pbest/1 mutation with covariance guidance."""
        pbest_idx = self._compute_pbest_indices(fitness)
        pbest = population[pbest_idx]
        
        # Select three distinct random indices different from target
        indices = np.arange(self.np)
        perm = np.random.permutation(indices)
        
        # Covariance-guided direction for one of the vectors
        cov_direction = self._sample_from_covariance(1).flatten()
        
        mutated = np.empty_like(population)
        for i in range(self.np):
            r1, r2, r3 = perm[i], perm[(i + 1) % self.np], perm[(i + 2) % self.np]
            
            # Standard DE mutation
            base = population[i]
            v1 = population[r1]
            v2 = population[r2]
            v3 = population[r3]
            
            # Current-to-pbest component
            pbest_i = pbest[np.random.randint(len(pbest))]
            
            # Mix standard DE with covariance-guided mutation
            if np.random.random() < 0.3:
                # Use covariance-guided mutation occasionally
                f_adapted = self.f_scale * (1.0 + 0.2 * np.random.randn())
                f_adapted = np.clip(f_adapted, 0.1, 1.5)
                mutated[i] = base + f_adapted * cov_direction
            else:
                # Standard current-to-pbest/1/bin
                f_adapted = self.f_scale * (1.0 + 0.1 * np.random.randn())
                f_adapted = np.clip(f_adapted, 0.1, 1.5)
                mutated[i] = base + f_adapted * (pbest_i - base) + f_adapted * (v1 - v2)
            
            # Store mutation vector for covariance update
            self._record_mutation_vector(mutated[i] - base, i)
        
        return mutated
    
    def _sample_from_covariance(self, n_samples):
        """Sample directions from current covariance distribution."""
        noise = np.random.randn(n_samples, self.dim)
        # Transform by eigenvectors scaled by sqrt(eigenvalues)
        scaled = noise * np.sqrt(self.eigenvalues)
        return scaled @ self.eigenvectors.T
    
    def _record_mutation_vector(self, mutation_vec, idx):
        """Record mutation vector for covariance adaptation."""
        self.successful_mutations.append(mutation_vec.copy())
        self.successful_fitness_deltas.append(1.0)
        # Keep buffer bounded
        max_buffer = self.np * 5
        if len(self.successful_mutations) > max_buffer:
            self.successful_mutations = self.successful_mutations[-max_buffer:]
            self.successful_fitness_deltas = self.successful_fitness_deltas[-max_buffer:]
    
    def _crossover_batch(self, population, mutated, strategy='binomial'):
        """Binomial crossover between population and mutants."""
        trials = np.empty_like(population)
        cr_per_individual = self.cr_prob * np.ones(self.np) + 0.1 * np.random.randn(self.np)
        cr_per_individual = np.clip(cr_per_individual, 0.1, 0.95)
        
        for i in range(self.np):
            j_rand = np.random.randint(self.dim)
            mask = np.random.rand(self.dim) < cr_per_individual[i]
            mask[j_rand] = True
            trials[i] = np.where(mask, mutated[i], population[i])
        
        return trials, cr_per_individual
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection between parent and trial individuals."""
        better_mask = trial_fitness < fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        # Record successful mutations
        successful_trials = trials[better_mask]
        for i, idx in enumerate(np.where(better_mask)[0]):
            mutation_vec = trials[idx] - population[idx]
            improvement = fitness[idx] - trial_fitness[idx]
            if improvement > 0:
                self.successful_mutations.append(mutation_vec)
                self.successful_fitness_deltas.append(improvement)
        
        return new_population, new_fitness
    
    def _adapt_step_size(self, population, fitness, trials, trial_fitness):
        """Adapt mutation scale F based on success rate."""
        better_mask = trial_fitness < fitness
        success_rate = np.mean(better_mask)
        
        # Target success rate around 0.25
        if success_rate > 0.3:
            self.f_scale = min(1.0, self.f_scale * 1.1)
        elif success_rate < 0.15:
            self.f_scale = max(0.1, self.f_scale * 0.9)
        
        # Record for history
        self.f_history.append(self.f_scale)
        if len(self.f_history) > 50:
            self.f_history = self.f_history[-50:]
    
    def _adapt_crossover_prob(self, cr_per_trial):
        """Adapt crossover probability based on dimension success."""
        self.cr_history.append(np.mean(cr_per_trial))
        if len(self.cr_history) > 50:
            self.cr_history = self.cr_history[-50:]
    
    def _update_covariance_matrix(self):
        """Update covariance matrix from successful mutation history."""
        if len(self.successful_mutations) < 5:
            return
        
        # Weight by fitness improvement
        deltas = np.array(self.successful_fitness_deltas)
        weights = deltas / (np.sum(deltas) + 1e-10)
        
        # Stack successful mutations
        mutations = np.array(self.successful_mutations)
        
        # Compute weighted mean
        mean_mut = np.sum(mutations * weights[:, np.newaxis], axis=0)
        
        # Update covariance with rank-1 approximation
        learning_rate = 0.1 / (1 + len(self.successful_mutations) / (self.np * 10))
        self.cov_matrix = (1 - learning_rate) * self.cov_matrix + learning_rate * np.outer(mean_mut, mean_mut)
        
        # Ensure positive definiteness
        self.cov_matrix = 0.5 * (self.cov_matrix + self.cov_matrix.T)
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if np.any(eigvals < 1e-8):
            self.cov_matrix += (1e-6 - np.min(eigvals)) * np.eye(self.dim)
        
        # Clear history periodically
        if len(self.successful_mutations) > self.np * 10:
            self.successful_mutations = self.successful_mutations[-self.np:]
            self.successful_fitness_deltas = self.successful_fitness_deltas[-self.np:]
    
    def _perform_eigendecomposition(self):
        """Perform eigendecomposition of covariance matrix."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
        except np.linalg.LinAlgError:
            self.cov_matrix = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
    
    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        mean_pop = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - mean_pop) ** 2, axis=1))
        return np.mean(distances)
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
        if current_best < self.fitness_best - 1e-10:
            self.fitness_best = current_best
            self.stagnation_counter = 0
            best_idx = np.argmin(fitness)
            self.x_best = population[best_idx].copy() if self.x_best is None else self.x_best
            return False
        else:
            self.stagnation_counter += 1
            return self.stagnation_counter > 30
    
    def _restart_if_needed(self, population, fitness):
        """Perform partial restart if stagnated or low diversity."""
        current_diversity = self._compute_diversity(population)
        
        should_restart = (
            self.stagnation_counter > 30 or
            current_diversity < 0.1 * self.dim or
            self.generation > 500
        )
        
        if should_restart:
            # Reinitialize portion of population around best
            n_replace = self.np // 2
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = self.x_best + np.random.randn(self.dim) * current_diversity
            
            population = self._clip_to_bounds(population)
            fitness = self._evaluate_batch(population, func)
            
            # Reset adaptation state
            self.stagnation_counter = 0
            self.cov_matrix = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.successful_mutations = []
            self.successful_fitness_deltas = []
            
        return population, fitness
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population, fitness = self._initialize_population(func)
        
        if len(fitness) == 0 or np.any(np.isnan(fitness)):
            return np.inf, np.zeros(self.dim)
        
        self.fitness_best = np.min(fitness)
        best_idx = np.argmin(fitness)
        self.x_best = population[best_idx].copy()
        
        # Main evolution loop
        while not stopping_condition():
            self.generation += 1
            
            # Mutation with covariance guidance
            mutated = self._mutate_batch_current_to_pbest(population, fitness)
            mutated = self._clip_to_bounds(mutated)
            
            # Crossover
            trials, cr_per_trial = self._crossover_batch(population, mutated)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trials
            trial_fitness = self._evaluate_batch(trials, func)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trial_fitness) == 0:
                    break
            
            if stopping_condition():
                break
            
            # Selection
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            current_best = np.min(fitness)
            if current_best < self.fitness_best:
                self.fitness_best = current_best
                best_idx = np.argmin(fitness)
                self.x_best = population[best_idx].copy()
            
            # Adaptation
            self._adapt_step_size(population, fitness, trials, trial_fitness)
            self._adapt_crossover_prob(cr_per_trial)
            
            # Periodic covariance update
            self.cov_update_counter += 1
            if self.cov_update_counter >= 3:
                self._update_covariance_matrix()
                self._perform_eigendecomposition()
                self.cov_update_counter = 0
            
            # Diversity monitoring and restart
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness)
        
        return self.fitness_best, self.x_best
