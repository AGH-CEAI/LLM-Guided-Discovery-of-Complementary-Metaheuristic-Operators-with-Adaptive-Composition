import numpy as np

class CovarianceGuidedES:
    """
    Covariance-Guided Evolutionary Strategy with Eigendecomposition-based Adaptation.
    
    A population-based optimizer that maintains a multivariate normal distribution
    over the search space, adapting its mean and covariance matrix based on
    successful mutations. Uses IPOP-style step size adaptation and stagnation-triggered restarts.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        self.NP = min(150, max(50, 4 * dim))
        
        # Population state
        self.population = None
        self.fitness = None
        
        # Best solution tracking
        self.best_x = None
        self.best_f = np.inf
        self.gen_of_best = 0
        
        # CMA-ES inspired distribution parameters
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.step_size = 1.0
        
        # History for covariance adaptation
        self.success_history = []
        self.max_history_size = 5 * dim
        
        # Step size control (IPOP-CMA-ES inspired)
        self.success_window = []
        self.failure_window = []
        self.p_target = 0.4
        
        # Stagnation detection and restart
        self.stagnation_gen = 0
        self.stagnation_threshold = 30
        self.restart_count = 0
        
        # Generation counter
        self.current_gen = 0
        
        self._initialize()
    
    def _initialize(self):
        """Initialize population, distribution mean, and eigendecomposition."""
        self.population = self._random_population()
        self.fitness = self._evaluate_batch(self.population)
        
        self.mean = np.mean(self.population, axis=0)
        self._initialize_eigendecomposition()
        
        self.success_history = []
        self.success_window = []
        self.failure_window = []
        self.stagnation_gen = 0
        self.current_gen = 0
        
        best_idx = np.argmin(self.fitness)
        self._update_best(self.population[best_idx], self.fitness[best_idx])
    
    def _random_population(self):
        """Generate random population within bounds."""
        return np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
    
    def _initialize_eigendecomposition(self):
        """Initialize eigendecomposition of covariance matrix."""
        range_scale = (self.upper - self.lower) / 6.0
        self.eigenvalues = np.full(self.dim, (range_scale * self.step_size) ** 2)
        self.eigenvectors = np.eye(self.dim)
    
    def _evaluate_batch(self, pop):
        """Evaluate fitness for entire population at once."""
        return self.func(pop)
    
    def _sample_mutation_directions_batch(self):
        """Sample mutation directions from current eigendecomposition-distribution."""
        noise = np.random.randn(self.NP, self.dim)
        scaled = noise * np.sqrt(self.eigenvalues)
        directions = scaled @ self.eigenvectors.T
        return directions * self.step_size
    
    def _mutate_batch(self):
        """Generate mutant vectors centered on distribution mean."""
        directions = self._sample_mutation_directions_batch()
        mutants = self.mean + directions
        return mutants
    
    def _crossover_batch(self, mutants):
        """Binomial crossover between current population and mutants."""
        cr = np.random.uniform(0.3, 0.9)
        mask = np.random.rand(self.NP, self.dim) < cr
        trials = np.where(mask, mutants, self.population)
        return trials
    
    def _clip_to_bounds(self, pop):
        """Ensure all candidates are within search bounds."""
        return np.clip(pop, self.lower, self.upper)
    
    def _update_best(self, x, f):
        """Update global best solution if new candidate is better."""
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.gen_of_best = self.current_gen
            return True
        return False
    
    def _update_best_batch(self, trials, trial_fitness):
        """Check if any trial improves the global best."""
        improved_mask = trial_fitness < self.best_f
        if np.any(improved_mask):
            best_idx = np.argmin(trial_fitness)
            self._update_best(trials[best_idx], trial_fitness[best_idx])
    
    def _select_survivors_batch(self, trials, trial_fitness):
        """Greedy selection between population and trials, track successes."""
        better = trial_fitness < self.fitness
        self.population = np.where(better[:, np.newaxis], trials, self.population)
        self.fitness = np.where(better, trial_fitness, self.fitness)
        
        self.success_window.append(np.sum(better))
        self.failure_window.append(np.sum(~better))
        
        if len(self.success_window) > 10:
            self.success_window = self.success_window[-10:]
            self.failure_window = self.failure_window[-10:]
        
        if np.any(better):
            successful_deltas = trials[better] - self.population[better]
            for delta in successful_deltas:
                self.success_history.append(delta)
            if len(self.success_history) > self.max_history_size:
                self.success_history = self.success_history[-self.max_history_size:]
            self.stagnation_gen = 0
        else:
            self.stagnation_gen += 1
    
    def _adapt_step_size(self):
        """Adapt step size based on recent success rate (IPOP-CMA-ES style)."""
        if len(self.success_window) < 5:
            return
        
        recent_success = sum(self.success_window[-5:])
        recent_total = recent_success + sum(self.failure_window[-5:])
        p_success = recent_success / recent_total if recent_total > 0 else 0.0
        
        if p_success > self.p_target:
            self.step_size *= 1.15
        elif p_success < (1.0 - self.p_target):
            self.step_size *= 0.85
        
        self.step_size = np.clip(self.step_size, 1e-10, 100.0)
    
    def _update_distribution_mean(self):
        """Update distribution mean toward successful mutation centroid."""
        if len(self.success_history) < 1:
            return
        
        recent_success = np.array(self.success_history[-self.NP:])
        if len(recent_success) == 0:
            return
        
        shift = np.mean(recent_success, axis=0)
        self.mean += 0.25 * shift
        self.mean = np.clip(self.mean, self.lower, self.upper)
    
    def _update_eigendecomposition(self):
        """Update eigendecomposition from covariance of successful mutations."""
        min_samples = max(self.dim, 10)
        if len(self.success_history) < min_samples:
            return
        
        recent = np.array(self.success_history[-min_samples * 2:])
        if len(recent) < min_samples:
            return
        
        mean_shift = np.mean(recent, axis=0)
        centered = recent - mean_shift
        sample_cov = (centered.T @ centered) / len(recent)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(sample_cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            
            inherit_rate = 0.7
            self.eigenvalues = inherit_rate * self.eigenvalues + (1.0 - inherit_rate) * eigvals
            self.eigenvalues = np.clip(self.eigenvalues, 1e-10, None)
            
            self.eigenvectors = eigvecs
        except np.linalg.LinAlgError:
            pass
    
    def _compute_diversity(self):
        """Compute population diversity as mean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_if_stagnant(self):
        """Reinitialize population if stagnation threshold is exceeded."""
        if self.stagnation_gen < self.stagnation_threshold:
            return False
        
        old_best_x = self.best_x.copy()
        old_best_f = self.best_f
        
        self.population = self._random_population()
        self.fitness = self._evaluate_batch(self.population)
        
        self.mean = np.mean(self.population, axis=0)
        self._initialize_eigendecomposition()
        
        self.success_history = []
        self.success_window = []
        self.failure_window = []
        self.stagnation_gen = 0
        self.restart_count += 1
        
        worst_idx = np.argmax(self.fitness)
        self.population[worst_idx] = old_best_x
        self.fitness[worst_idx] = old_best_f
        self.best_x = old_best_x
        self.best_f = old_best_f
        
        return True
    
    def _should_stop_for_diversity(self):
        """Check if population has converged to a single point."""
        if self._compute_diversity() < 1e-6:
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer until stopping condition is met.
        
        Returns:
            tuple: (best_fitness, best_solution)
        """
        self.func = func
        
        while not stopping_condition():
            mutants = self._mutate_batch()
            trials = self._crossover_batch(mutants)
            trials = self._clip_to_bounds(trials)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                actual_len = len(trial_fitness)
                trials = trials[:actual_len]
                trial_fitness = trial_fitness[:actual_len]
                if actual_len == 0:
                    break
            
            self._update_best_batch(trials, trial_fitness)
            
            if stopping_condition():
                break
            
            self._select_survivors_batch(trials, trial_fitness)
            
            self._clip_to_bounds(self.population)
            
            self._adapt_step_size()
            self._update_distribution_mean()
            self._update_eigendecomposition()
            
            if self._should_stop_for_diversity():
                self._restart_if_stagnant()
            
            self._restart_if_stagnant()
            
            self.current_gen += 1
        
        return self.best_x, self.best_f
