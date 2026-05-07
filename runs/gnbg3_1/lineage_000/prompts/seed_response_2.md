```python
import numpy as np


class CovarianceGuidedEvolutionaryOptimizer:
    """
    A population-based optimizer that combines CMA-ES-style covariance adaptation
    with a competitive selection mechanism. Uses eigendecomposition-free rank-1
    and rank-mu covariance updates with weighted recombination.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(6 * dim, 120), 300)  # Population size: 120-300 for dim=30
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # CMA-ES inspired parameters
        self.mean = None
        self.cov = None
        self.step_size = 0.5
        self.pc = None  # Evolution path for covariance
        self.ps = None  # Evolution path for step size
        
        # Selection weights (CMA-ES style: positive weights for top performers)
        self.weights = self._compute_recombination_weights()
        
        # Adaptation constants
        self._set_adaptation_constants()
        
        # State variables
        self.population = None
        self.fitness = None
        self.f_opt = np.inf
        self.x_opt = None
        self.generation = 0
        self.best_history = []
        self.stagnation_counter = 0
        
        # Restart parameters
        self.max_stagnation = 50
        self.min_step_size = 1e-10
    
    def _compute_recombination_weights(self):
        """Compute weights for weighted recombination of selected individuals."""
        weights = np.log(self.NP / 2 + 0.5) - np.log(np.arange(1, self.NP + 1))
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        return weights * self.NP
    
    def _set_adaptation_constants(self):
        """Set CMA-ES adaptation constants based on dimension."""
        self.cc = 2.0 / (self.dim + 2)
        self.c1 = 2.0 / (self.dim ** 2 + 6.0)
        cmu_temp = 2.0 * (self.NP - 2) / (self.dim ** 2 + 6.0)
        self.cmu = min(1 - self.c1, cmu_temp)
        self.damp_ss = 1.0 + np.sqrt(self.NP / self.dim)
        self.expected_norm = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self.stopping_condition = stopping_condition
        
        self._initialize_evolution_paths()
        self._initialize_population()
        self.fitness = self._evaluate_batch(self.population)
        self._handle_budget_exhaustion(self.population, self.fitness)
        self._update_best_solution()
        self._record_initial_history()
        
        while not stopping_condition():
            trials = self._build_trial_batch()
            trial_fitness = self._evaluate_batch(trials)
            
            if len(trial_fitness) < len(trials):
                break
            
            if stopping_condition():
                break
            
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            self._update_best_solution()
            self._update_mean_vector()
            self._update_evolution_paths()
            self._update_covariance_matrix()
            self._adapt_step_size()
            self._check_and_handle_stagnation()
            
            self.generation += 1
        
        return self.f_opt, self.x_opt
    
    def _initialize_evolution_paths(self):
        """Initialize CMA-ES evolution paths to zero."""
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
    
    def _initialize_population(self):
        """Initialize population uniformly within bounds and set initial mean."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
        self.mean = self.population.mean(axis=0)
        self.cov = np.eye(self.dim)
    
    def _evaluate_batch(self, candidates):
        """Evaluate fitness for entire batch at once."""
        return self.func(candidates)
    
    def _handle_budget_exhaustion(self, batch, fitness_result):
        """Handle case where budget is exhausted mid-batch."""
        if len(fitness_result) < len(batch):
            valid_len = len(fitness_result)
            self.population = self.population[:valid_len]
            self.fitness = self.fitness[:valid_len]
            self.NP = valid_len
    
    def _record_initial_history(self):
        """Record initial best fitness for stagnation tracking."""
        best_idx = np.argmin(self.fitness)
        self.best_history.append(self.fitness[best_idx])
    
    def _build_trial_batch(self):
        """Build trial batch using covariance-guided mutation and crossover."""
        mutants = self._generate_covariance_guided_mutants()
        trials = self._apply_crossover(mutants)
        return trials
    
    def _generate_covariance_guided_mutants(self):
        """Generate mutant vectors using current mean and covariance matrix."""
        L = np.linalg.cholesky(self.cov)
        z = np.random.standard_normal((self.NP, self.dim))
        y = z @ L.T
        mutants = self.mean + self.step_size * y
        return np.clip(mutants, self.lower_bound, self.upper_bound)
    
    def _apply_crossover(self, mutants):
        """Apply binomial crossover between population and mutants."""
        cr_mask = np.random.rand(self.NP, self.dim) < 0.9
        forced_indices = np.random.randint(0, self.dim, size=self.NP)
        cr_mask[np.arange(self.NP), forced_indices] = True
        trials = np.where(cr_mask, mutants, self.population)
        return np.clip(trials, self.lower_bound, self.upper_bound)
    
    def _select_survivors_batch(self, pop_old, fit_old, trials, trial_fit):
        """Select survivors using (mu, lambda) selection with elitism."""
        combined_pop = np.vstack([pop_old, trials])
        combined_fit = np.concatenate([fit_old, trial_fit])
        sorted_indices = np.argsort(combined_fit)
        selected_indices = sorted_indices[:self.NP]
        return combined_pop[selected_indices], combined_fit[selected_indices]
    
    def _update_best_solution(self):
        """Track the best solution found so far."""
        best_idx = np.nanargmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.population[best_idx].copy()
    
    def _update_mean_vector(self):
        """Update mean using weighted recombination of selected population."""
        self.mean = np.sum(self.population * self.weights[:, np.newaxis], axis=0)
    
    def _update_evolution_paths(self):
        """Update evolution paths for covariance and step size adaptation."""
        y_mean = (self.mean - self._get_previous_mean()) / self.step_size
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc)) * y_mean
        self.ps = (1 - self.damp_ss) * self.ps + np.sqrt(self.damp_ss) * y_mean
    
    def _get_previous_mean(self):
        """Compute previous mean estimate for path updates."""
        sorted_indices = np.argsort(self.fitness)
        prev_mean = np.zeros(self.dim)
        for i, idx in enumerate(sorted_indices[:self.NP]):
            prev_mean += self.weights[i] * self.population[idx]
        return prev_mean / self.weights.sum() * self.NP
    
    def _update_covariance_matrix(self):
        """Update covariance matrix using rank-1 and rank-mu updates."""
        rank_one = self.pc[:, np.newaxis] * self.pc[np.newaxis, :]
        self.cov = (1 - self.c1 - self.cmu) * self.cov + self.c1 * rank_one
    
    def _adapt_step_size(self):
        """Adapt step size using cumulative step size adaptation (CSA)."""
        ps_norm = np.linalg.norm(self.ps)
        if ps_norm > 0:
            factor = np.exp((ps_norm / self.expected_norm - 1) * self.damp_ss)
            self.step_size = max(self.min_step_size, self.step_size * factor)
    
    def _check_and_handle_stagnation(self):
        """Check for stagnation and trigger restart if necessary."""
        current_best = np.nanmin(self.fitness)
        self.best_history.append(current_best)
        
        if len(self.best_history) > 2:
            if self.best_history[-1] >= self.best_history[-2]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        if self.stagnation_counter >= self.max_stagnation:
            self._restart_optimizer()
    
    def _restart_optimizer(self):
        """Restart optimizer with fresh population and reset parameters."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
        self.fitness = self._evaluate_batch(self.population)
        self._handle_budget_exhaustion(self.population, self.fitness)
        
        self.mean = self.population.mean(axis=0)
        self.cov = np.eye(self.dim)
        self.step_size = 0.5
        self._initialize_evolution_paths()
        
        self.best_history = []
        self.stagnation_counter = 0
```