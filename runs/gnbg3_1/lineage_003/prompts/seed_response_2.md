```python
import numpy as np


class CovarianceGuidedOptimizer:
    """Covariance-guided population optimizer with adaptive sampling and step-size control.
    
    Key features:
    - Simplified CMA-ES covariance adaptation
    - Multiple sampling strategies (global, localized, directional)
    - Adaptive step size based on success history
    - Diversity-triggered restarts
    - Fully vectorized batch operations
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(8 * dim, 240)  # Population size
        self.bounds = np.array([-100.0, 100.0])
        self.lb = self.bounds[0]
        self.ub = self.bounds[1]
        
        # Evolution path and covariance state
        self.pc = None
        self.ps = None
        self.cov = None
        self.mean = None
        self.step_size = 1.0
        
        # Adaptation parameters
        self.cov Learning_rate = 0.02
        self.p Learning_rate = 0.1
        self.cs = 0.3
        self.sigma Learning_rate = 0.1
        
        # Success history for step size adaptation
        self.success_buffer = np.zeros(5 + int(3 * np.log(dim)))
        self.sigma_increase = 1.2
        self.sigma_decrease = 0.9
        self.sigma_target = 0.2
        
        # Diversity tracking
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.max_stagnation = 50 + dim
    
    def _initialize_population(self, func):
        """Initialize population, covariance matrix, and evolution paths."""
        self.mean = np.random.uniform(self.lb, self.ub, self.dim)
        self.cov = np.eye(self.dim) * (self.ub - self.lb) * 0.5
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.step_size = (self.ub - self.lb) * 0.1
        
        # Initial population sampling
        population = self._sample_trials_batch(self.np)
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            population = population[:len(fitness)]
            fitness = fitness[:len(fitness)]
        
        return population, fitness
    
    def _sample_trials_batch(self, n_samples):
        """Sample candidates using covariance matrix with adaptive strategies."""
        # Mix of sampling strategies
        strategy_weights = np.array([0.6, 0.25, 0.15])  # global, localized, directional
        
        # Strategy 1: Global sampling from multivariate normal
        L = np.linalg.cholesky(self.cov + np.eye(self.dim) * 1e-8)
        global_samples = self.mean + self.step_size * (L @ np.random.randn(self.dim, n_samples)).T
        
        # Strategy 2: Localized sampling around current best
        best_idx = np.argmin(self.best_fitness_history) if self.best_fitness_history else 0
        best_pos = self.mean + np.random.randn(self.dim) * self.step_size * 0.5
        local_samples = best_pos + self.step_size * 0.3 * np.random.randn(n_samples, self.dim)
        
        # Strategy 3: Directional sampling along improvement direction
        if len(self.best_fitness_history) > 1:
            direction = np.random.randn(self.dim)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            directional_offset = self.step_size * 1.5 * np.outer(np.random.randn(n_samples), direction)
            directional_samples = self.mean + directional_offset
        else:
            directional_samples = global_samples
        
        # Blend strategies based on weights
        samples = (strategy_weights[0] * global_samples + 
                   strategy_weights[1] * local_samples + 
                   strategy_weights[2] * directional_samples)
        
        return samples
    
    def _clip_to_bounds(self, population):
        """Clip all candidates to search bounds."""
        return np.clip(population, self.lb, self.ub)
    
    def _update_covariance(self, selected_pop, selected_weights):
        """Update covariance matrix using CMA-ES style adaptation."""
        if len(selected_pop) < 2:
            return
        
        # Compute weighted mean
        new_mean = np.sum(selected_pop * selected_weights[:, np.newaxis], axis=0)
        mean_diff = new_mean - self.mean
        
        # Update evolution paths
        self.pc = (1 - self.p Learning_rate) * self.pc + np.sqrt(self.p Learning_rate * (2 - self.p Learning_rate)) * mean_diff / self.step_size
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)) * mean_diff / self.step_size
        
        # Update covariance
        rank_one_update = self.pc[:, np.newaxis] @ self.pc[np.newaxis, :]
        rank_mu_update = np.zeros((self.dim, self.dim))
        
        for i in range(len(selected_pop)):
            diff = (selected_pop[i] - self.mean) / self.step_size
            rank_mu_update += selected_weights[i] * (diff[:, np.newaxis] @ diff[np.newaxis, :])
        
        self.cov = ((1 - self.cov Learning_rate) * self.cov + 
                    self.cov Learning_rate * (rank_one_update + (1 + (1 - 0)**2) * rank_mu_update))
        
        # Ensure positive definiteness
        self.cov = 0.5 * (self.cov + self.cov.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        self.mean = new_mean
    
    def _adapt_step_size(self, recent_success_rate):
        """Adapt step size based on success history."""
        # Update success buffer
        self.success_buffer[:-1] = self.success_buffer[1:]
        self.success_buffer[-1] = recent_success_rate
        
        # Compute moving average success rate
        avg_success = np.mean(self.success_buffer)
        
        # Adapt step size
        if avg_success > self.sigma_target:
            self.step_size *= self.sigma_increase
        elif avg_success < self.sigma_target * 0.5:
            self.step_size *= self.sigma_decrease
        
        # Bound step size
        range_size = self.ub - self.lb
        self.step_size = np.clip(self.step_size, range_size * 0.001, range_size * 0.5)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors based on fitness comparison."""
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        
        if len(combined_fit) > self.np:
            # Keep best NP individuals
            sorted_indices = np.argsort(combined_fit)[:self.np]
            new_population = combined_pop[sorted_indices]
            new_fitness = combined_fit[sorted_indices]
        else:
            new_population = combined_pop
            new_fitness = combined_fit
        
        return new_population, new_fitness
    
    def _compute_diversity(self, population):
        """Compute population diversity measure."""
        if len(population) < 2:
            return 0.0
        
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        diversity = np.mean(distances)
        
        return diversity
    
    def _compute_selection_weights(self, fitness):
        """Compute weights for selected individuals based on fitness."""
        sorted_indices = np.argsort(fitness)
        ranks = np.argsort(sorted_indices) + 1
        weights = 1.0 / (ranks + 1)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _restart_if_stagnant(self, current_best):
        """Check for stagnation and trigger restart if needed."""
        self.best_fitness_history.append(current_best)
        
        if len(self.best_fitness_history) > 1:
            if np.isclose(self.best_fitness_history[-1], self.best_fitness_history[-2], rtol=1e-6):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        diversity = self._compute_diversity(self.mean.reshape(1, -1))
        should_restart = (self.stagnation_counter > self.max_stagnation or 
                         self.step_size < 1e-6 or 
                         diversity < 1e-6)
        
        if should_restart:
            self._reset_evolution_state()
        
        return should_restart
    
    def _reset_evolution_state(self):
        """Reset evolution state for restart."""
        self.mean = np.random.uniform(self.lb, self.ub, self.dim)
        self.cov = np.eye(self.dim) * (self.ub - self.lb) * 0.3
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.step_size = (self.ub - self.lb) * 0.15
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.success_buffer = np.zeros_like(self.success_buffer)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        population, fitness = self._initialize_population(func)
        
        if len(fitness) == 0:
            return np.nan, np.zeros(self.dim)
        
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()
        
        while not stopping_condition():
            # Sample trial population
            trials = self._sample_trials_batch(len(population))
            trials = self._clip_to_bounds(trials)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if len(trial_fitness) == 0:
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness)
            
            # Track best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < f_opt:
                f_opt = fitness[best_idx]
                x_opt = population[best_idx].copy()
            
            # Compute selection weights for covariance update
            selection_weights = self._compute_selection_weights(fitness)
            
            # Select top portion for covariance update
            n_selected = max(int(len(population) * 0.5), 2)
            selected_indices = np.argsort(fitness)[:n_selected]
            selected_pop = population[selected_indices]
            selected_weights = selection_weights[selected_indices]
            selected_weights = selected_weights / np.sum(selected_weights)
            
            # Update covariance and step size
            self._update_covariance(selected_pop, selected_weights)
            
            # Compute recent success rate for step size adaptation
            n_success = np.sum(trial_fitness < fitness[best_idx])
            recent_success_rate = n_success / len(trial_fitness)
            self._adapt_step_size(recent_success_rate)
            
            # Check for stagnation
            self._restart_if_stagnant(f_opt)
        
        return f_opt, x_opt
```