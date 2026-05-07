import numpy as np


class CovarianceRankEvolutionStrategy:
    """
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with an adaptive archive and rank-based (mu, lambda) selection.
    
    Key differences from AdaptiveArchiveDE:
    - Uses Cholesky-decomposed covariance matrix for mutation directions
    - Rank-based linear selection instead of pure elitism
    - Adaptive mutation strategy probability based on success history
    - Covariance adaptation from centroid displacement (simplified CMA-ES)
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(50, 5 * dim))
        self.bounds = np.array([-100.0, 100.0])
        self.archive_max_size = 2 * self.np
        
        self.step_size = (self.bounds[1] - self.bounds[0]) * 0.05
        self.cov_prob = 0.3
        self.CR = 0.85
        self.F = 0.7
        
        self.cov_adapt_rate = 0.1
        self.max_stagnation = 150
        self.min_diversity = 1e-4 * self.dim
        
        self.population = None
        self.fitness = None
        self.best_fitness = np.inf
        self.best_position = None
        self.stagnation_counter = 0
        self.last_improvement = 0
        self.success_history = []
        self.archive = np.zeros((0, dim))
        self.archive_fitness = np.array([])
        self.cov = np.eye(dim)
        self.L = np.eye(dim)
        
    def _initialize_population(self) -> np.ndarray:
        pop = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            size=(self.np, self.dim)
        )
        return pop
    
    def _update_covariance_batch(self, pop: np.ndarray):
        """Update covariance matrix using simplified CMA-ES update"""
        centroid = pop.mean(axis=0)
        centered = pop - centroid
        pop_cov = np.cov(centered.T, bias=True)
        self.cov = ((1 - self.cov_adapt_rate) * self.cov + 
                    self.cov_adapt_rate * pop_cov)
        eigvals = np.linalg.eigvalsh(self.cov)
        if np.any(eigvals <= 0):
            self.cov = np.eye(self.dim)
        self.L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(self.dim))
    
    def _update_archive_batch(self, pop: np.ndarray, fit: np.ndarray):
        """Maintain archive of good solutions"""
        if len(self.archive) == 0:
            self.archive = pop.copy()
            self.archive_fitness = fit.copy()
            return
        
        combined_pop = np.vstack([self.archive, pop])
        combined_fit = np.concatenate([self.archive_fitness, fit])
        top_indices = np.argsort(combined_fit)[:self.archive_max_size]
        
        self.archive = combined_pop[top_indices]
        self.archive_fitness = combined_fit[top_indices]
    
    def _generate_mutation_indices_batch(self) -> np.ndarray:
        """Generate indices for archive-based mutation"""
        n_arc = len(self.archive)
        arc_indices = np.random.randint(0, max(1, n_arc), size=(self.np, 2))
        pop_indices = np.random.randint(0, self.np, size=(self.np, 2))
        return arc_indices, pop_indices
    
    def _compute_archive_mutation_batch(self, arc_idx: np.ndarray, 
                                        pop_idx: np.ndarray) -> np.ndarray:
        """Compute archive-guided mutation vectors"""
        r1 = self.archive[arc_idx[:, 0]]
        r2 = self.archive[arc_idx[:, 1]]
        archive_mutant = r1 + self.F * (r2 - r1)
        
        p1 = self.population[pop_idx[:, 0]]
        p2 = self.population[pop_idx[:, 1]]
        pop_mutant = p1 + self.F * (p2 - p1)
        
        return (archive_mutant + pop_mutant) / 2.0
    
    def _compute_covariance_mutation_batch(self) -> np.ndarray:
        """Compute covariance-guided mutation using Cholesky decomposition"""
        z = np.random.randn(self.np, self.dim)
        cov_mutant = self.population + self.step_size * (z @ self.L.T)
        return cov_mutant
    
    def _mutate_batch(self) -> np.ndarray:
        """Generate trial population using hybrid mutation"""
        archive_mutant = self._compute_archive_mutation_batch(*self._generate_mutation_indices_batch())
        cov_mutant = self._compute_covariance_mutation_batch()
        
        use_cov = np.random.random(self.np) < self.cov_prob
        mutant = np.where(use_cov[:, np.newaxis], cov_mutant, archive_mutant)
        
        cross_mask = np.random.random((self.np, self.dim)) < self.CR
        if not np.any(cross_mask):
            cross_mask[np.arange(self.np), np.random.randint(0, self.dim, self.np)] = True
        
        trial = np.where(cross_mask, mutant, self.population)
        return np.clip(trial, self.bounds[0], self.bounds[1])
    
    def _evaluate_batch(self, pop: np.ndarray, func) -> np.ndarray:
        """Evaluate fitness for entire population"""
        return func(pop)
    
    def _select_survivors_batch(self, current_pop: np.ndarray, 
                                current_fit: np.ndarray,
                                trial_pop: np.ndarray, 
                                trial_fit: np.ndarray) -> tuple:
        """Rank-based (mu, lambda) selection with elitism"""
        combined_pop = np.vstack([current_pop, trial_pop])
        combined_fit = np.concatenate([current_fit, trial_fit])
        
        best_idx = np.argmin(combined_fit)
        best_pos = combined_pop[best_idx]
        best_fit = combined_fit[best_idx]
        
        ranks = np.argsort(np.argsort(combined_fit))
        n = len(combined_fit)
        eta_plus = 1.5
        
        probs = (2.0 * (eta_plus + 1) - 2.0 * ranks * eta_plus / (n - 1)) / n
        probs = np.maximum(probs, 1e-10)
        probs = probs / probs.sum()
        
        n_select = self.np - 1
        survivor_indices = np.random.choice(n, size=n_select, replace=True, p=probs)
        
        new_pop = np.vstack([best_pos, combined_pop[survivor_indices]])
        new_fit = np.concatenate([[best_fit], combined_fit[survivor_indices]])
        
        return new_pop, new_fit
    
    def _adapt_parameters(self, success_rate: float):
        """Adapt step size and mutation probability based on success rate"""
        if len(self.success_history) >= 5:
            recent_rate = np.mean(self.success_history[-5:])
            if recent_rate > 0.25:
                self.step_size *= 1.15
                self.cov_prob = min(0.6, self.cov_prob + 0.02)
            elif recent_rate < 0.1:
                self.step_size *= 0.85
                self.cov_prob = max(0.1, self.cov_prob - 0.02)
        
        self.step_size = np.clip(self.step_size, 1e-4, 10.0)
        self.F = np.clip(self.F + np.random.randn() * 0.05, 0.3, 1.5)
    
    def _compute_diversity(self, pop: np.ndarray) -> float:
        """Compute population diversity as average distance to centroid"""
        centroid = pop.mean(axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_if_stagnant(self) -> bool:
        """Check if restart is needed"""
        if self.stagnation_counter > self.max_stagnation:
            return True
        if self._compute_diversity(self.population) < self.min_diversity:
            return True
        return False
    
    def _restart(self):
        """Reinitialize population while preserving best solution"""
        old_best = self.best_position.copy()
        old_best_fit = self.best_fitness
        
        self.population = self._initialize_population()
        self.fitness = np.full(self.np, np.inf)
        self.population[0] = old_best
        self.fitness[0] = old_best_fit
        
        self.stagnation_counter = 0
        self.success_history = []
        self.cov = np.eye(self.dim)
        self.L = np.eye(self.dim)
        self.step_size = (self.bounds[1] - self.bounds[0]) * 0.05
    
    def __call__(self, func, stopping_condition) -> tuple:
        self.population = self._initialize_population()
        self.fitness = self._evaluate_batch(self.population, func)
        self.best_fitness = float(np.min(self.fitness))
        best_idx = np.argmin(self.fitness)
        self.best_position = self.population[best_idx].copy()
        
        iteration = 0
        while not stopping_condition():
            trials = self._mutate_batch()
            trial_fitness = self._evaluate_batch(trials, func)
            
            if len(trial_fitness) < len(trials):
                break
            
            if stopping_condition():
                break
            
            self._update_covariance_batch(self.population)
            self._update_archive_batch(trials, trial_fitness)
            
            old_best = self.best_fitness
            if trial_fitness.min() < self.best_fitness:
                self.best_fitness = trial_fitness.min()
                best_idx = np.argmin(trial_fitness)
                self.best_position = trials[best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            success = 1.0 if self.best_fitness < old_best else 0.0
            self.success_history.append(success)
            if len(self.success_history) > 50:
                self.success_history = self.success_history[-50:]
            
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            success_rate = np.mean(self.success_history[-20:]) if self.success_history else 0.5
            self._adapt_parameters(success_rate)
            
            if self._restart_if_stagnant():
                self._restart()
            
            iteration += 1
        
        return self.best_fitness, self.best_position
