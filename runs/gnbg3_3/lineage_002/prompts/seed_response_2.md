```python
import numpy as np


class CovarianceEnhancedParticleSwarm:
    """
    Covariance-Enhanced Particle Swarm Optimizer (CE-PSO).
    
    Combines PSO velocity updates with a simplified CMA-ES-style covariance
    adaptation for the sampling distribution. Features adaptive inertia weight,
    covariance-based mutation, and restart on stagnation.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 10 * dim)  # Population size: 60-300 for dim=30
        self.bounds = np.array([-100.0, 100.0])
        self._init_hyperparameters()
        
    def _init_hyperparameters(self):
        """Initialize algorithm hyperparameters."""
        self.w_max = 0.9
        self.w_min = 0.4
        self.cognitive = 1.496
        self.social = 1.496
        self.cov_learning_rate = 0.05
        self.cov_decay = 0.99
        self.diversity_threshold = 1e-6
        self.stagnation_limit = 50
        self.elite_ratio = 0.2
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        pop = self._initialize_population()
        fitness = self._evaluate_batch(func, pop)
        self._update_best(pop, fitness)
        
        mean = pop.mean(axis=0)
        cov = self._init_covariance_matrix()
        step_size = self._init_step_size()
        
        stagnation_counter = 0
        generation = 0
        
        while not stopping_condition():
            # Adapt inertia weight based on generation and diversity
            w = self._adapt_inertia_weight(generation, self._compute_diversity(pop))
            
            # Compute cognitive (personal best attractor) influence
            cognitive_attractor = self._compute_cognitive_attractor_batch(pop)
            
            # Compute social (global best) influence
            social_attractor = self._compute_social_attractor()
            
            # Update velocities with covariance-enhanced mutation
            velocities = self._velocity_update_batch(
                pop, velocities=None, w=w,
                cognitive_attractor=cognitive_attractor,
                social_attractor=social_attractor,
                cov=cov, step_size=step_size
            )
            
            # Update positions
            pop = self._position_update_batch(pop, velocities)
            
            # Evaluate batched fitness
            if stopping_condition():
                break
            fitness = self._evaluate_batch(func, pop)
            if len(fitness) < self.np:
                fitness = self._handle_truncated_evaluation(func, pop, fitness)
                if len(fitness) < self.np:
                    break
                    
            # Update personal and global bests
            self._update_best(pop, fitness)
            
            # Adapt covariance matrix (simplified CMA-ES)
            cov, step_size = self._adapt_covariance(
                pop, fitness, mean, cov, step_size
            )
            mean = pop.mean(axis=0)
            
            # Check for stagnation and handle restart if needed
            stagnation_counter = self._check_stagnation(fitness, stagnation_counter)
            if stagnation_counter >= self.stagnation_limit:
                pop, velocities, cov, step_size, mean = self._restart_if_stagnant(
                    pop, fitness, cov, step_size
                )
                stagnation_counter = 0
                
            generation += 1
            
        return self.best_fitness, self.best_position
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        pop = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            size=(self.np, self.dim)
        )
        return pop
    
    def _init_covariance_matrix(self):
        """Initialize diagonal covariance matrix."""
        sigma0 = (self.bounds[1] - self.bounds[0]) / 6.0
        return np.eye(self.dim) * (sigma0 ** 2)
    
    def _init_step_size(self):
        """Initialize step size for covariance sampling."""
        return (self.bounds[1] - self.bounds[0]) / 6.0
    
    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population at once."""
        clipped = self._clip_to_bounds(population)
        return func(clipped)
    
    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.bounds[0], self.bounds[1])
    
    def _handle_truncated_evaluation(self, func, population, fitness):
        """Handle case where func returns fewer values than expected."""
        valid_count = len(fitness)
        if valid_count == 0:
            return fitness
        # Re-evaluate only valid portion
        valid_pop = population[:valid_count]
        valid_fit = self._evaluate_batch(func, valid_pop)
        return valid_fit
    
    def _update_best(self, population, fitness):
        """Track global best solution."""
        if not hasattr(self, 'best_fitness') or np.nanmin(fitness) < self.best_fitness:
            best_idx = np.nanargmin(fitness)
            self.best_fitness = fitness[best_idx]
            self.best_position = population[best_idx].copy()
    
    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self, generation, diversity):
        """Adapt inertia weight based on generation and diversity."""
        w_range = self.w_max - self.w_min
        w = self.w_max - (w_range * generation / 200.0)
        w = np.clip(w, self.w_min, self.w_max)
        
        # Boost exploration if diversity is low
        if diversity < 10.0:
            w = min(w + 0.1, self.w_max)
        return w
    
    def _compute_cognitive_attractor_batch(self, population):
        """Compute per-particle personal best attractors."""
        if not hasattr(self, 'personal_best'):
            self.personal_best = population.copy()
            self.personal_best_fitness = np.full(self.np, np.inf)
            
        return self.personal_best
    
    def _compute_social_attractor(self):
        """Compute global best attractor."""
        return self.best_position
    
    def _velocity_update_batch(self, population, velocities, w, 
                                cognitive_attractor, social_attractor,
                                cov, step_size):
        """Update velocities with cognitive, social, and covariance components."""
        if velocities is None:
            velocities = np.zeros_like(population)
            
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r3 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        cognitive = self.cognitive * r1 * (cognitive_attractor - population)
        social = self.social * r2 * (social_attractor - population)
        
        # Covariance-based mutation term
        cov_mut = self._sample_from_covariance(cov, step_size)
        
        velocities = w * velocities + cognitive + social + r3 * cov_mut * 0.5
        
        # Velocity clamping
        vmax = (self.bounds[1] - self.bounds[0]) * 0.2
        velocities = np.clip(velocities, -vmax, vmax)
        
        return velocities
    
    def _sample_from_covariance(self, cov, step_size):
        """Sample mutation from current covariance distribution."""
        # Use diagonal approximation for efficiency
        diag_cov = np.diag(cov)
        diag_cov = np.maximum(diag_cov, 1e-10)
        samples = np.random.normal(0, np.sqrt(diag_cov) * step_size, size=(self.np, self.dim))
        return samples
    
    def _position_update_batch(self, population, velocities):
        """Update positions using velocities."""
        new_pop = population + velocities
        return self._clip_to_bounds(new_pop)
    
    def _adapt_covariance(self, population, fitness, mean, cov, step_size):
        """Simplified CMA-ES style covariance adaptation."""
        # Select elite individuals
        n_elite = max(int(self.np * self.elite_ratio), 1)
        elite_indices = np.argsort(fitness)[:n_elite]
        elite_pop = population[elite_indices]
        
        # Update mean estimate
        new_mean = elite_pop.mean(axis=0)
        
        # Update covariance using rank-1 and rank-μ updates
        diff = new_mean - mean
        rank1 = np.outer(diff, diff)
        
        # Rank-μ update: variance of elite population
        diffs_from_mean = elite_pop - new_mean
        rank_mu = np.dot(diffs_from_mean.T, diffs_from_mean) / n_elite
        
        # Combine updates
        cov = (1 - self.cov_learning_rate) * cov * self.cov_decay + \
              self.cov_learning_rate * (0.5 * rank1 + 0.5 * rank_mu)
              
        # Ensure positive definiteness
        cov = self._ensure_positive_definite(cov)
        
        # Adapt step size based on success
        step_size = self._adapt_step_size(fitness, cov, step_size)
        
        return cov, step_size
    
    def _ensure_positive_definite(self, matrix):
        """Ensure matrix is positive definite."""
        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals <= 0):
            # Add small regularization
            matrix = matrix + np.eye(self.dim) * abs(np.min(eigvals)) + 1e-8
        return matrix
    
    def _adapt_step_size(self, fitness, cov, step_size):
        """Adapt step size based on covariance eigenvalues."""
        eigvals = np.linalg.eigvalsh(cov)
        avg_eig = np.mean(eigvals)
        step_size = step_size * 0.99 + 0.01 * np.sqrt(avg_eig)
        step_size = np.clip(step_size, 0.1, 50.0)
        return step_size
    
    def _check_stagnation(self, fitness, counter):
        """Check if optimization has stagnated."""
        if not hasattr(self, 'prev_fitness'):
            self.prev_fitness = fitness.copy()
            return 0
            
        improvement = np.nanmin(self.prev_fitness) - np.nanmin(fitness)
        self.prev_fitness = fitness.copy()
        
        if improvement > 1e-6:
            return 0
        return counter + 1
    
    def _restart_if_stagnant(self, population, fitness, cov, step_size):
        """Reinitialize portion of population on stagnation."""
        n_restart = max(self.np // 2, 1)
        restart_indices = np.argsort(fitness)[-n_restart:]
        
        # Keep best individuals
        new_pop = population.copy()
        new_pop[restart_indices] = self._initialize_population()[restart_indices]
        
        # Reinitialize velocities for restarted individuals
        velocities = np.zeros_like(new_pop)
        
        # Reset covariance toward identity
        new_cov = self._init_covariance_matrix()
        new_step = self._init_step_size()
        new_mean = new_pop.mean(axis=0)
        
        return new_pop, velocities, new_cov, new_step, new_mean
    
    def _select_survivors_batch(self, old_pop, old_fitness, new_pop, new_fitness):
        """Select survivors between old and new populations."""
        combined_pop = np.vstack([old_pop, new_pop])
        combined_fit = np.concatenate([old_fitness, new_fitness])
        
        # Keep best NP individuals
        selected_indices = np.argsort(combined_fit)[:self.np]
        return combined_pop[selected_indices], combined_fit[selected_indices]
```