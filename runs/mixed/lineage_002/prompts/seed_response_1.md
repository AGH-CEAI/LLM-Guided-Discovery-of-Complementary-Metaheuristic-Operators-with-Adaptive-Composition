```python
import numpy as np


class CulturalDifferentialEvolutionWithVelocity:
    """
    Cultural Differential Evolution with Velocity (CDE-V)
    
    A DE variant combining:
    - Current-to-pbest mutation with archive (JADE-style)
    - Velocity-guided perturbation (PSO-inspired)
    - Cultural memory for knowledge-guided search
    - Adaptive F and CR parameters per individual
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(200, max(4 * dim, 120))  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Adaptive parameters
        self.c = 0.1  # Learning rate for F/CR adaptation
        self.p_best_rate = 0.05  # Top p% for pbest selection
        self.arc_rate = 0.5  # Archive size relative to NP
        self.velocity_weight = 0.3  # Inertia weight for velocity
        
        # Diversity thresholds
        self.diversity_threshold = 1e-6
        self.max_stagnation = 50
        
        # Internal state
        self.population = None
        self.fitness = None
        self.velocity = None
        self.archive = None
        self.f_history = None
        self.cultural_memory = None
        
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self._initialize_velocity()
        self._initialize_cultural_memory()
        
        fitness = func(self.population)
        fitness = self._handle_budget_exhaustion(fitness, self.population)
        
        f_opt = np.min(fitness)
        x_opt = self.population[np.argmin(fitness)].copy()
        stagnation_counter = 0
        
        while not stopping_condition():
            # Generate trial population (vectorized)
            trials, f_values, cr_values = self._generate_trials_batch()
            
            if stopping_condition():
                break
                
            # Evaluate trials in batch
            trial_fitness = func(trials)
            trial_fitness = self._handle_budget_exhaustion(trial_fitness, trials)
            
            if len(trial_fitness) == 0:
                break
            
            # Select survivors
            self.population, fitness, self.velocity = self._select_survivors_batch(
                self.population, fitness, trials, trial_fitness, self.velocity
            )
            
            # Update archive
            self._update_archive_batch(trials, trial_fitness)
            
            # Update cultural memory
            self._update_cultural_memory_batch(fitness)
            
            # Adapt parameters
            self._adapt_parameters_batch(f_values, cr_values, trial_fitness, fitness)
            
            # Check convergence
            current_best = np.min(fitness)
            if current_best < f_opt:
                f_opt = current_best
                x_opt = self.population[np.argmin(fitness)].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Restart if stagnant
            if stagnation_counter >= self.max_stagnation:
                self._restart_if_stagnant(fitness)
                stagnation_counter = 0
            
            if stopping_condition():
                break
                
        return f_opt, x_opt
    
    def _initialize_population(self):
        """Initialize population with Latin Hypercube Sampling."""
        self.population = np.random.uniform(self.lower, self.upper, (self.np, self.dim))
        self._apply_lhs_sampling()
        self._clip_to_bounds_batch()
        
    def _apply_lhs_sampling(self):
        """Apply LHS for better initial distribution."""
        grid = np.linspace(self.lower, self.upper, self.np + 1)
        for d in range(self.dim):
            offsets = np.random.uniform(0, 1, self.np) * (grid[1] - grid[0])
            self.population[:, d] = grid[:-1] + offsets
            np.random.shuffle(self.population[:, d])
    
    def _initialize_velocity(self):
        """Initialize velocity vectors for momentum-based search."""
        range_val = (self.upper - self.lower) * 0.1
        self.velocity = np.random.uniform(-range_val, range_val, (self.np, self.dim))
    
    def _initialize_cultural_memory(self):
        """Initialize cultural memory storing best positions per region."""
        self.cultural_memory = {
            'positions': np.zeros((4, self.dim)),  # 4 cultural best positions
            'fitness': np.full(4, np.inf),
            'ages': np.zeros(4)
        }
        # Initialize with diverse points
        for i in range(4):
            self.cultural_memory['positions'][i] = np.random.uniform(
                self.lower, self.upper, self.dim
            )
    
    def _clip_to_bounds_batch(self):
        """Clip entire population to search bounds."""
        self.population = np.clip(self.population, self.lower, self.upper)
    
    def _generate_trials_batch(self):
        """
        Generate trial vectors using current-to-pbest mutation with velocity.
        Returns trials, F_values, CR_values for each trial.
        """
        n_trials = self.np
        trials = np.empty((n_trials, self.dim))
        f_values = np.empty(n_trials)
        cr_values = np.empty(n_trials)
        
        # Compute pbest indices
        p_best_indices = self._get_pbest_indices()
        
        # Get current-to-pbest donors
        donors = self._compute_current_to_pbest_donors(p_best_indices)
        
        # Generate random indices for archive/base/mutant
        arc_size = len(self.archive) if len(self.archive) > 0 else self.np
        archive_extended = np.vstack([self.population, self.archive]) if arc_size > 0 else self.population
        
        for i in range(n_trials):
            # Adaptive F with velocity influence
            f_base = np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0)
            velocity_factor = np.linalg.norm(self.velocity[i]) / (self.dim * 10 + 1e-10)
            f_values[i] = f_base * (1 + self.velocity_weight * velocity_factor)
            
            # Adaptive CR
            cr_values[i] = np.clip(np.random.normal(0.5, 0.3), 0.0, 1.0)
            
            # Mutation with velocity perturbation
            r1, r2 = self._sample_different_indices(i, 2, arc_size)
            mutant = self.population[i] + f_values[i] * (
                donors[i] - self.population[i]
            ) + f_values[i] * (
                self.population[r1] - archive_extended[r2]
            ) + self.velocity_weight * self.velocity[i]
            
            # Crossover
            j_rand = np.random.randint(self.dim)
            cross_mask = np.random.random(self.dim) < cr_values[i]
            cross_mask[j_rand] = True
            
            trials[i] = np.where(cross_mask, mutant, self.population[i])
        
        # Clip trials to bounds
        trials = np.clip(trials, self.lower, self.upper)
        
        return trials, f_values, cr_values
    
    def _get_pbest_indices(self):
        """Get indices of top p% best individuals for pbest mutation."""
        sorted_indices = np.argsort(self.fitness)
        p_size = max(1, int(self.np * self.p_best_rate))
        pbest_pool = sorted_indices[:p_size]
        return np.random.choice(pbest_pool, size=self.np, replace=True)
    
    def _compute_current_to_pbest_donors(self, pbest_indices):
        """Compute current-to-pbest donor vectors."""
        donors = np.empty((self.np, self.dim))
        for i in range(self.np):
            donors[i] = self.population[pbest_indices[i]]
        return donors
    
    def _sample_different_indices(self, exclude, count, pool_size):
        """Sample count different indices excluding 'exclude'."""
        indices = np.random.choice(
            [i for i in range(pool_size) if i != exclude],
            size=count,
            replace=False
        )
        return indices
    
    def _select_survivors_batch(self, pop, fit, trials, trial_fit, velocity):
        """
        Select survivors using greedy selection.
        Also update velocity based on successful mutations.
        """
        new_pop = np.empty_like(pop)
        new_fit = np.empty_like(fit)
        new_velocity = np.empty_like(velocity)
        
        improvements = trial_fit < fit
        
        for i in range(len(pop)):
            if improvements[i]:
                new_pop[i] = trials[i]
                new_fit[i] = trial_fit[i]
                # Update velocity: v = w*v + (x_trial - x_old)
                new_velocity[i] = (
                    self.velocity_weight * velocity[i] + 
                    (trials[i] - pop[i])
                )
            else:
                new_pop[i] = pop[i]
                new_fit[i] = fit[i]
                # Decay velocity
                new_velocity[i] = self.velocity_weight * velocity[i]
        
        return new_pop, new_fit, new_velocity
    
    def _update_archive_batch(self, trials, trial_fitness):
        """Update archive with improved solutions."""
        if self.archive is None:
            self.archive = np.empty((0, self.dim))
        
        improvements = trial_fitness < self.fitness
        improved_solutions = trials[improvements]
        
        if len(improved_solutions) > 0:
            self.archive = np.vstack([self.archive, improved_solutions])
            
            # Limit archive size
            max_archive_size = int(self.np * self.arc_rate)
            if len(self.archive) > max_archive_size:
                # Remove random excess
                drop_indices = np.random.choice(
                    len(self.archive), 
                    len(self.archive) - max_archive_size, 
                    replace=False
                )
                self.archive = np.delete(self.archive, drop_indices, axis=0)
    
    def _update_cultural_memory_batch(self, fitness):
        """Update cultural memory with best solutions from different regions."""
        best_idx = np.argmin(fitness)
        best_pos = self.population[best_idx]
        best_fit = fitness[best_idx]
        
        # Update ages
        self.cultural_memory['ages'] += 1
        
        # Find worst cultural memory slot
        worst_cultural = np.argmax(self.cultural_memory['fitness'])
        
        # Replace if new solution is significantly better
        if best_fit < self.cultural_memory['fitness'][worst_cultural] * 0.9:
            self.cultural_memory['positions'][worst_cultural] = best_pos.copy()
            self.cultural_memory['fitness'][worst_cultural] = best_fit
            self.cultural_memory['ages'][worst_cultural] = 0
    
    def _adapt_parameters_batch(self, f_values, cr_values, trial_fitness, fitness):
        """Adapt F and CR based on success rates."""
        # This method prepares next generation's parameters
        # The actual adaptation happens during trial generation
        improvements = trial_fitness < fitness
        if np.sum(improvements) > 0:
            success_f = f_values[improvements]
            success_cr = cr_values[improvements]
            # Update base F/CR for next generation (stored in instance)
            self._f_mean = np.mean(success_f) if len(success_f) > 0 else 0.5
            self._cr_mean = np.mean(success_cr) if len(success_cr) > 0 else 0.5
    
    def _restart_if_stagnant(self, fitness):
        """Restart population when convergence is detected."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold:
            # Reinitialize worst half of population
            sorted_indices = np.argsort(fitness)
            worst_half = sorted_indices[self.np // 2:]
            
            # Generate new solutions using cultural memory guidance
            for idx in worst_half:
                # Bias toward cultural memory with some mutation
                cultural_idx = np.random.randint(4)
                base = self.cultural_memory['positions'][cultural_idx].copy()
                mutation = np.random.normal(0, (self.upper - self.lower) * 0.1, self.dim)
                self.population[idx] = base + mutation
            
            # Reset velocity for restarted individuals
            self.velocity[worst_half] *= 0.1
            
            # Re-evaluate
            new_fitness = fitness.copy()
            new_fitness[worst_half] = np.inf  # Mark for re-evaluation
            
            # Note: caller should re-evaluate fitness for these indices
            self.fitness = new_fitness
    
    def _compute_diversity(self):
        """Compute population diversity using average pairwise distance."""
        if self.population is None or len(self.population) < 2:
            return 1.0
        
        # Compute centroid
        centroid = np.mean(self.population, axis=0)
        
        # Compute average distance to centroid
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _handle_budget_exhaustion(self, fitness, population):
        """Handle cases where budget is exhausted and fitness returns fewer values."""
        if len(fitness) < len(population):
            # Truncate population to match fitness
            self.population = self.population[:len(fitness)]
            self.velocity = self.velocity[:len(fitness)]
            self.np = len(fitness)
        return fitness
```