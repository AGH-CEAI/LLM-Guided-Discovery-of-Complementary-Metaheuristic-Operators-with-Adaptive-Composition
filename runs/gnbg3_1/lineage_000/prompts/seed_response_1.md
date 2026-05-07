```python
import numpy as np


class AdaptiveReflectiveDE:
    """
    Adaptive Reflective Differential Evolution.
    
    Key innovations:
    - Orthogonal reflection mutation: mutants are reflected around a p-best
      direction using orthogonal perturbations derived from population structure.
    - Topology-aware mutation: each target selects random neighbors to ensure
      structured diversity.
    - Adaptive F and CR with momentum: parameters adapt based on cumulative
      success history with dampening.
    - Novelty-triggered restart: re-seeds around best when diversity collapses.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 20), 10 * dim)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        self.diameter = self.upper - self.lower
        
        # Adaptive parameters
        self.f = 0.5 + 0.1 * np.random.rand()
        self.cr = 0.5 + 0.1 * np.random.rand()
        self.f_damp = 0.9
        self.cr_damp = 0.1
        
        # Memory for adaptive parameters
        self.f_memory = np.full(5, 0.5)
        self.cr_memory = np.full(5, 0.5)
        self.memory_idx = 0
        
        # Stagnation tracking
        self.stagnation_window = 15
        self.min_improvement = 1e-8
        
    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling for space-filling."""
        sampler = np.linspace(0, 1, self.np)
        population = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            population[:, d] = self.lower + (sampler[perm] + 
                                np.random.rand(self.np) / self.np) * self.diameter
        return np.clip(population, self.lower, self.upper)
    
    def _select_random_indices_batch(self, exclude, count):
        """Select random indices excluding specified ones, batched."""
        valid_pool = np.concatenate([np.arange(0, exclude), 
                                     np.arange(exclude + 1, self.np)])
        selected = np.random.choice(valid_pool, size=count, replace=False)
        return selected
    
    def _compute_orthogonal_direction(self, v1, v2):
        """Compute a direction orthogonal to v1 using cross-product strategy."""
        # Create a vector orthogonal to v1 by cross-product with standard basis
        # or with v2 if v1 and e_i are nearly parallel
        eps = 1e-10
        v1_norm = v1 / (np.linalg.norm(v1) + eps)
        
        # Try standard basis vectors
        for i in range(min(3, self.dim)):
            e_i = np.zeros(self.dim)
            e_i[i] = 1.0
            cross = np.cross(v1_norm, e_i)
            if np.linalg.norm(cross) > 0.1:
                return cross / (np.linalg.norm(cross) + eps)
        
        # Fallback: use v2 to construct orthogonal direction
        if v2 is not None:
            v2_proj = v2 - np.dot(v2, v1_norm) * v1_norm
            if np.linalg.norm(v2_proj) > 0.1:
                return v2_proj / (np.linalg.norm(v2_proj) + eps)
        
        # Last resort: random orthogonal direction
        random_vec = np.random.randn(self.dim)
        random_vec -= np.dot(random_vec, v1_norm) * v1_norm
        return random_vec / (np.linalg.norm(random_vec) + eps)
    
    def _mutate_batch(self, population, fitness):
        """
        Current-to-pbest-with-orthogonal-reflection mutation.
        Each mutant is reflected around the p-best direction using orthogonal perturbations.
        """
        mutants = np.empty((self.np, self.dim))
        
        # Determine p-best set (top 20% by fitness)
        sorted_indices = np.argsort(fitness)
        p_best_count = max(1, self.np // 5)
        p_best_indices = sorted_indices[:p_best_count]
        
        for i in range(self.np):
            # Select base vector (current individual)
            target_idx = i
            
            # Select three distinct random indices for structure
            others = self._select_random_indices_batch(target_idx, 3)
            r1, r2, r3 = others[0], others[1], others[2]
            
            # Select p-best from elite set
            p_best_idx = np.random.choice(p_best_indices)
            
            # Current-to-pbest direction
            base = population[target_idx]
            p_best = population[p_best_idx]
            direction = p_best - base
            
            # Compute orthogonal perturbation using r1 and r2
            ortho_dir = self._compute_orthogonal_direction(
                population[r1] - base,
                population[r2] - base
            )
            
            # Adaptive reflection coefficient
            reflect_coeff = self.f * np.random.uniform(-1.0, 1.0)
            
            # Orthogonal perturbation magnitude
            ortho_scale = self.f * np.random.uniform(0.0, 1.0) * np.linalg.norm(direction)
            
            # Construct mutant with reflection
            mutants[i] = base + direction + reflect_coeff * direction + ortho_scale * ortho_dir
            
        return mutants
    
    def _crossover_batch(self, targets, mutants):
        """
        Binomial crossover with adaptive rate and forced dimension mixing.
        """
        trials = np.copy(targets)
        
        # Random crossover points for each individual
        n_dim = mutants.shape[1]
        force_dims = np.random.randint(0, n_dim, size=self.np)
        
        # Generate crossover masks
        rand_mask = np.random.rand(self.np, n_dim) < self.cr
        
        # Ensure forced dimension is always crossed
        force_mask = np.zeros((self.np, n_dim), dtype=bool)
        force_mask[np.arange(self.np), force_dims] = True
        
        # Combined mask
        crossover_mask = rand_mask | force_mask
        
        trials[crossover_mask] = mutants[crossover_mask]
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Greedy (μ+λ) selection based on fitness values.
        """
        improved = trial_fitness < fitness
        population = np.copy(population)
        
        # Replace improved individuals
        population[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]
        
        return population, fitness
    
    def _update_best_tracking(self, fitness, population):
        """Track the best fitness and solution found."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return None, None, np.nan
        
        min_idx = np.nanargmin(fitness)
        return fitness[min_idx], population[min_idx].copy(), min_idx
    
    def _adapt_parameters(self, improved_mask, trial_fitness, fitness):
        """
        Adapt F and CR based on success history with momentum.
        """
        if not np.any(improved_mask):
            # No improvements: increase exploration
            self.f = min(1.0, self.f * 1.1)
            self.cr = max(0.0, self.cr * 0.9)
            return
        
        # Compute success rate
        success_rate = np.mean(improved_mask)
        
        # Weighted mean of successful F values (smaller F for exploitation)
        f_values = self.f * (1.0 + np.random.rand(np.sum(improved_mask)) * 0.5 - 0.25)
        f_successful = np.mean(f_values)
        
        # CR adaptation toward successful values
        cr_successful = np.mean(trial_fitness[improved_mask] < fitness[improved_mask])
        
        # Update with dampening
        self.f = self.f_damp * self.f + (1 - self.f_damp) * f_successful
        self.f = np.clip(self.f, 0.1, 1.5)
        
        self.cr = self.cr_damp * self.cr + (1 - self.cr_damp) * cr_successful
        self.cr = np.clip(self.cr, 0.0, 1.0)
    
    def _compute_diversity(self, population):
        """Compute population diversity as average distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances) / (self.diameter * np.sqrt(self.dim) / 2)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_pos):
        """
        Reinitialize population around current best when stagnant.
        Uses Cauchy-distributed perturbation for long-tailed exploration.
        """
        diversity = self._compute_diversity(population)
        
        if diversity < 0.05:  # Diversity threshold
            # Re-seed around best with Cauchy perturbation
            new_population = np.tile(best_pos, (self.np, 1))
            
            # Cauchy distribution for heavy-tailed perturbation
            scale = 0.5 * self.diameter * (1.0 - diversity)
            for d in range(self.dim):
                new_population[:, d] += scale * np.random.standard_cauchy(self.np)
            
            population = np.clip(new_population, self.lower, self.upper)
            
            # Re-evaluate
            fitness = np.full(self.np, np.nan)
            
            return population, fitness, True
        
        return population, fitness, False
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness.
            stopping_condition: Callable returning True when to stop.
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution.
        """
        # Initialize
        population = self._initialize_population()
        fitness = np.full(self.np, np.nan)
        
        best_fitness = np.inf
        best_pos = None
        generation = 0
        
        # Initial evaluation
        fitness = func(population)
        if len(fitness) < self.np:
            fitness = np.pad(fitness, (0, self.np - len(fitness)), 
                           constant_values=np.nan)
        
        f_opt, x_opt, _ = self._update_best_tracking(fitness, population)
        if f_opt is not None:
            best_fitness, best_pos = f_opt, x_opt
        
        # Main loop
        while not stopping_condition():
            generation += 1
            
            # Generate mutants using orthogonal reflection
            mutants = self._mutate_batch(population, fitness)
            
            # Binomial crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip to bounds before evaluation
            trials = np.clip(trials, self.lower, self.upper)
            
            # Evaluate trial population
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            n_valid = len(trial_fitness)
            if n_valid < self.np:
                trials = trials[:n_valid]
                trial_fitness = np.pad(trial_fitness, 
                                      (0, self.np - n_valid), 
                                      constant_values=np.nan)
            
            # Check stopping after evaluation
            if stopping_condition():
                break
            
            # Selection
            improved_mask = trial_fitness < fitness
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            f_opt, x_opt, _ = self._update_best_tracking(fitness, population)
            if f_opt is not None and f_opt < best_fitness:
                best_fitness, best_pos = f_opt, x_opt
            
            # Adapt mutation parameters
            self._adapt_parameters(improved_mask, trial_fitness, fitness)
            
            # Check for stagnation and restart if needed
            population, fitness, did_restart = self._restart_if_stagnant(
                population, fitness, best_fitness, best_pos
            )
            
            if did_restart:
                # Re-evaluate after restart
                fitness = func(population)
                if len(fitness) < self.np:
                    fitness = np.pad(fitness, (0, self.np - len(fitness)),
                                   constant_values=np.nan)
                
                if stopping_condition():
                    break
        
        return best_fitness, best_pos
```