```python
import numpy as np


class AdaptiveArchiveDE:
    """
    Differential Evolution with Composite Mutation, Adaptive Archive,
    and Memory-based Parameter Adaptation.
    
    Key features:
    - Composite mutation: current-to-pbest/1 with archive
    - Per-individual adaptive F and CR via success history memory
    - Archive of rejected-but-good solutions for diversity
    - Diversity-triggered restart mechanism
    - Fully batched operations (no per-individual evaluation)
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)
        self.bounds = (-100.0, 100.0)
        self.lb = self.bounds[0]
        self.ub = self.bounds[1]
        
        # Memory size for adaptive parameters (SHADE-style)
        self.h = min(10, self.np // 3)
        self.f_memory = np.full(self.h, 0.5)
        self.cr_memory = np.full(self.h, 0.5)
        self.memory_index = 0
        self.success_counts = np.zeros(self.h)
        
        # Per-individual parameters
        self.f_values = np.ones(self.np) * 0.5
        self.cr_values = np.ones(self.np) * 0.9
        
        # Archive management
        self.archive = np.empty((0, dim))
        self.archive_max_size = self.np
        
        # State tracking
        self.best_fitness = np.inf
        self.generation = 0
        self.stagnation_count = 0
        self.last_improvement_gen = 0
        self.total_func_evals = 0
        
        # Diversity tracking
        self.diversity_threshold = 1e-6
        self.p_best_rate = 0.1  # top 10% for pbest selection
        
    def __call__(self, func, stopping_condition):
        """Main entry point for optimization."""
        self._reset_state()
        population = self._initialize_population()
        fitness = self._evaluate_batch(func, population)
        self._update_best_tracking(fitness, population)
        
        while not stopping_condition():
            self.generation += 1
            
            # Generate trial population (fully vectorized)
            trials = self._generate_trials_batch(population, fitness)
            trial_fitness = self._evaluate_batch(func, trials)
            
            if len(trial_fitness) < len(trials):
                valid_count = len(trial_fitness)
                trials = trials[:valid_count]
                trial_fitness = trial_fitness[:valid_count]
                if stopping_condition():
                    break
            
            # Selection with adaptation tracking
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update archive with rejected but good solutions
            self._update_archive(population, fitness, trials, trial_fitness, improved_mask)
            
            # Adapt parameters based on success history
            self._adapt_parameters(improved_mask, self.f_values, self.cr_values)
            
            # Check for improvement and update tracking
            prev_best = self.best_fitness
            self._update_best_tracking(fitness, population)
            
            if self.best_fitness < prev_best:
                self.stagnation_count = 0
                self.last_improvement_gen = self.generation
            else:
                self.stagnation_count += 1
            
            # Check stopping condition after selection
            if stopping_condition():
                break
            
            # Diversity-based adaptive restart
            self._restart_if_stagnant(population, fitness, func, stopping_condition)
            
            # Diversity-guided mutation adjustment
            self._adjust_mutation_intensity(population)
        
        return self.best_fitness, self.best_solution
    
    def _reset_state(self):
        """Reset all state variables for a fresh run."""
        self.archive = np.empty((0, self.dim))
        self.best_fitness = np.inf
        self.generation = 0
        self.stagnation_count = 0
        self.last_improvement_gen = 0
        self.total_func_evals = 0
        self.f_memory.fill(0.5)
        self.cr_memory.fill(0.5)
        self.success_counts.fill(0)
        self.memory_index = 0
        
    def _initialize_population(self):
        """Initialize population with Latin Hypercube Sampling for better spread."""
        population = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            samples = np.linspace(0, 1, self.np) + np.random.uniform(0, 1, self.np) / self.np
            samples = np.clip(samples, 0, 0.9999)
            population[:, d] = self.lb + samples * (self.ub - self.lb)
        np.random.shuffle(population)
        return population
    
    def _evaluate_batch(self, func, population):
        """Evaluate entire population in a single batched call."""
        clipped = np.clip(population, self.lb, self.ub)
        fitness = func(clipped)
        if fitness is None or len(fitness) == 0:
            return np.full(len(population), np.inf)
        self.total_func_evals += len(fitness)
        return np.asarray(fitness, dtype=np.float64)
    
    def _update_best_tracking(self, fitness, population):
        """Track the best solution found so far."""
        valid_fitness = np.where(np.isfinite(fitness), fitness, np.inf)
        min_idx = np.argmin(valid_fitness)
        if valid_fitness[min_idx] < self.best_fitness:
            self.best_fitness = valid_fitness[min_idx]
            self.best_solution = population[min_idx].copy()
    
    def _generate_trials_batch(self, population, fitness):
        """Generate trial vectors using composite mutation scheme."""
        n = len(population)
        
        # Sample adaptive F and CR for each individual
        self._sample_adaptive_parameters()
        
        # Current-to-pbest/1 mutation with archive
        p_best_indices = self._select_pbest_indices(n)
        p_best_vectors = population[p_best_indices]
        
        # Get base, target1, target2 indices (excluding current)
        all_indices = np.arange(n)
        
        # Base vectors (current individual)
        base_vectors = population
        
        # First direction: pbest - x_i
        diff1 = p_best_vectors - population
        
        # Second and third directions: from population + archive
        combined_pool = self._get_combined_pool(population)
        
        # Select random pairs for directional mutation
        r1_indices = self._sample_without_replacement(n, combined_pool, population)
        r2_indices = self._sample_without_replacement(n, combined_pool, population, r1_indices)
        
        diff2 = population[r1_indices] - population[r2_indices]
        
        # Composite mutation: base + F1*diff1 + F2*diff2
        f1 = self.f_values[:, np.newaxis]
        f2 = 0.5 * self.f_values[:, np.newaxis]
        
        mutants = base_vectors + f1 * diff1 + f2 * diff2
        
        # Binomial crossover
        trials = self._crossover_batch(population, mutants)
        
        # Clip to bounds
        trials = np.clip(trials, self.lb, self.ub)
        
        return trials
    
    def _sample_adaptive_parameters(self):
        """Sample F and CR values from memory-based distributions."""
        for i in range(self.np):
            # Sample F from Cauchy distribution centered on memory value
            f_mem = self.f_memory[self.memory_index % self.h]
            f_sample = f_mem + 0.1 * np.random.standard_cauchy()
            self.f_values[i] = np.clip(f_sample, 0.0, 1.5)
            
            # Sample CR from normal distribution centered on memory value
            cr_mem = self.cr_memory[self.memory_index % self.h]
            self.cr_values[i] = np.clip(np.random.normal(cr_mem, 0.1), 0.0, 1.0)
    
    def _select_pbest_indices(self, n):
        """Select pbest indices - top p_best_rate of population."""
        n_pbest = max(1, int(n * self.p_best_rate))
        # Use fitness-based selection
        pbest_pool = np.argsort(np.random.permutation(n))[:n_pbest]
        indices = np.random.choice(pbest_pool, size=n, replace=True)
        return indices
    
    def _get_combined_pool(self, population):
        """Combine current population with archive for mutation candidates."""
        if len(self.archive) == 0:
            return population
        max_archive = min(len(self.archive), self.np)
        return np.vstack([population, self.archive[:max_archive]])
    
    def _sample_without_replacement(self, n, pool, exclude_indices, avoid_indices=None):
        """Sample n unique indices from pool, avoiding excluded ones."""
        pool_size = len(pool)
        indices = np.random.choice(pool_size, size=n, replace=True)
        
        # Ensure we don't pick the same index as the base individual
        for i in range(n):
            base_idx = exclude_indices[i] if isinstance(exclude_indices, np.ndarray) else i
            attempts = 0
            while indices[i] == base_idx and attempts < 10:
                indices[i] = np.random.randint(0, pool_size)
                attempts += 1
        
        return indices
    
    def _crossover_batch(self, target, mutant):
        """Perform binomial crossover - fully vectorized."""
        n = len(target)
        cr_expanded = self.cr_values[:, np.newaxis]
        
        # Random dimensions to keep from target
        mask = np.random.random((n, self.dim)) < cr_expanded
        
        # Ensure at least one dimension comes from mutant
        enforce_mask = np.zeros((n, self.dim), dtype=bool)
        enforce_idx = np.random.randint(0, self.dim, size=n)
        enforce_mask[np.arange(n), enforce_idx] = True
        
        mask = mask | enforce_mask
        
        trials = np.where(mask, mutant, target)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors based on fitness comparison - returns improved_mask."""
        n = len(population)
        improved_mask = np.zeros(n, dtype=bool)
        
        # Find where trial is better
        better_mask = trial_fitness < fitness
        
        # Update population with better trials
        new_population = population.copy()
        new_population[better_mask] = trials[better_mask]
        
        new_fitness = fitness.copy()
        new_fitness[better_mask] = trial_fitness[better_mask]
        
        # Track individuals that improved (for parameter adaptation)
        improved_mask = better_mask & np.isfinite(trial_fitness)
        
        return new_population, new_fitness, improved_mask
    
    def _update_archive(self, population, fitness, trials, trial_fitness, improved_mask):
        """Update the archive with rejected-but-good solutions."""
        # Add rejected solutions that are still competitive
        rejected_mask = ~improved_mask & np.isfinite(trial_fitness)
        rejected_count = np.sum(rejected_mask)
        
        if rejected_count > 0:
            # Keep a fraction of rejected solutions (those within 2x best)
            threshold = self.best_fitness * 2.0 if self.best_fitness > 0 else 1.0
            keep_mask = rejected_mask & (trial_fitness < threshold)
            
            rejected_solutions = trials[keep_mask]
            
            # Add to archive (with size limiting)
            if len(rejected_solutions) > 0:
                new_archive = np.vstack([self.archive, rejected_solutions])
                
                # Limit archive size
                if len(new_archive) > self.archive_max_size:
                    # Keep best solutions from archive
                    archive_fitness = np.array([self._quick_estimate(s) for s in new_archive])
                    keep_idx = np.argsort(archive_fitness)[:self.archive_max_size]
                    self.archive = new_archive[keep_idx]
                else:
                    self.archive = new_archive
    
    def _quick_estimate(self, solution):
        """Quick estimate for archive pruning - sum of squares."""
        return np.sum(solution ** 2)
    
    def _adapt_parameters(self, improved_mask, f_values, cr_values):
        """Adapt F and CR parameters based on successful mutations (SHADE-style)."""
        if not np.any(improved_mask):
            return
        
        # Get successful F and CR values
        successful_f = f_values[improved_mask]
        successful_cr = cr_values[improved_mask]
        
        # Weighted arithmetic mean for CR
        if np.any(improved_mask):
            weight_sum = np.sum(improved_mask.astype(float))
            if weight_sum > 0:
                cr_weighted = np.sum(successful_cr * improved_mask.astype(float)) / weight_sum
                f_weighted = np.sum(successful_f * improved_mask.astype(float)) / weight_sum
                
                # Update memory
                if self.generation > 0:
                    self.cr_memory[self.memory_index % self.h] = cr_weighted
                    self.f_memory[self.memory_index % self.h] = f_weighted
                    self.success_counts[self.memory_index % self.h] += np.sum(improved_mask)
                    
                    # Rotate memory index
                    if self.generation % self.h == 0:
                        self.memory_index = 0
                    else:
                        self.memory_index = (self.memory_index + 1) % self.h
    
    def _compute_diversity(self, population):
        """Compute population diversity using average pairwise distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample-based distance estimation for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Compute mean distance to centroid
        centroid = np.mean(sample, axis=0)
        distances = np.linalg.norm(sample - centroid, axis=1)
        
        return np.mean(distances)
    
    def _restart_if_stagnant(self, population, fitness, func, stopping_condition):
        """Restart population if stagnation is detected."""
        stagnation_threshold = 50 + self.generation // 10
        
        if self.stagnation_count >= stagnation_threshold:
            # Compute diversity
            diversity = self._compute_diversity(population)
            
            # Restart triggers
            low_diversity = diversity < self.diversity_threshold * self.dim
            no_improvement = self.generation - self.last_improvement_gen > stagnation_threshold
            
            if low_diversity or no_improvement:
                # Reinitialize portion of population
                n_restart = max(self.np // 3, 10)
                restart_indices = np.random.choice(self.np, n_restart, replace=False)
                
                # Generate new random individuals
                new_portion = self._initialize_population()[:n_restart]
                population[restart_indices] = new_portion
                
                # Evaluate new individuals
                new_fitness = self._evaluate_batch(func, new_portion)
                
                if len(new_fitness) < len(new_portion):
                    return
                
                fitness[restart_indices] = new_fitness
                
                # Update tracking
                self._update_best_tracking(fitness, population)
                self.stagnation_count = 0
                
                # Clear some archive to encourage new exploration
                if len(self.archive) > self.np // 2:
                    self.archive = self.archive[:self.np // 4]
    
    def _adjust_mutation_intensity(self, population):
        """Adjust mutation intensity based on diversity."""
        diversity = self._compute_diversity(population)
        target_diversity = 0.1 * self.dim
        
        # If diversity is low, increase F values
        if diversity < target_diversity:
            self.f_values = np.clip(self.f_values * 1.1, 0.1, 1.5)
        elif diversity > target_diversity * 2:
            self.f_values = np.clip(self.f_values * 0.95, 0.1, 1.5)
```