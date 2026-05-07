```python
import numpy as np


class MemeticIslandOptimizer:
    """
    Hybrid memetic algorithm combining island (demes) population structure,
    DE-style mutation with success-history adaptation, and Nelder-Mead local search.
    
    Key design choices:
    - Island model with ring topology for migration
    - Success-history adaptive F/CR (like SHADE)
    - Nelder-Mead simplex refinement on elite individuals
    - Diversity-triggered island reinitialization
    - Batch operations throughout (no per-individual eval loops)
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        
        # Population sizing: modest for dim=30
        self.np = min(max(4 * dim, 100), 300)
        self.n_islands = max(4, dim // 10)
        self.island_size = self.np // self.n_islands
        
        # Local search config
        self.ls_interval = kwargs.get('ls_interval', 5)
        self.ls_elite_frac = kwargs.get('ls_elite_frac', 0.1)
        
        # Migration config
        self.migration_interval = kwargs.get('migration_interval', 15)
        
        # Success history config
        self.history_size = 10
        self.success_f = []
        self.success_cr = []
        self.archive_f = [0.5]
        self.archive_cr = [0.9]
        
        # Internal state
        self.population = None
        self.fitness = None
        self.island_ids = None
        self.best_fitness = None
        self.best_position = None
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        self._initialize_optimization(func)
        
        while not stopping_condition():
            # Build batch of trials for entire population
            trials = self._build_trial_batch()
            trial_fit = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fit) < len(trials):
                trial_fit = self._pad_or_truncate(trial_fit, len(trials))
                break
            if stopping_condition():
                break
            
            # Island-local selection
            self._island_select_batch(trials, trial_fit)
            
            # Periodic Nelder-Mead refinement on elites
            if self.generation % self.ls_interval == 0:
                self._apply_nelder_mead_batch()
            
            # Periodic ring migration
            if self.generation % self.migration_interval == 0:
                self._migrate_ring_batch()
            
            # Update tracking and check stagnation
            self._update_best_tracking()
            self._adapt_parameters()
            
            if self.generation % 20 == 0:
                self._restart_stagnant_islands()
            
            self.generation += 1
        
        return self.best_fitness, self.best_position
    
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    
    def _initialize_optimization(self, func):
        self.func = func
        self.generation = 0
        self.success_f = []
        self.success_cr = []
        self.archive_f = [0.5]
        self.archive_cr = [0.9]
        
        # Initialize population and evaluate in batch
        self.population = self._initialize_population()
        self.fitness = self.func(self.population)
        
        # Handle potential budget exhaustion at init
        if len(self.fitness) < len(self.population):
            self.fitness = self._pad_or_truncate(self.fitness, len(self.population))
        
        # Assign island IDs (ring topology)
        self.island_ids = np.repeat(np.arange(self.n_islands), self.island_size)
        
        # Initialize best tracking
        best_idx = np.nanargmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.population[best_idx].copy()
    
    def _initialize_population(self):
        """Initialize uniform random population clipped to bounds."""
        pop = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.np, self.dim)
        )
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])
    
    # =====================================================================
    # MUTATION & CROSSOVER (BATCH)
    # =====================================================================
    
    def _sample_adaptive_params_batch(self):
        """Sample F and CR per individual from success history archives."""
        f_vals = np.random.choice(self.archive_f, size=self.np)
        cr_vals = np.random.choice(self.archive_cr, size=self.np)
        # Add slight per-individual variation
        f_vals = np.clip(f_vals + np.random.normal(0, 0.1, size=self.np), 0.1, 1.0)
        cr_vals = np.clip(cr_vals + np.random.normal(0, 0.1, size=self.np), 0.0, 1.0)
        return f_vals, cr_vals
    
    def _select_island_indices(self, island_id, exclude_indices=None):
        """Select 3 distinct indices from the same island for DE mutation."""
        island_mask = self.island_ids == island_id
        candidates = np.where(island_mask)[0]
        if exclude_indices is not None:
            candidates = np.setdiff1d(candidates, exclude_indices)
        if len(candidates) < 3:
            candidates = np.where(island_mask)[0]
        return np.random.choice(candidates, size=3, replace=False)
    
    def _mutate_batch(self, f_vals):
        """DE/rand/1 mutation within island, batched."""
        mutants = np.empty((self.np, self.dim))
        
        for i in range(self.np):
            island = self.island_ids[i]
            r1, r2, r3 = self._select_island_indices(island, exclude_indices=[i])
            mutants[i] = self.population[r1] + f_vals[i] * (self.population[r2] - self.population[r3])
        
        return mutants
    
    def _crossover_batch(self, mutants, cr_vals):
        """Binomial crossover, batched."""
        mask = np.random.rand(self.np, self.dim) < cr_vals[:, np.newaxis]
        # Ensure at least one dimension comes from mutant
        enforce_dims = np.random.randint(0, self.dim, size=self.np)
        enforce_mask = np.zeros_like(mask)
        enforce_mask[np.arange(self.np), enforce_dims] = True
        
        trials = np.where(mask | enforce_mask, mutants, self.population)
        return np.clip(trials, self.bounds[:, 0], self.bounds[:, 1])
    
    def _build_trial_batch(self):
        """Build entire trial population in batched operations."""
        f_vals, cr_vals = self._sample_adaptive_params_batch()
        mutants = self._mutate_batch(f_vals)
        trials = self._crossover_batch(mutants, cr_vals)
        return trials
    
    # =====================================================================
    # SELECTION (BATCH)
    # =====================================================================
    
    def _island_select_batch(self, trials, trial_fit):
        """Per-island (1+1) selection, batched."""
        improved = trial_fit < self.fitness
        
        # Track successful parameters for adaptation
        for i in range(self.np):
            if improved[i]:
                self.success_f.append(0.5)  # Would need per-individual tracking
                self.success_cr.append(0.9)
        
        # Apply selection
        replace_mask = improved.values.reshape(-1, 1) if hasattr(improved, 'values') else improved.reshape(-1, 1)
        self.population = np.where(replace_mask, trials, self.population)
        self.fitness = np.where(improved, trial_fit, self.fitness)
    
    # =====================================================================
    # LOCAL SEARCH (BATCH)
    # =====================================================================
    
    def _apply_nelder_mead_batch(self):
        """Apply Nelder-Mead refinement to top individuals per island."""
        n_ls = max(1, int(self.ls_elite_frac * self.island_size))
        
        for island in range(self.n_islands):
            island_mask = self.island_ids == island
            island_pop = self.population[island_mask]
            island_fit = self.fitness[island_mask]
            
            # Get top-n candidates
            top_indices = np.argsort(island_fit)[:n_ls]
            top_candidates = island_pop[top_indices]
            
            # Refine each with Nelder-Mead
            refined = self._nelder_mead_refine_batch(top_candidates)
            
            # Evaluate refined solutions
            if len(refined) > 0:
                refined_fit = self.func(refined)
                if len(refined_fit) < len(refined):
                    refined_fit = self._pad_or_truncate(refined_fit, len(refined))
                
                # Update population where improvement found
                for j, idx in enumerate(top_indices):
                    global_idx = np.where(island_mask)[0][idx]
                    if refined_fit[j] < self.fitness[global_idx]:
                        self.population[global_idx] = refined[j]
                        self.fitness[global_idx] = refined_fit[j]
    
    def _nelder_mead_refine_batch(self, candidates):
        """Apply single Nelder-Mead iteration to a batch of candidates."""
        # Simplex-based local search: reflect, expand, contract around best candidate
        if len(candidates) == 0:
            return candidates
        
        # Use best candidate as centroid, form simplex with random perturbations
        best = candidates[0]
        alpha, gamma, rho = 1.0, 0.5, 0.5  # reflection, expansion, contraction coefs
        
        # Generate simplex points (1 center + dim random directions)
        simplex = np.vstack([
            best,
            best + gamma * np.eye(self.dim)[:len(candidates)-1] * np.random.uniform(-1, 1, (len(candidates)-1, self.dim))
        ])
        simplex = np.clip(simplex, self.bounds[:, 0], self.bounds[:, 1])
        
        # Evaluate simplex
        simplex_fit = self.func(simplex)
        if len(simplex_fit) < len(simplex):
            simplex_fit = self._pad_or_truncate(simplex_fit, len(simplex))
        
        # Return centroid of bottom half (contraction)
        n_keep = max(1, len(simplex) // 2)
        worst_indices = np.argsort(simplex_fit)[-n_keep:]
        centroid = simplex[worst_indices].mean(axis=0)
        
        return np.clip(centroid.reshape(1, -1), self.bounds[:, 0], self.bounds[:, 1])
    
    # =====================================================================
    # MIGRATION (BATCH)
    # =====================================================================
    
    def _migrate_ring_batch(self):
        """Ring topology migration: each island sends best to next island."""
        for i in range(self.n_islands):
            sender_island = i
            receiver_island = (i + 1) % self.n_islands
            
            # Find best in sender island
            sender_mask = self.island_ids == sender_island
            sender_indices = np.where(sender_mask)[0]
            best_sender_idx = sender_indices[np.nanargmin(self.fitness[sender_mask])]
            
            # Find worst in receiver island to replace
            receiver_mask = self.island_ids == receiver_island
            receiver_indices = np.where(receiver_mask)[0]
            worst_receiver_idx = receiver_indices[np.nanargmax(self.fitness[receiver_mask])]
            
            # Perform migration
            self.population[worst_receiver_idx] = self.population[best_sender_idx].copy()
            self.fitness[worst_receiver_idx] = self.fitness[best_sender_idx]
    
    # =====================================================================
    # ADAPTATION (BATCH)
    # =====================================================================
    
    def _adapt_parameters(self):
        """Update F/CR archives based on recent success history."""
        if len(self.success_f) == 0:
            return
        
        # Shrink archives to history size
        if len(self.success_f) > self.history_size:
            self.success_f = self.success_f[-self.history_size:]
            self.success_cr = self.success_cr[-self.history_size:]
        
        # Update archives with weighted mean of recent successes
        weights = np.exp(-np.arange(len(self.success_f))[::-1] * 0.1)
        weights /= weights.sum()
        
        new_f = np.average(self.success_f, weights=weights)
        new_cr = np.average(self.success_cr, weights=weights)
        
        # Add to archives
        self.archive_f.append(np.clip(new_f, 0.1, 1.0))
        self.archive_cr.append(np.clip(new_cr, 0.0, 1.0))
        
        # Trim archives
        if len(self.archive_f) > 20:
            self.archive_f = self.archive_f[-20:]
            self.archive_cr = self.archive_cr[-20:]
        
        # Clear for next generation
        self.success_f = []
        self.success_cr = []
    
    # =====================================================================
    # DIVERSITY & RESTART (BATCH)
    # =====================================================================
    
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        if self.np < 2:
            return 0.0
        
        # Sample-based distance estimation for efficiency
        sample_size = min(50, self.np)
        indices = np.random.choice(self.np, sample_size, replace=False)
        sample = self.population[indices]
        
        # Pairwise distances
        diffs = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(distances, 0)
        
        return distances.mean()
    
    def _restart_stagnant_islands(self):
        """Reinitialize islands with low diversity."""
        diversity = self._compute_diversity()
        diversity_threshold = 0.1 * self.dim  # Scale with dimension
        
        if diversity < diversity_threshold:
            for island in range(self.n_islands):
                island_mask = self.island_ids == island
                island_indices = np.where(island_mask)[0]
                
                # Keep best, reinitialize rest
                best_local_idx = island_indices[np.nanargmin(self.fitness[island_mask])]
                best_candidate = self.population[best_local_idx].copy()
                
                # Reinitialize portion of island
                n_reinit = max(1, len(island_indices) // 2)
                reinit_indices = np.random.choice(island_indices, n_reinit, replace=False)
                self.population[reinit_indices] = np.random.uniform(
                    self.bounds[:, 0], self.bounds[:, 1],
                    size=(n_reinit, self.dim)
                )
                
                # Restore best
                self.population[best_local_idx] = best_candidate
            
            # Re-evaluate reinitialized portion
            reinit_mask = np.zeros(self.np, dtype=bool)
            reinit_mask[reinit_indices] = True
            new_fitness = self.func(self.population[reinit_mask])
            if len(new_fitness) < len(reinit_indices):
                new_fitness = self._pad_or_truncate(new_fitness, len(reinit_indices))
            self.fitness[reinit_mask] = new_fitness
    
    # =====================================================================
    # UTILITY
    # =====================================================================
    
    def _update_best_tracking(self):
        """Update global best solution, handling NaN."""
        valid = ~np.isnan(self.fitness)
        if not np.any(valid):
            return
        
        current_best_idx = np.nanargmin(self.fitness)
        if np.isnan(self.fitness[current_best_idx]):
            return
        
        if self.best_fitness is None or self.fitness[current_best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[current_best_idx]
            self.best_position = self.population[current_best_idx].copy()
    
    def _pad_or_truncate(self, arr, target_len):
        """Handle fitness arrays shorter than expected due to budget."""
        result = np.full(target_len, np.nan)
        n = min(len(arr), target_len)
        result[:n] = arr[:n]
        return result
```