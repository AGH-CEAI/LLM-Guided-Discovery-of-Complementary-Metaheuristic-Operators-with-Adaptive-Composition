```python
import numpy as np


class GradientGuidedMemeticOptimizer:
    """
    A memetic optimizer that combines DE-style mutation guided by an
    accumulated historical gradient (momentum-like) with periodic
    Nelder-Mead local polishing of elite individuals.
    
    Key differences from prior proposals:
    - Historical Gradient Memory: Tracks successful mutation directions
      over generations to build an implicit gradient estimate (unlike
      covariance-based or topology-based approaches).
    - Memetic Structure: Explicit local search phase using Nelder-Mead
      simplex method on a schedule, not triggered by stagnation.
    - Adaptive Momentum Decay: Historical gradient is exponentially
      smoothed with decay rate that adapts based on success.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(200, max(4 * dim, 8 * dim))  # Population size
        self.lower = -100.0
        self.upper = 100.0
        # Mutable state initialized in __call__
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = None
        self.historical_gradient = None
        self.momentum_weight = 0.3
        self.F = 0.5
        self.F_best = 0.2
        self.CR = 0.7
        self.successful_mutations = None
        self.failed_mutations = None
        
    def _initialize_population(self):
        """Generate uniformly random initial population within bounds."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        self.fitness = np.full(self.np, np.inf)
        self.historical_gradient = np.zeros(self.dim)
        self.successful_mutations = []
        self.failed_mutations = []
        
    def _mutate_batch(self, indices):
        """
        Generate mutant vectors using historical gradient guidance.
        mutant = population[i] + F * historical_gradient + F_best * (best - population[i])
        """
        mutants = np.empty((len(indices), self.dim))
        for j, i in enumerate(indices):
            r1, r2 = self._select_two_different(i)
            base = self.population[i].copy()
            to_best = self.population[0] - base  # best is at index 0 after sorting
            diff = self.population[r1] - self.population[r2]
            mutants[j] = base + self.F * (diff + self.historical_gradient) + self.F_best * to_best
        return mutants
    
    def _select_two_different(self, exclude):
        """Select two distinct random indices different from exclude."""
        indices = list(range(self.np))
        indices.remove(exclude)
        selected = np.random.choice(indices, size=2, replace=False)
        return selected[0], selected[1]
    
    def _crossover_batch(self, targets, mutants):
        """
        Binomial crossover: mix target and mutant per dimension.
        Returns trial vectors (not yet clipped).
        """
        mask = np.random.random((len(targets), self.dim)) < self.CR
        trials = np.where(mask, mutants, targets)
        # Ensure at least one dimension comes from mutant
        enforce_dim = np.random.randint(self.dim, size=len(targets))
        trials[np.arange(len(targets)), enforce_dim] = mutants[np.arange(len(targets)), enforce_dim]
        return trials
    
    def _clip_to_bounds(self, candidates):
        """Hard clip all candidates to search bounds."""
        return np.clip(candidates, self.lower, self.upper)
    
    def _select_survivors_batch(self, targets, target_fitness, trials, trial_fitness):
        """
        One-to-one elitist selection: keep better of target vs trial.
        Returns (new_population, new_fitness).
        """
        better_mask = trial_fitness < target_fitness
        new_pop = targets.copy()
        new_fit = target_fitness.copy()
        new_pop[better_mask] = trials[better_mask]
        new_fit[better_mask] = trial_fitness[better_mask]
        return new_pop, new_fit
    
    def _update_historical_gradient(self):
        """
        Exponentially smooth successful mutation directions to build
        an implicit gradient estimate. Adapts momentum weight.
        """
        if len(self.successful_mutations) == 0:
            return
        # Compute mean successful direction
        success_arr = np.array(self.successful_mutations)
        mean_direction = np.mean(success_arr, axis=0)
        # Normalize to prevent explosion
        grad_norm = np.linalg.norm(mean_direction)
        if grad_norm > 1e-8:
            mean_direction = mean_direction / grad_norm
        # Adapt momentum weight based on recent success rate
        total = len(self.successful_mutations) + len(self.failed_mutations) + 1
        success_rate = len(self.successful_mutations) / total
        self.momentum_weight = 0.1 + 0.5 * success_rate
        # Exponential smoothing update
        self.historical_gradient = (
            (1 - self.momentum_weight) * self.historical_gradient +
            self.momentum_weight * mean_direction * 10.0
        )
        # Clear buffers
        self.successful_mutations.clear()
        self.failed_mutations.clear()
        
    def _record_mutation_outcome(self, original, mutated, improved):
        """Store mutation direction for historical gradient learning."""
        direction = mutated - original
        if improved:
            self.successful_mutations.append(direction)
        else:
            self.failed_mutations.append(direction)
    
    def _adapt_parameters(self):
        """Adjust F and CR based on recent success patterns."""
        total = len(self.successful_mutations) + len(self.failed_mutations)
        if total < 5:
            return
        success_rate = len(self.successful_mutations) / total
        if success_rate > 0.4:
            self.F = np.clip(self.F * 1.1, 0.3, 1.5)
            self.CR = np.clip(self.CR * 1.05, 0.3, 0.95)
        elif success_rate < 0.15:
            self.F = np.clip(self.F * 0.9, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.95, 0.3, 0.95)
            
    def _nelder_mead_polish(self, x_start, max_iter=50):
        """
        Nelder-Mead simplex local search from x_start.
        Operates on single candidate, returns polished solution.
        """
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        # Initialize simplex around start point
        simplex = np.empty((self.dim + 1, self.dim))
        simplex[0] = x_start.copy()
        step = 0.5
        for i in range(self.dim):
            simplex[i + 1] = x_start.copy()
            simplex[i + 1, i] += step
        simplex = np.clip(simplex, self.lower, self.upper)
        # Evaluate simplex
        vals = np.array([func(x.reshape(1, -1))[0] for x in simplex])
        for _ in range(max_iter):
            if np.isnan(vals).any():
                break
            # Sort simplex
            sorted_idx = np.argsort(vals)
            vals = vals[sorted_idx]
            simplex = simplex[sorted_idx]
            xo = np.mean(simplex[:-1], axis=0)
            xr = xo + alpha * (xo - simplex[-1])
            xr = np.clip(xr, self.lower, self.upper)
            f_xr = func(xr.reshape(1, -1))[0]
            if f_xr < vals[0]:
                xe = xo + gamma * (xr - xo)
                xe = np.clip(xe, self.lower, self.upper)
                f_xe = func(xe.reshape(1, -1))[0]
                if f_xe < f_xr:
                    simplex[-1] = xe
                    vals[-1] = f_xe
                else:
                    simplex[-1] = xr
                    vals[-1] = f_xr
            elif f_xr < vals[-2]:
                simplex[-1] = xr
                vals[-1] = f_xr
            else:
                xc = xo + rho * (simplex[-1] - xo)
                xc = np.clip(xc, self.lower, self.upper)
                f_xc = func(xc.reshape(1, -1))[0]
                if f_xc < vals[-1]:
                    simplex[-1] = xc
                    vals[-1] = f_xc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], self.lower, self.upper)
                        vals[i] = func(simplex[i].reshape(1, -1))[0]
        best_idx = np.argmin(vals)
        return simplex[best_idx], vals[best_idx]
    
    def _apply_local_search_to_elite(self, n_elite=3):
        """Polish top-n_elite individuals with Nelder-Mead on schedule."""
        for i in range(n_elite):
            if len(self.fitness) <= i:
                break
            x_candidate = self.population[i].copy()
            x_polished, f_polished = self._nelder_mead_polish(x_candidate)
            if f_polished < self.fitness[i]:
                self.population[i] = x_polished
                self.fitness[i] = f_polished
                
    def _compute_diversity(self):
        """Population spread: mean pairwise Euclidean distance."""
        if self.np < 2:
            return 0.0
        # Sample for efficiency
        sample_size = min(50, self.np)
        indices = np.random.choice(self.np, size=sample_size, replace=False)
        subpop = self.population[indices]
        diffs = subpop[:, np.newaxis, :] - subpop[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        upper_mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
        return np.mean(dists[upper_mask])
    
    def _restart_if_stagnant(self, diversity_threshold=1e-3):
        """Reinitialize if population diversity is too low."""
        diversity = self._compute_diversity()
        range_size = self.upper - self.lower
        if diversity < diversity_threshold * range_size:
            # Keep best individual, reinitialize rest
            best_idx = np.argmin(self.fitness)
            best_x = self.population[best_idx].copy()
            best_f = self.fitness[best_idx]
            self._initialize_population()
            self.population[0] = best_x
            self.fitness[0] = best_f
            self.historical_gradient *= 0.1  # Dampen momentum on restart
            
    def _sort_by_fitness(self):
        """Sort population by fitness, best first."""
        sorted_idx = np.argsort(self.fitness)
        self.population = self.population[sorted_idx]
        self.fitness = self.fitness[sorted_idx]
        
    def _update_optimum(self):
        """Track global best if current best is better."""
        if self.fitness[0] < self.f_opt:
            self.f_opt = self.fitness[0]
            self.x_opt = self.population[0].copy()
            
    def __call__(self, func, stopping_condition):
        """
        Main optimization loop: evolutionary search with local polishing.
        Returns (f_opt, x_opt).
        """
        self._initialize_population()
        # Initial batch evaluation
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = np.resize(self.fitness, self.np)
            self.fitness[len(self.fitness):] = np.nan
        self._sort_by_fitness()
        self.f_opt = self.fitness[0]
        self.x_opt = self.population[0].copy()
        gen = 0
        local_search_counter = 0
        local_search_interval = 10
        
        while not stopping_condition():
            # Evolutionary operators
            indices = np.arange(self.np)
            mutants = self._mutate_batch(indices)
            trials = self._crossover_batch(self.population, mutants)
            trials = self._clip_to_bounds(trials)
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.resize(trial_fitness, len(trials))
                trial_fitness[len(trial_fitness):] = np.nan
            if stopping_condition():
                break
            # Record mutation outcomes for learning
            for i in range(self.np):
                improved = trial_fitness[i] < self.fitness[i]
                self._record_mutation_outcome(
                    self.population[i], trials[i], improved
                )
            # Selection
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            self._sort_by_fitness()
            self._update_optimum()
            # Parameter adaptation
            self._update_historical_gradient()
            self._adapt_parameters()
            # Periodic local search on elite
            local_search_counter += 1
            if local_search_counter >= local_search_interval:
                self._apply_local_search_to_elite(n_elite=3)
                self._sort_by_fitness()
                self._update_optimum()
                local_search_counter = 0
            # Check for stagnation
            self._restart_if_stagnant()
            gen += 1
            
        return self.f_opt, self.x_opt
```