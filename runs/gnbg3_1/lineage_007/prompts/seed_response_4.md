```python
import numpy as np


class AdaptiveDescentMemeticOptimizer:
    """
    Hybrid optimizer combining DE-style population-based exploration with
    Nelder-Mead local search on elite individuals. Features adaptive mutation
    scaling, strategy adaptation, and diversity-triggered restarts.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self._dim = dim
        self._NP = min(200, max(60, 5 * dim))
        self._max_iter = kwargs.get('max_iter', 1000)
        
        # Search bounds
        self._lb = -100.0
        self._ub = 100.0
        
        # Local search parameters
        self._local_search_freq = 5
        self._local_search_elite_ratio = 0.1
        self._ls_max_iter = 20
        
        # Diversity and stagnation
        self._diversity_threshold = 1e-6
        self._stagnation_patience = 30
        
        # Adaptive parameters
        self._F = 0.6
        self._CR = 0.8
        self._F_amplify = 1.1
        self._F_decay = 0.9
        self._F_min = 0.1
        self._F_max = 1.5
        
        # Strategy pool: (type, F_scale, CR)
        self._strategies = [
            ('rand', 1.0, 0.7),
            ('best', 0.8, 0.9),
            ('current-to-best', 0.9, 0.8),
            ('rand-to-best', 1.1, 0.7),
        ]
        self._strategy_scores = np.ones(len(self._strategies))
        self._current_strategy = 0
        
        # Tracking
        self._f_opt = np.inf
        self._x_opt = None
        self._gen = 0
        self._func_evals = 0
        
    def _initialize_population(self, func):
        """Initialize population with Latin hypercube sampling."""
        pop = np.zeros((self._NP, self._dim))
        for d in range(self._dim):
            pop[:, d] = np.random.uniform(self._lb, self._ub, self._NP)
        # LHS-style shuffling for better spread
        for d in range(self._dim):
            idx = np.random.permutation(self._NP)
            pop[:, d] = pop[idx, d]
        self._population = np.clip(pop, self._lb, self._ub)
        self._fitness = func(self._population)
        self._func_evals += len(self._fitness)
        if len(self._fitness) < self._NP:
            self._fitness = np.resize(self._fitness, self._NP)
        self._update_best_tracking()
        self._best_history = [self._f_opt]
        self._stagnation_count = 0
        
    def _update_best_tracking(self):
        """Update best solution tracking."""
        valid = np.isfinite(self._fitness)
        if not np.any(valid):
            return
        best_idx = np.argmin(self._fitness[valid])
        if self._fitness[valid][best_idx] < self._f_opt - 1e-12:
            self._f_opt = self._fitness[valid][best_idx]
            self._x_opt = self._population[valid][best_idx].copy()
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1
            
    def _mutate_batch(self):
        """Generate donor vectors using adaptive strategy pool."""
        strategy_type, f_scale, _ = self._strategies[self._current_strategy]
        F_adapted = np.clip(self._F * f_scale, self._F_min, self._F_max)
        
        # Select indices for mutation
        indices = np.arange(self._NP)
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]
        
        donors = np.empty((self._NP, self._dim))
        
        if strategy_type == 'rand':
            donors = self._population[r1] + F_adapted * (self._population[r2] - self._population[r3])
        elif strategy_type == 'best':
            donors = self._x_opt + F_adapted * (self._population[r1] - self._population[r2])
        elif strategy_type == 'current-to-best':
            donors = self._population + F_adapted * (self._x_opt - self._population) + \
                     F_adapted * (self._population[r1] - self._population[r2])
        elif strategy_type == 'rand-to-best':
            donors = self._population[r1] + F_adapted * (self._x_opt - self._population[r1]) + \
                     F_adapted * (self._population[r2] - self._population[r3])
                    
        return np.clip(donors, self._lb, self._ub)
        
    def _crossover_batch(self, donors):
        """Binomial crossover with dimension guarantee."""
        CR_adapted = np.clip(self._CR, 0.3, 0.95)
        trials = self._population.copy()
        mask = np.random.random((self._NP, self._dim)) < CR_adapted.reshape(-1, 1)
        # Ensure at least one dimension from donor
        enforce_idx = np.random.randint(0, self._dim, self._NP)
        mask[np.arange(self._NP), enforce_idx] = True
        trials[mask] = donors[mask]
        return np.clip(trials, self._lb, self._ub)
        
    def _select_survivors_batch(self, trials, trial_fitness):
        """Greedy selection with adaptive F update based on success."""
        success_mask = trial_fitness < self._fitness
        
        # Update F based on success rate
        success_rate = np.mean(success_mask)
        if success_rate > 0.2:
            self._F = np.clip(self._F * self._F_amplify, self._F_min, self._F_max)
        else:
            self._F = np.clip(self._F * self._F_decay, self._F_min, self._F_max)
            
        # Update strategy score
        self._strategy_scores[self._current_strategy] += success_rate * 0.1
        # Periodically select best strategy
        if self._gen % 10 == 0:
            self._current_strategy = np.argmax(self._strategy_scores)
            
        # Create new population
        better_trials = trials[success_mask]
        better_fitness = trial_fitness[success_mask]
        
        new_pop = self._population.copy()
        new_fit = self._fitness.copy()
        
        # Replace where trials are better
        replace_idx = np.where(success_mask)[0]
        new_pop[replace_idx] = trials[replace_idx]
        new_fit[replace_idx] = trial_fitness[replace_idx]
        
        self._population = new_pop
        self._fitness = new_fit
        
    def _apply_nelder_mead_batch(self, elite_indices, func):
        """Apply Nelder-Mead local search to elite individuals in batch."""
        if len(elite_indices) == 0:
            return
            
        n_ls = len(elite_indices)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        # Build simplex for each elite (current point + dim random offsets)
        perturbations = np.random.uniform(-0.5, 0.5, (n_ls, self._dim, self._dim + 1))
        perturbations[np.arange(n_ls), :, 0] = 0.0  # First point is center
        
        simplices = self._population[elite_indices].reshape(n_ls, 1, self._dim) + \
                    perturbations * (self._ub - self._lb) * 0.1
        simplices = np.clip(simplices, self._lb, self._ub)
        
        # Evaluate all simplex points
        all_points = simplices.reshape(n_ls * (self._dim + 1), self._dim)
        all_fitness = func(all_points)
        self._func_evals += len(all_fitness)
        
        if len(all_fitness) < n_ls * (self._dim + 1):
            return
            
        all_fitness = all_fitness.reshape(n_ls, self._dim + 1)
        
        improved = np.zeros(n_ls, dtype=bool)
        new_centroids = np.zeros((n_ls, self._dim))
        
        for i in range(n_ls):
            fitness = all_fitness[i]
            points = simplices[i]
            sorted_idx = np.argsort(fitness)
            points = points[sorted_idx]
            fitness = fitness[sorted_idx]
            
            centroid = points[:-1].mean(axis=0)
            new_centroids[i] = centroid
            
            # Reflection
            reflected = centroid + alpha * (centroid - points[-1])
            reflected = np.clip(reflected, self._lb, self._ub)
            reflected_fit = func(reflected.reshape(1, -1))[0]
            self._func_evals += 1
            
            if np.isfinite(reflected_fit) and fitness[0] <= reflected_fit < fitness[-1]:
                points[-1] = reflected
                fitness[-1] = reflected_fit
                improved[i] = True
            elif reflected_fit < fitness[0]:
                # Expansion
                expanded = centroid + gamma * (reflected - centroid)
                expanded = np.clip(expanded, self._lb, self._ub)
                expanded_fit = func(expanded.reshape(1, -1))[0]
                self._func_evals += 1
                
                if np.isfinite(expanded_fit) and expanded_fit < reflected_fit:
                    points[-1] = expanded
                    fitness[-1] = expanded_fit
                else:
                    points[-1] = reflected
                    fitness[-1] = reflected_fit
                improved[i] = True
            else:
                # Contraction
                contracted = centroid + rho * (points[-1] - centroid)
                contracted = np.clip(contracted, self._lb, self._ub)
                contracted_fit = func(contracted.reshape(1, -1))[0]
                self._func_evals += 1
                
                if np.isfinite(contracted_fit) and contracted_fit < fitness[-1]:
                    points[-1] = contracted
                    fitness[-1] = contracted_fit
                    improved[i] = True
                    
            # Shrink
            for j in range(1, self._dim + 1):
                if not improved[i]:
                    shrunk = points[0] + sigma * (points[j] - points[j])
                    shrunk = np.clip(shrunk, self._lb, self._ub)
                    points[j] = shrunk
                    
            # Update population if improved
            if improved[i] and fitness[0] < self._fitness[elite_indices[i]]:
                self._population[elite_indices[i]] = points[0]
                self._fitness[elite_indices[i]] = fitness[0]
                
    def _adapt_parameters_batch(self):
        """Adapt CR based on recent performance."""
        if len(self._best_history) > 5:
            recent_improvement = self._best_history[-1] - min(self._best_history[-5:])
            if recent_improvement > 1e-8:
                self._CR = np.clip(self._CR * 1.05, 0.3, 0.95)
            else:
                self._CR = np.clip(self._CR * 0.95, 0.3, 0.95)
                
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        centroid = self._population.mean(axis=0)
        distances = np.sqrt(((self._population - centroid) ** 2).sum(axis=1))
        return distances.mean()
        
    def _restart_if_stagnant(self, func):
        """Restart population if stagnated or low diversity."""
        should_restart = False
        
        if self._stagnation_count > self._stagnation_patience:
            should_restart = True
        elif self._compute_diversity() < self._diversity_threshold:
            should_restart = True
            
        if should_restart:
            # Keep best, reinitialize others
            n_keep = max(1, self._NP // 5)
            best_indices = np.argsort(self._fitness)[:n_keep]
            self._population[:n_keep] = self._population[best_indices]
            
            for j in range(n_keep, self._NP):
                self._population[j] = self._x_opt + np.random.uniform(-10, 10, self._dim)
                
            self._population = np.clip(self._population, self._lb, self._ub)
            self._fitness = func(self._population)
            self._func_evals += len(self._fitness)
            
            if len(self._fitness) < self._NP:
                self._fitness = np.resize(self._fitness, self._NP)
                
            self._stagnation_count = 0
            self._F = 0.6
            self._strategy_scores[:] = 1.0
            
    def _select_elite_for_local_search(self):
        """Select elite individuals for local search."""
        n_elite = max(1, int(self._NP * self._local_search_elite_ratio))
        elite_threshold = np.percentile(self._fitness, 20)
        elite_mask = self._fitness <= elite_threshold
        elite_indices = np.where(elite_mask)[0]
        
        if len(elite_indices) > n_elite:
            elite_indices = elite_indices[np.argsort(self._fitness[elite_indices])[:n_elite]]
        return elite_indices
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer."""
        self._initialize_population(func)
        
        while not stopping_condition():
            self._gen += 1
            
            # Generate and evaluate trials
            donors = self._mutate_batch()
            trials = self._crossover_batch(donors)
            trials = np.clip(trials, self._lb, self._ub)
            
            trial_fitness = func(trials)
            self._func_evals += len(trial_fitness)
            
            if len(trial_fitness) < self._NP:
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                
            if stopping_condition():
                break
                
            # Selection
            self._select_survivors_batch(trials, trial_fitness)
            
            # Adapt parameters
            self._adapt_parameters_batch()
            self._best_history.append(self._f_opt)
            
            # Periodic local search on elites
            if self._gen % self._local_search_freq == 0:
                elite_indices = self._select_elite_for_local_search()
                self._apply_nelder_mead_batch(elite_indices, func)
                
            # Update best tracking
            self._update_best_tracking()
            
            # Check for restart
            self._restart_if_stagnant(func)
            
        return self._f_opt, self._x_opt
```