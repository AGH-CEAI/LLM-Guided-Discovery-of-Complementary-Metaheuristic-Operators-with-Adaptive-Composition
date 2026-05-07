Looking at the feedback, the best algorithms used cultural mechanisms, covariance adaptation, and velocity-based updates. I should propose something strategically different.

I'll create **"MemeticDifferentialArchive"** - a hybrid optimizer combining:
1. Multi-strategy DE mutation with success-based adaptation (JADE-style)
2. Secondary archive for storing successful solutions  
3. Periodic Nelder-Mead local search on elite solutions
4. Population diversity management with restart

This is different from previous proposals because:
- Uses JADE-style adaptive mutation strategy selection (not cultural)
- Has explicit archive-based exploitation (not covariance-based)
- Combines DE with Nelder-Mead simplex (not PSO-based)

```python
import numpy as np


class MemeticDifferentialArchive:
    """
    Hybrid optimizer: Adaptive DE with JADE-style parameter adaptation,
    secondary success archive, periodic Nelder-Mead local search,
    and diversity-triggered restarts.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(10 * dim, 300)  # Population size: 4*dim to 10*dim
        self.lower = -100.0
        self.upper = 100.0
        
        # JADE-style adaptive parameters
        self.c = 0.1  # Adaptation rate for mu_cr and mu_f
        self.p_best_rate = 0.05  # Top p% candidates for mutation
        
        # Archive for successful solutions
        self.archive = []
        self.archive_max_size = self.np
        
        # Local search control
        self.ls_interval = 50  # Generations between local search
        self.ls_rho = 0.5  # Initial simplex size for NM
        
        # Restart control
        self.max_stagnant_generations = 150
        self.min_diversity_threshold = 1e-6
        
        # Internal state
        self.mu_cr = 0.5
        self.mu_f = 0.5
        self.f_opt = np.inf
        self.x_opt = None
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        pop = np.random.uniform(self.lower, self.upper, (self.np, self.dim))
        pop = self._clip_to_bounds_batch(pop)
        return pop
    
    def _clip_to_bounds_batch(self, pop):
        """Clip all individuals to search bounds."""
        return np.clip(pop, self.lower, self.upper)
    
    def _eval_wrapper(self, pop, func):
        """Evaluate population, handle budget truncation."""
        fit = func(pop)
        if len(fit) < len(pop):
            fit = np.resize(fit, len(pop))
            fit[len(fit):] = np.inf
        return fit
    
    def _update_archive_batch(self, pop, fitness):
        """Update secondary archive with successful solutions."""
        n_added = self.np // 5
        sorted_idx = np.argsort(fitness)
        for i in range(min(n_added, len(sorted_idx))):
            idx = sorted_idx[i]
            if len(self.archive) < self.archive_max_size:
                self.archive.append(pop[idx].copy())
            else:
                replace_idx = np.random.randint(len(self.archive))
                self.archive[replace_idx] = pop[idx].copy()
    
    def _select_mutation_strategy_batch(self):
        """Select mutation strategy indices per individual (JADE-style)."""
        # 3 strategies: current-to-pbest, rand, best
        strategies = np.random.randint(0, 3, self.np)
        # Generate Cr and F per individual
        cr = np.random.normal(self.mu_cr, 0.1, self.np)
        cr = np.clip(cr, 0.0, 1.0)
        f = np.abs(np.random.normal(self.mu_f, 0.1, self.np))
        f = np.clip(f, 0.0, 2.0)
        return strategies, cr, f
    
    def _mutate_current_to_pbest_batch(self, pop, fitness, f, p_best_rate):
        """Current-to-pbest mutation with archive extension."""
        p_best_size = max(2, int(p_best_rate * self.np))
        sorted_idx = np.argsort(fitness)
        p_best_idx = sorted_idx[:p_best_size]
        
        p_best = pop[np.random.choice(p_best_idx)]
        
        # Select 2 random distinct indices different from current
        idx_range = np.arange(self.np)
        candidates = np.random.choice(idx_range, size=(self.np, 3), replace=False)
        
        # Build combined pool: population + archive
        combined = pop
        if len(self.archive) > 0:
            combined = np.vstack([pop, np.array(self.archive)])
            ext_size = len(combined)
            ext_idx = np.random.choice(ext_size, size=(self.np, 2), replace=False)
            r1 = ext_idx[:, 0]
            r2 = ext_idx[:, 1]
        else:
            r1 = candidates[:, 0]
            r2 = candidates[:, 1]
        
        current = pop
        mutant = current + f.reshape(-1, 1) * (p_best.reshape(1, -1) - current) + \
                 f.reshape(-1, 1) * (pop[r1] - pop[r2])
        return mutant
    
    def _mutate_rand_batch(self, pop, f):
        """Classic rand/1 mutation."""
        idx = np.arange(self.np)
        candidates = np.random.choice(idx, size=(self.np, 3), replace=False)
        r1, r2, r3 = candidates[:, 0], candidates[:, 1], candidates[:, 2]
        mutant = pop[r1] + f.reshape(-1, 1) * (pop[r2] - pop[r3])
        return mutant
    
    def _mutate_best_batch(self, pop, f):
        """Best/1 mutation."""
        best_idx = np.argmin(np.inf)
        best = pop[best_idx]
        idx = np.arange(self.np)
        candidates = np.random.choice(idx, size=(self.np, 2), replace=False)
        r1, r2 = candidates[:, 0], candidates[:, 1]
        mutant = best + f.reshape(-1, 1) * (pop[r1] - pop[r2])
        return mutant
    
    def _mutate_batch(self, pop, fitness, strategies, f):
        """Apply selected mutation strategies to generate trial population."""
        p_best_rate = self.p_best_rate
        mutants = np.empty((self.np, self.dim))
        
        for i in range(self.np):
            if strategies[i] == 0:
                mutants[i] = self._mutate_current_to_pbest_batch(
                    pop, fitness, f[i:i+1], p_best_rate
                )[0]
            elif strategies[i] == 1:
                mutants[i] = self._mutate_rand_batch(pop, f[i:i+1])[0]
            else:
                mutants[i] = self._mutate_best_batch(pop, f[i:i+1])[0]
        
        return mutants
    
    def _crossover_batch(self, pop, mutant, cr):
        """Binomial crossover with per-individual Cr."""
        cross_mask = np.random.random((self.np, self.dim)) < cr.reshape(-1, 1)
        trials = np.where(cross_mask, mutant, pop)
        
        # Ensure at least one dimension from mutant
        enforce_dim = np.random.randint(0, self.dim, self.np)
        enforce_mask = np.zeros((self.np, self.dim), dtype=bool)
        enforce_mask[np.arange(self.np), enforce_dim] = True
        trials = np.where(enforce_mask, mutant, trials)
        
        return trials
    
    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        """Greedy selection between parent and trial."""
        better_mask = trial_fitness < fitness
        new_pop = np.where(better_mask.reshape(-1, 1), trials, pop)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        return new_pop, new_fitness
    
    def _adapt_parameters_batch(self, successful_cr, successful_f, strategy_counts):
        """Update mu_cr and mu_f based on successful adaptations (JADE-style)."""
        if len(successful_cr) > 0:
            self.mu_cr = (1 - self.c) * self.mu_cr + self.c * np.mean(successful_cr)
        if len(successful_f) > 0:
            self.mu_f = (1 - self.c) * self.mu_f + self.c * np.mean(successful_f)
    
    def _nelder_mead_local_search(self, x, func, rho, max_iter=100):
        """Nelder-Mead simplex local search on a single candidate."""
        alpha = 1.0
        gamma = 2.0
        sigma = 0.5
        
        # Build initial simplex
        simplex = np.empty((self.dim + 1, self.dim))
        simplex[0] = x
        for i in range(self.dim):
            y = x.copy()
            y[i] += rho * (self.upper - self.lower) if y[i] < 0 else -rho * (self.upper - self.lower)
            y = np.clip(y, self.lower, self.upper)
            simplex[i + 1] = y
        
        # Evaluate simplex
        fit_vals = func(simplex)
        if len(fit_vals) < len(simplex):
            fit_vals = np.resize(fit_vals, len(simplex))
            fit_vals[len(fit_vals):] = np.inf
        
        for _ in range(max_iter):
            # Sort by fitness
            sorted_idx = np.argsort(fit_vals)
            simplex = simplex[sorted_idx]
            fit_vals = fit_vals[sorted_idx]
            
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = x_centroid + alpha * (x_centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, self.lower, self.upper)
            reflect_fitness = func(x_reflect.reshape(1, -1))[0]
            
            if fit_vals[0] <= reflect_fitness < fit_vals[-2]:
                simplex[-1] = x_reflect
                fit_vals[-1] = reflect_fitness
                continue
            
            # Expansion
            if reflect_fitness < fit_vals[0]:
                x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                x_expand = np.clip(x_expand, self.lower, self.upper)
                expand_fitness = func(x_expand.reshape(1, -1))[0]
                if expand_fitness < reflect_fitness:
                    simplex[-1] = x_expand
                    fit_vals[-1] = expand_fitness
                else:
                    simplex[-1] = x_reflect
                    fit_vals[-1] = reflect_fitness
                continue
            
            # Contraction
            x_contract = x_centroid - sigma * (x_centroid - simplex[-1])
            x_contract = np.clip(x_contract, self.lower, self.upper)
            contract_fitness = func(x_contract.reshape(1, -1))[0]
            
            if contract_fitness < fit_vals[-1]:
                simplex[-1] = x_contract
                fit_vals[-1] = contract_fitness
            else:
                # Shrink
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                simplex = self._clip_to_bounds_batch(simplex)
                fit_vals[1:] = func(simplex[1:])
                if len(fit_vals) < len(simplex):
                    fit_vals = np.resize(fit_vals, len(simplex))
                    fit_vals[len(fit_vals):] = np.inf
        
        # Return best
        best_idx = np.argmin(fit_vals)
        return simplex[best_idx], fit_vals[best_idx]
    
    def _apply_local_search_batch(self, pop, fitness, func, gen):
        """Apply Nelder-Mead to top candidates periodically."""
        if gen > 0 and gen % self.ls_interval != 0:
            return pop, fitness
        
        n_ls = max(1, self.np // 10)  # Apply to top 10%
        sorted_idx = np.argsort(fitness)
        top_idx = sorted_idx[:n_ls]
        
        for idx in top_idx:
            x_improved, f_improved = self._nelder_mead_local_search(
                pop[idx], func, self.ls_rho
            )
            if f_improved < fitness[idx]:
                pop[idx] = x_improved
                fitness[idx] = f_improved
        
        return pop, fitness
    
    def _compute_diversity(self, pop):
        """Compute population diversity (average pairwise distance)."""
        if len(pop) < 2:
            return 1.0
        diff = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        upper_tri = distances[np.triu_indices(len(pop), k=1)]
        return np.mean(upper_tri)
    
    def _restart_if_stagnant(self, pop, fitness, stagnant_gen):
        """Restart population if stagnated or diversity is too low."""
        if stagnant_gen < self.max_stagnant_generations:
            return pop, False
        
        diversity = self._compute_diversity(pop)
        if diversity > self.min_diversity_threshold:
            return pop, False
        
        # Partial restart: keep best 20%
        n_keep = max(1, self.np // 5)
        sorted_idx = np.argsort(fitness)
        keep_idx = sorted_idx[:n_keep]
        
        new_pop = pop[keep_idx].copy()
        n_new = self.np - n_keep
        new_individuals = np.random.uniform(self.lower, self.upper, (n_new, self.dim))
        new_pop = np.vstack([new_pop, new_individuals])
        new_pop = self._clip_to_bounds_batch(new_pop)
        
        # Reset adaptive parameters
        self.mu_cr = 0.5
        self.mu_f = 0.5
        self.archive = []
        
        return new_pop, True
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        pop = self._initialize_population()
        fitness = self._eval_wrapper(pop, func)
        
        # Track best
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx].copy()
        
        gen = 0
        stagnant_gen = 0
        successful_cr_history = []
        successful_f_history = []
        strategy_success = [0, 0, 0]
        
        while not stopping_condition():
            # Select mutation strategies
            strategies, cr, f = self._select_mutation_strategy_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(pop, fitness, strategies, f)
            mutants = self._clip_to_bounds_batch(mutants)
            
            # Crossover
            trials = self._crossover_batch(pop, mutants, cr)
            trials = self._clip_to_bounds_batch(trials)
            
            # Check budget after trial generation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = self._eval_wrapper(trials, func)
            
            if len(trial_fitness) < len(trials):
                valid_mask = np.arange(len(trials)) < len(trial_fitness)
                trials = trials[valid_mask]
                trial_fitness = trial_fitness[:sum(valid_mask)]
                pop = pop[valid_mask]
                fitness = fitness[valid_mask]
                if len(pop) == 0:
                    break
            
            if stopping_condition():
                break
            
            # Selection
            pop, fitness = self._select_survivors_batch(pop, fitness, trials, trial_fitness)
            
            # Track successful adaptations for JADE
            better_mask = trial_fitness < fitness[:len(trial_fitness)]
            if np.any(better_mask):
                successful_cr_history.extend(cr[:len(trial_fitness)][better_mask])
                successful_f_history.extend(f[:len(trial_fitness)][better_mask])
                
                for s in range(3):
                    strategy_success[s] += np.sum(strategies[:len(trial_fitness)][better_mask] == s)
                
                self._update_archive_batch(trials, trial_fitness)
            
            # Adapt parameters
            self._adapt_parameters_batch(successful_cr_history, successful_f_history, strategy_success)
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = pop[best_idx].copy()
                stagnant_gen = 0
            else:
                stagnant_gen += 1
            
            # Local search
            pop, fitness = self._apply_local_search_batch(pop, fitness, func, gen)
            
            # Update best after local search
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = pop[best_idx].copy()
            
            # Check for restart
            pop, did_restart = self._restart_if_stagnant(pop, fitness, stagnant_gen)
            if did_restart:
                fitness = self._eval_wrapper(pop, func)
                stagnant_gen = 0
                successful_cr_history = []
                successful_f_history = []
                strategy_success = [0, 0, 0]
            
            gen += 1
            
            # Periodic archive cleanup
            if gen % 100 == 0 and len(self.archive) > self.archive_max_size // 2:
                self.archive = self.archive[:self.archive_max_size // 2]
        
        return self.f_opt, self.x_opt
```