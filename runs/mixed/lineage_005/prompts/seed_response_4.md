```python
import numpy as np


class ArchipelagoDifferentialEvolution:
    """
    Island-model DE with adaptive strategy pool and periodic local search refinement.
    
    Key differences from prior approaches:
    - Spatially-structured island model (3 subpopulations with migration)
    - Multiple mutation strategies in a pool, selected probabilistically by success
    - Periodic Nelder-Mead-like simplex refinement on top individuals
    - Bounded archive of best solutions for diversity
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(8 * dim, 240)  # 3 islands of this size
        self.island_size = self.np // 3
        self.n_islands = 3
        self.bounds = np.array([[-100.0, 100.0]])
        
        # Strategy pool: (name, base_cr, F_range)
        self.strategies = [
            ('current_to_pbest', 0.8, (0.3, 1.0)),
            ('rand', 0.7, (0.4, 1.2)),
            ('best', 0.9, (0.2, 0.8)),
        ]
        self.strategy_scores = np.ones(len(self.strategies))
        self.strategy_counts = np.ones(len(self.strategies))
        self.active_strategy = 0
        
        # Archive for diversity
        self.archive_size = self.np
        self.archive = None
        
        # Local search params
        self.ls_interval = 15
        self.ls_fraction = 0.15
        self.alpha = 1.0
        self.gamma = 0.5
        self.sigma = 0.5
        
        # Migration params
        self.mig_interval = 20
        self.mig_count = 2
        
        # DE parameters
        self.cr = 0.8
        self.F = 0.5
        
        # Tracking
        self.f_opt = np.inf
        self.x_opt = None
        self.gen = 0
        
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        self._update_best(population, fitness)
        
        while not stopping_condition():
            self.gen += 1
            
            # Adapt strategy selection
            self._adapt_strategy_selection()
            
            # Build island structure
            islands = self._split_into_islands(population)
            island_fits = [fitness[i * self.island_size:(i + 1) * self.island_size] 
                          for i in range(self.n_islands)]
            
            # Mutation and crossover for each island
            trials_list = []
            for i in range(self.n_islands):
                trials = self._mutate_batch_island(
                    islands[i], island_fits[i], i, self.archive
                )
                trials = self._crossover_batch(islands[i], trials)
                trials = self._clip_to_bounds_batch(trials)
                trials_list.append(trials)
            
            trials = np.vstack(trials_list)
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                actual = len(trial_fitness)
                trials = trials[:actual]
                trial_fitness = trial_fitness[:actual]
                if len(trials) == 0:
                    break
            
            if stopping_condition():
                break
            
            # Selection
            population, fitness, successes = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update strategy scores
            self._update_strategy_scores(successes)
            
            # Update archive
            self._update_archive_batch(population, fitness)
            
            # Update best
            self._update_best(population, fitness)
            
            # Periodic local search on top individuals
            if self.gen % self.ls_interval == 0:
                population, fitness = self._apply_adaptive_local_search(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
            
            # Periodic migration
            if self.gen % self.mig_interval == 0:
                population = self._migrate_between_islands(population, fitness)
                fitness = func(population)
                self._update_best(population, fitness)
            
            # Restart if stagnant
            population, fitness = self._restart_if_stagnant(
                population, fitness, func
            )
            
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        pop = np.random.uniform(
            self.bounds[0, 0], self.bounds[0, 1],
            size=(self.np, self.dim)
        )
        return pop
    
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.bounds[0, 0], self.bounds[0, 1])
    
    def _split_into_islands(self, population):
        return [
            population[i * self.island_size:(i + 1) * self.island_size]
            for i in range(self.n_islands)
        ]
    
    def _mutate_batch_island(self, island, fit_island, island_idx, archive):
        np_island = len(island)
        trials = np.empty((np_island, self.dim))
        
        # Select mutation type based on active strategy
        strat_name = self.strategies[self.active_strategy][0]
        cr = self.strategies[self.active_strategy][1]
        
        if strat_name == 'current_to_pbest':
            for i in range(np_island):
                pbest_idx = np.argmin(fit_island)
                candidates = [j for j in range(np_island) if j != i and j != pbest_idx]
                if len(candidates) >= 3:
                    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                else:
                    r1, r2, r3 = np.random.choice(np_island, 3, replace=True)
                
                pbest = island[pbest_idx]
                base = island[i]
                
                F = np.random.uniform(*self.strategies[self.active_strategy][2])
                mutant = base + F * (pbest - base) + F * (island[r1] - island[r2])
                trials[i] = mutant
                
        elif strat_name == 'rand':
            for i in range(np_island):
                idxs = [j for j in range(np_island) if j != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False) if len(idxs) >= 3 \
                            else np.random.choice(np_island, 3, replace=True)
                
                F = np.random.uniform(*self.strategies[self.active_strategy][2])
                trials[i] = island[r1] + F * (island[r2] - island[r3])
                
        else:  # best
            best_idx = np.argmin(fit_island)
            best = island[best_idx]
            for i in range(np_island):
                idxs = [j for j in range(np_island) if j != i and j != best_idx]
                if len(idxs) >= 2:
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                else:
                    r1, r2 = np.random.choice(np_island, 2, replace=True)
                
                F = np.random.uniform(*self.strategies[self.active_strategy][2])
                trials[i] = best + F * (island[r1] - island[r2])
        
        return trials
    
    def _crossover_batch(self, pop, mutants):
        np_pop = len(pop)
        cr = self.strategies[self.active_strategy][1]
        trials = np.copy(mutants)
        
        # Vectorized binomial crossover
        mask = np.random.rand(np_pop, self.dim) < cr
        # Ensure at least one dimension comes from mutant
        rnd_dim = np.random.randint(0, self.dim, size=np_pop)
        mask[np.arange(np_pop), rnd_dim] = True
        
        trials = np.where(mask, mutants, pop)
        return trials
    
    def _select_survivors_batch(self, pop, fit, trials, trial_fit):
        successes = trial_fit < fit
        survivors = np.where(successs.reshape(-1, 1), trials, pop)
        new_fit = np.where(successes, trial_fit, fit)
        return survivors, new_fit, successes
    
    def _adapt_strategy_selection(self):
        # Softmax-based selection with temperature
        scores = self.strategy_scores / (self.strategy_counts + 1e-8)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        self.active_strategy = np.random.choice(len(probs), p=probs)
    
    def _update_strategy_scores(self, successes):
        reward = successes.sum() / len(successes)
        self.strategy_scores[self.active_strategy] += reward
        self.strategy_counts[self.active_strategy] += 1
    
    def _update_archive_batch(self, pop, fit):
        if self.archive is None:
            self.archive = pop[fit.argsort()[:self.archive_size // 2]].copy()
        else:
            combined = np.vstack([self.archive, pop])
            combined_fit = np.concatenate([
                np.full(len(self.archive), -np.inf), fit
            ])
            best_idx = combined_fit.argsort()[:self.archive_size]
            self.archive = combined[best_idx]
    
    def _apply_adaptive_local_search(self, pop, fit, func, stopping_condition):
        n_ls = max(1, int(len(pop) * self.ls_fraction))
        top_idx = fit.argsort()[:n_ls]
        
        ls_pop = pop[top_idx].copy()
        ls_fit = fit[top_idx].copy()
        
        for i in range(len(ls_pop)):
            if stopping_condition():
                break
            ls_pop[i], ls_fit[i] = self._nelder_mead_step(
                ls_pop[i], ls_fit[i], func
            )
        
        # Create simplex around each LS point
        simplex_batch = np.zeros((len(ls_pop), self.dim + 1, self.dim))
        for i in range(len(ls_pop)):
            simplex_batch[i, 0] = ls_pop[i]
            for j in range(self.dim):
                simplex_batch[i, j + 1] = ls_pop[i].copy()
                simplex_batch[i, j + 1, j] += np.random.uniform(-0.5, 0.5)
            simplex_batch[i] = self._clip_to_bounds_batch(simplex_batch[i])
        
        # Evaluate simplexes
        simplex_flat = simplex_batch.reshape(-1, self.dim)
        simplex_fit = func(simplex_flat)
        
        if len(simplex_fit) < len(simplex_flat):
            simplex_fit = simplex_fit[:len(simplex_flat)]
        
        simplex_fit = simplex_fit.reshape(len(ls_pop), -1)
        
        # Update best from each simplex
        for i in range(len(ls_pop)):
            best_simplex_idx = np.argmin(simplex_fit[i])
            if simplex_fit[i, best_simplex_idx] < ls_fit[i]:
                ls_pop[i] = simplex_batch[i, best_simplex_idx]
                ls_fit[i] = simplex_fit[i, best_simplex_idx]
        
        # Merge back
        pop[top_idx] = ls_pop
        fit[top_idx] = ls_fit
        
        return pop, fit
    
    def _nelder_mead_step(self, x, fx, func):
        # Simple 1-step Nelder-Mead reflection
        alpha, gamma, sigma = self.alpha, self.gamma, self.sigma
        n = len(x)
        
        # Generate simplex
        simplex = np.tile(x, (n + 1, 1))
        for i in range(n):
            simplex[i + 1] = x.copy()
            simplex[i + 1, i] += np.random.uniform(0.05, 0.5)
        simplex = self._clip_to_bounds_batch(simplex)
        
        # Evaluate
        simplex_fit = func(simplex)
        if len(simplex_fit) < len(simplex):
            simplex_fit = simplex_fit[:len(simplex)]
        
        best_idx = np.argmin(simplex_fit)
        x_new = simplex[best_idx]
        fx_new = simplex_fit[best_idx]
        
        return x_new, fx_new
    
    def _migrate_between_islands(self, population, fitness):
        islands = self._split_into_islands(population)
        island_fits = [fitness[i * self.island_size:(i + 1) * self.island_size] 
                      for i in range(self.n_islands)]
        
        # Send best from each island to next
        for src in range(self.n_islands):
            dst = (src + 1) % self.n_islands
            best_src_idx = np.argmin(island_fits[src])
            migrants = islands[src][best_src_idx : best_src_idx + self.mig_count]
            
            # Replace worst in destination
            worst_dst_idxs = np.argsort(island_fits[dst])[-self.mig_count:]
            islands[dst][worst_dst_idxs] = migrants
        
        return np.vstack(islands)
    
    def _restart_if_stagnant(self, pop, fit, func):
        if self.gen < 50:
            return pop, fit
        
        best_fit_recent = fit.min()
        if not hasattr(self, '_best_fit_history'):
            self._best_fit_history = []
        
        self._best_fit_history.append(best_fit_recent)
        if len(self._best_fit_history) > 50:
            self._best_fit_history.pop(0)
        
        if len(self._best_fit_history) >= 50:
            improvement = self._best_fit_history[0] - self._best_fit_history[-1]
            if improvement < 1e-6:
                # Random reinitialization of half the population
                n_replace = self.np // 2
                replace_idx = np.random.choice(self.np, n_replace, replace=False)
                pop[replace_idx] = np.random.uniform(
                    self.bounds[0, 0], self.bounds[0, 1],
                    size=(n_replace, self.dim)
                )
                fit = func(pop)
                self._best_fit_history = []
                self._strategy_scores = np.ones(len(self.strategies))
        
        return pop, fit
    
    def _update_best(self, pop, fit):
        best_idx = np.argmin(fit)
        if fit[best_idx] < self.f_opt:
            self.f_opt = fit[best_idx]
            self.x_opt = pop[best_idx].copy()
    
    def _compute_diversity(self, pop):
        centroid = pop.mean(axis=0)
        return np.mean(np.linalg.norm(pop - centroid, axis=1))
```
