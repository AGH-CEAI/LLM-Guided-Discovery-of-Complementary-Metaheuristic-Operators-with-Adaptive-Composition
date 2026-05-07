Below is the lineage champion metaheuristic. You will plan this lineage by
choosing EXACTLY 3 methods from the CANDIDATE list. Those 3 methods will
each be adapted in a separate cycle, so your ranking determines impact.

Focus on maximizing improvement in final convergence accuracy (target <= 1e-8).

CRITICAL — USE THE ROLE METADATA:
The candidate list below includes a 'role' tag and signals for every method.
Roles indicate where in the algorithm the method sits:

  - hot_path     : modifies the search distribution OR evaluates fitness.
                   These methods directly drive search — HIGHEST priority.
  - leaf         : called from the main loop but does not directly modify
                   the search distribution. Medium priority.
  - control_flow : orchestrates other methods (typically __call__-like).
                   Low priority — usually too small or too generic.
  - observer     : returns a value, does NOT modify state, does NOT call
                   func. These are MEASUREMENTS (e.g. _compute_diversity).
                   LOWEST priority — almost always wrong to pick. Adapting
                   how the algorithm *measures itself* rarely changes how
                   it *searches*. Pick an observer ONLY if you can
                   explicitly justify a feedback loop where its output
                   drives a hot-path decision.

CANDIDATE methods with metadata (you MUST pick exactly 3 distinct names):
  name                                         role          loc  signals
  -------------------------------------------- ------------ ---- -----------------------------------
  _mutate_rand                                 observer     4    PURE_observer
  _mutate_two_rail                             observer     4    PURE_observer
  _initialize_population                       observer     5    in_main_loop  PURE_observer  called_by=[_restart_if_needed, __call__]
  _compute_fitness_ranking                     observer     5    in_main_loop  PURE_observer  called_by=[_mutate_compass_batch]
  _mutate_best                                 observer     5    PURE_observer
  _mutate_current_to_rand_best                 observer     5    PURE_observer
  _select_strategy_batch                       observer     5    in_main_loop  PURE_observer  called_by=[_mutate_ensemble_batch]
  _compute_diversity                           observer     5    in_main_loop  PURE_observer  called_by=[__call__]
  _mutate_local_ring                           observer     6    PURE_observer
  _select_survivors_batch                      observer     6    in_main_loop  PURE_observer  called_by=[__call__]
  _crossover_batch                             observer     7    in_main_loop  PURE_observer  called_by=[__call__]
  _mutate_compass_batch                        observer     9    in_main_loop  PURE_observer  called_by=[__call__]
  _update_archive                              control_flow 9    in_main_loop  called_by=[__call__]
  _check_stagnation                            leaf         9    in_main_loop  called_by=[__call__]
  _update_strategy_scores                      control_flow 10   in_main_loop  called_by=[__call__]
  _mutate_ensemble_batch                       observer     11   in_main_loop  PURE_observer  called_by=[__call__]
  _restart_if_needed                           leaf         16   in_main_loop  called_by=[__call__]
  _adapt_parameters                            leaf         20   in_main_loop  called_by=[__call__]
  _inject_diversity                            observer     22   in_main_loop  PURE_observer  called_by=[__call__]
  _compass_directions                          observer     30   in_main_loop  PURE_observer  called_by=[_mutate_compass_batch]

Signals legend:
  CALLS_func()          — invokes self.func(...) (fitness evaluation)
  MODIFIES_distribution — assigns to self.population/fitness/mean/C/sigma/...
  in_main_loop          — reachable from a loop in __call__
  PURE_observer         — returns a value, doesn't modify any self.X
  called_by=[...]       — methods that call this one

HISTORY of cycles in this lineage so far (most recent last):
(no prior cycles in this lineage)

Current champion source:
```python
import numpy as np


class AdaptiveCompassDE:
    """
    Adaptive Compass Differential Evolution.
    
    Novel features:
    1. Compass Mutation: Samples 3 candidate directions per individual,
       evaluates them cheaply via fitness ranking proximity, picks the best.
    2. Adaptive Strategy Pool: 5 mutation strategies with softmax-weighted selection.
    3. Success-rate adaptation for F and CR with momentum.
    4. Diversity-triggered restart to escape local optima.
    5. Archive of recently improved solutions for diversity.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool with softmax-converted success history
        self.strategy_names = ['rand', 'best', 'current_to_rand_best', 'local_ring', 'two_rail']
        self.strategy_scores = np.ones(len(self.strategy_names))
        self._strategy_funcs = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / (self.NP - 1)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        """Sample 3 candidate mutation directions, score by rank improvement potential."""
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        i_rank = fitness_ranks[i]
        
        candidates = []
        # Direction A: DE/rand/1 style
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        # Direction B: DE/best/1 style
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
        # Direction C: Current-to-pbest style with archive
        p = 0.1
        top_p = int(np.ceil(p * self.NP))
        pbest_candidates = np.argsort(fitness_ranks)[:top_p]
        pbest = population[np.random.choice(pbest_candidates)]
        dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
        score_C = i_rank - fitness_ranks[pbest_candidates].min()
        
        candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
        best_dir, _ = max(candidates, key=lambda x: x[1])
        return best_dir
    
    def _mutate_compass_batch(self, population, fitness):
        """Vectorized compass mutation: compute direction scores, select best per individual."""
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        
        for i in range(self.NP):
            mutants[i] = self._compass_directions(population, i, fitness_ranks)
        
        return mutants
    
    def _mutate_rand(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_best(self, population, idx, F):
        best_idx = np.argmin(population.sum(axis=1))
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 2, replace=False)]
        return population[best_idx] + F * (population[r[0]] - population[r[1]])
    
    def _mutate_current_to_rand_best(self, population, idx, F):
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[i] + F * (population[r[0]] - population[i]) + F * (population[r[1]] - population[r[2]])
    
    def _mutate_local_ring(self, population, idx, F):
        left = (idx - 2) % self.NP
        right = (idx + 2) % self.NP
        others = [left, (idx - 1) % self.NP, (idx + 1) % self.NP, right]
        r = np.array(others)[np.random.choice(4, 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_two_rail(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[r[0]] + population[r[1]] + F * (population[r[2]] - population[r[3]]) - population[idx]
    
    def _select_strategy_batch(self):
        """Softmax-weighted strategy selection."""
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        """Use adaptive strategy pool to generate mutants."""
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        
        return mutants, strategy_used
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with dimension-wise CR."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        """Credit assignment: which strategies produced improvements."""
        for s in range(len(self.strategy_scores)):
            mask = (strategy_used == s) & improved
            if mask.sum() > 0:
                self.strategy_scores[s] += 0.1 * mask.sum()
        
        # Decay all scores slightly
        self.strategy_scores *= 0.99
        self.strategy_scores = np.clip(self.strategy_scores, 0.01, 10.0)
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on improvement rate."""
        # Momentum-based F adaptation
        if improvement_rate > 0.25:
            target_F = min(self.F * 1.15, 1.5)
        elif improvement_rate < 0.1:
            target_F = max(self.F * 0.85, 0.2)
        else:
            target_F = self.F
        self.F = 0.7 * self.F + 0.3 * target_F
        
        # CR adaptation toward success
        if improvement_rate > 0.2:
            target_CR = min(self.CR * 1.05, 0.98)
        else:
            target_CR = max(self.CR * 0.95, 0.4)
        self.CR = 0.6 * self.CR + 0.4 * target_CR
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    def _compute_diversity(self, population):
        """Population spread: average pairwise Euclidean distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _update_archive(self, population, fitness, improved_mask):
        """Maintain archive of improved solutions for diversity injection."""
        improved_pop = population[improved_mask]
        if len(improved_pop) > 0:
            self.archive.append(improved_pop)
            total_len = sum(len(a) for a in self.archive)
            while total_len > self.archive_max:
                oldest = self.archive.pop(0)
                total_len -= len(oldest)
    
    def _inject_diversity(self, population, fitness):
        """Replace worst individuals with archive + random samples."""
        if len(self.archive) < 2:
            return population
        
        archive_concat = np.vstack(self.archive)
        if len(archive_concat) < self.NP // 4:
            return population
        
        n_replace = max(self.NP // 5, 2)
        worst_indices = np.argsort(fitness)[-n_replace:]
        
        for idx in worst_indices:
            if np.random.rand() < 0.5 and len(archive_concat) > 0:
                donor_idx = np.random.randint(len(archive_concat))
                donor = archive_concat[donor_idx]
                noise = np.random.randn(self.dim) * 10.0
                population[idx] = np.clip(donor + noise, -100.0, 100.0)
            else:
                population[idx] = np.random.uniform(-100.0, 100.0, self.dim)
        
        return population
    
    def _check_stagnation(self, best_fitness):
        """Detect stagnation and low diversity."""
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
    def _restart_if_needed(self, population, fitness):
        """Full restart with diversity injection."""
        # Keep best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Reinitialize rest
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        self.stagnation_counter = 0
        self.F = 0.6
        self.CR = 0.85
        
        return new_pop, best_fitness, best_solution
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Evaluate
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.full(len(trials), np.nan)
            
            # Selection
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            self._adapt_parameters(improvement_rate)
            
            # Diversity injection
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check
            if self._check_stagnation(f_opt):
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt

```

Reply with JSON only, listing picks in DESCENDING order of expected impact.
For each method you score, the role_match field measures whether the role
makes the method a sensible target — observers should score 0–1 unless you
have a strong feedback-loop justification.

{"reasoning": "Step-by-step comparison of candidate methods using the role metadata to filter out observers and prioritise hot-path methods.", "analysis": {"method_scores": {"method_name": {"role_match": "0-5 \u2014 does the role match what we want to adapt?", "influence": "0-5", "replaceability": "0-5", "precision_impact": "0-5"}}, "top_candidates": ["method_a", "method_b", "..."]}, "reason": "...", "picks": ["...", "...", "..."]}
Field "picks" is the most important here.
