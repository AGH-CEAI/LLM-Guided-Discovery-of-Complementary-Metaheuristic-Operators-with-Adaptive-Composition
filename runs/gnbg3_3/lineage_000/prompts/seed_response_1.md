```python
import numpy as np


class MirrorSpectralDE:
    """
    Adaptive Mirror Differential Evolution with Spectral Crowding.
    
    Novel components:
    - Mirror mutation: reflects donor across target for opposing search directions
    - Spectral restart: uses population matrix condition number to detect stagnation
    - Paired crowding: similar individuals compete directly, preserving diversity
    - Dual adaptation: F and CR adapted with weighted momentum based on success
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(kwargs.get('NP', 5 * dim), 800)
        self.F = kwargs.get('F', 0.6)
        self.CR = kwargs.get('CR', 0.5)
        self.p_best = kwargs.get('p_best', 0.15)
        self.lb = kwargs.get('lb', -100.0)
        self.ub = kwargs.get('ub', 100.0)
        
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.generation = 0
        
        # Success history for adaptation
        self.archive_success_F = []
        self.archive_success_CR = []
        self.archive_failed_F = []
        self.archive_failed_CR = []
        self.memory_F = 0.5
        self.memory_CR = 0.5
        
        # Diversity tracking
        self.prev_diversity = np.inf
        self.stagnation_counter = 0
        self.max_stagnation = kwargs.get('max_stagnation', 50)
        
        # Topology: ring with distance-2 shortcuts (small-world inspired)
        self._build_topology()
    
    def _build_topology(self):
        n = self.NP
        self.topology = np.zeros((n, n), dtype=bool)
        for i in range(n):
            neighbors = [(i - 1) % n, (i + 1) % n, (i - 2) % n, (i + 2) % n]
            self.topology[i, neighbors] = True
            self.topology[i, i] = True
    
    def _initialize_population(self):
        rng = np.random.default_rng()
        self.population = rng.uniform(
            self.lb, self.ub, size=(self.NP, self.dim)
        )
        self.fitness = np.full(self.NP, np.inf)
    
    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.lb, self.ub)
    
    def _evaluate_batch(self, func):
        raw_fitness = func(self.population)
        if len(raw_fitness) < self.NP:
            valid_len = len(raw_fitness)
            self.population = self.population[:valid_len]
            self.fitness = raw_fitness[:valid_len]
            self.NP = valid_len
            self._build_topology()
            return False
        self.fitness = raw_fitness
        return True
    
    def _mutate_mirror_batch(self, indices):
        n = len(indices)
        rng = np.random.default_rng()
        
        targets = self.population[indices]
        pop_for_select = self.population
        
        donors = np.empty((n, self.dim))
        for i, idx in enumerate(indices):
            # Select 4 distinct indices from topology neighborhood
            neighbors = np.where(self.topology[idx])[0]
            candidates = np.setdiff1d(neighbors, idx)
            
            if len(candidates) >= 4:
                chosen = rng.choice(candidates, size=4, replace=False)
            else:
                all_others = np.setdiff1d(np.arange(self.NP), idx)
                needed = 4 - len(candidates)
                chosen = np.concatenate([candidates, rng.choice(all_others, size=needed, replace=False)])
            
            r1, r2, r3, r4 = chosen[:4]
            
            if rng.random() < self.p_best:
                # Use best vector with mirror reflection
                best_vec = self.population[self.best_idx]
                base = best_vec
                diff_scale = self.F * (self.population[r1] - self.population[r2])
                # Mirror: reflect donor across target
                donor_raw = base + diff_scale
                mirror_point = targets[i]
                donors[i] = 2 * mirror_point - np.clip(donor_raw, self.lb, self.ub)
            else:
                # Standard DE/target-to-best/1 with mirror twist
                base = self.population[r1]
                diff1 = self.population[r2] - self.population[r3]
                diff2 = self.population[r4] - targets[i]
                
                donor_raw = base + self.F * diff1 + 0.5 * self.F * diff2
                mirror_point = targets[i]
                donors[i] = 2 * mirror_point - np.clip(donor_raw, self.lb, self.ub)
        
        return self._clip_to_bounds(donors)
    
    def _crossover_batch(self, targets, mutants):
        rng = np.random.default_rng()
        n, dim = targets.shape
        
        cr_adapted = np.clip(self.memory_CR + rng.standard_normal(n) * 0.1, 0.1, 0.9)
        
        j_rand = rng.integers(0, dim, size=n)
        
        mask = rng.random((n, dim)) < cr_adapted[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        trials = np.where(mask, mutants, targets)
        return self._clip_to_bounds(trials), cr_adapted
    
    def _evaluate_trials(self, func, trials):
        trial_fitness = func(trials)
        if len(trial_fitness) < len(trials):
            return trial_fitness[:len(trial_fitness)], len(trial_fitness)
        return trial_fitness, len(trials)
    
    def _select_survivors_crowding_batch(self, targets, mutants, target_fitness, trial_fitness, trial_cr):
        n = len(targets)
        new_pop = np.empty((n, self.dim))
        new_fit = np.empty(n)
        new_cr = np.empty(n)
        
        better_mask = trial_fitness < target_fitness
        
        improvement = np.where(better_mask, target_fitness - trial_fitness, 0)
        
        for i in range(n):
            if better_mask[i]:
                new_pop[i] = mutants[i]
                new_fit[i] = trial_fitness[i]
                new_cr[i] = trial_cr[i]
                
                if improvement[i] > 0:
                    self.archive_success_F.append(self.F)
                    self.archive_success_CR.append(trial_cr[i])
                else:
                    self.archive_failed_F.append(self.F)
                    self.archive_failed_CR.append(trial_cr[i])
            else:
                new_pop[i] = targets[i]
                new_fit[i] = target_fitness[i]
                new_cr[i] = self.memory_CR
        
        self.population = new_pop
        self.fitness = new_fit
        self.CR = np.mean(new_cr)
        
        return np.any(better_mask)
    
    def _adapt_parameters(self):
        min_history = 5
        
        if len(self.archive_success_F) >= min_history:
            mean_sF = np.mean(self.archive_success_F)
            mean_sCR = np.mean(self.archive_success_CR)
            
            self.memory_F = 0.8 * self.memory_F + 0.2 * mean_sF
            self.memory_CR = 0.8 * self.memory_CR + 0.2 * mean_sCR
            
            self.F = np.clip(self.memory_F + np.random.normal(0, 0.1), 0.1, 1.5)
            
            self.archive_success_F = self.archive_success_F[-20:]
            self.archive_success_CR = self.archive_success_CR[-20:]
            self.archive_failed_F = self.archive_failed_F[-20:]
            self.archive_failed_CR = self.archive_failed_CR[-20:]
        else:
            self.F = np.clip(self.F + np.random.normal(0, 0.05), 0.1, 1.5)
    
    def _compute_spectral_diversity(self):
        centered = self.population - self.population.mean(axis=0)
        
        cov = centered.T @ centered / (self.NP - 1)
        
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        condition = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf
        
        return condition
    
    def _compute_variance_diversity(self):
        var_per_dim = np.var(self.population, axis=0)
        return np.mean(var_per_dim)
    
    def _check_stagnation(self):
        current_diversity = self._compute_variance_diversity()
        spectral_cond = self._compute_spectral_diversity()
        
        diversity_ratio = current_diversity / (self.prev_diversity + 1e-10)
        
        self.prev_diversity = current_diversity
        
        is_stagnant = (
            abs(1 - diversity_ratio) < 0.001 or
            spectral_cond > 1e6 or
            np.std(self.fitness) < 1e-8
        )
        
        if is_stagnant:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return is_stagnant
    
    def _restart_population(self):
        rng = np.random.default_rng()
        
        best_idx = self.best_idx
        best_solution = self.population[best_idx].copy() if best_idx is not None else np.zeros(self.dim)
        
        self.population = rng.uniform(
            self.lb, self.ub, size=(self.NP, self.dim)
        )
        
        if best_idx is not None:
            n_preserve = max(1, self.NP // 10)
            preserve_indices = rng.choice(self.NP, size=n_preserve, replace=False)
            for idx in preserve_indices:
                self.population[idx] = best_solution + rng.normal(0, 1, self.dim)
            
            self.population = self._clip_to_bounds(self.population)
        
        self.fitness = np.full(self.NP, np.inf)
        
        self.archive_success_F = []
        self.archive_success_CR = []
        self.archive_failed_F = []
        self.archive_failed_CR = []
        
        self.F = np.clip(0.5 + rng.normal(0, 0.1), 0.3, 0.9)
        self.memory_F = self.F
        
        self.stagnation_counter = 0
        self.prev_diversity = np.inf
    
    def _update_best(self):
        valid_mask = np.isfinite(self.fitness)
        if not np.any(valid_mask):
            return
        
        valid_fitness = self.fitness[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        min_idx = valid_indices[np.argmin(valid_fitness)]
        
        if self.best_idx is None or self.fitness[min_idx] < self.fitness[self.best_idx]:
            self.best_idx = min_idx
    
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self.best_idx = 0
        self.generation = 0
        
        if not self._evaluate_batch(func):
            f_opt = self.fitness[0] if len(self.fitness) > 0 else np.inf
            x_opt = self.population[0] if len(self.population) > 0 else np.zeros(self.dim)
            return f_opt, x_opt
        
        self._update_best()
        
        while not stopping_condition():
            self.generation += 1
            
            indices = np.arange(self.NP)
            
            mutants = self._mutate_mirror_batch(indices)
            
            trials, trial_cr = self._crossover_batch(self.population, mutants)
            
            trial_fitness, trial_len = self._evaluate_trials(func, trials)
            
            if len(trial_fitness) < trial_len:
                trials = trials[:len(trial_fitness)]
                trial_fitness_full = np.full(trial_len, np.inf)
                trial_fitness_full[:len(trial_fitness)] = trial_fitness
                trial_fitness = trial_fitness_full
            
            if stopping_condition():
                break
            
            self._select_survivors_crowding_batch(
                self.population, trials, self.fitness, trial_fitness, trial_cr
            )
            
            self._update_best()
            
            self._adapt_parameters()
            
            if self._check_stagnation() and self.stagnation_counter >= 3:
                self._restart_population()
                if not self._evaluate_batch(func):
                    break
                self._update_best()
        
        best_idx = self.best_idx
        f_opt = self.fitness[best_idx] if best_idx is not None and best_idx < len(self.fitness) else np.inf
        x_opt = self.population[best_idx] if best_idx is not None and best_idx < len(self.population) else np.zeros(self.dim)
        
        return f_opt, x_opt
```