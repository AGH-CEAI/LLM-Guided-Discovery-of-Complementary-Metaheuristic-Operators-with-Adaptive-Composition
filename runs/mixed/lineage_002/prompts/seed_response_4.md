```python
import numpy as np


class AdaptiveSubspaceMemeticOptimizer:
    """
    A hybrid optimizer combining DE/current-to-pbest/1 mutation with
    periodic Nelder-Mead local search, archipelago sub-population topology,
    opposition-based initialization, and adaptive parameter control.
    """
    
    def __init__(self, dim, NP=None, F=0.5, Cr=0.9, p_best=0.1,
                 local_search_interval=5, local_search_fraction=0.05,
                 migration_interval=15, n_subpops=4,
                 stagnation_threshold=30, diversity_threshold=1e-6):
        self.dim = dim
        self.NP = NP if NP is not None else min(max(4 * dim, 30), 200)
        self.F = F
        self.Cr = Cr
        self.p_best = p_best
        self.local_search_interval = local_search_interval
        self.local_search_fraction = local_search_fraction
        self.migration_interval = migration_interval
        self.n_subpops = n_subpops
        self.stagnation_threshold = stagnation_threshold
        self.diversity_threshold = diversity_threshold
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self._validate_config()
    
    def _validate_config(self):
        assert self.dim > 0, "dim must be positive"
        assert self.NP >= 4, "NP must be at least 4"
        assert 0 < self.p_best <= 1.0, "p_best must be in (0, 1]"
        self.subpop_size = self.NP // self.n_subpops
    
    def _initialize_population(self, func):
        """Opposition-based population initialization"""
        uniform = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.NP, self.dim)
        )
        opposite = self.bounds[:, 0] + self.bounds[:, 1] - uniform
        combined = np.vstack([uniform, opposite])
        if len(combined) > self.NP:
            indices = np.argsort(func(combined))[:self.NP]
            combined = combined[indices]
        self.population = np.clip(combined, self.bounds[:, 0], self.bounds[:, 1])
        self.fitness = func(self.population)
        self._update_best()
        self._setup_subpopulations()
    
    def _setup_subpopulations(self):
        """Partition population into archipelago sub-populations"""
        perm = np.random.permutation(self.NP)
        self.subpops = [
            perm[i * self.subpop_size:(i + 1) * self.subpop_size]
            for i in range(self.n_subpops)
        ]
    
    def _update_best(self):
        """Track best solution across population"""
        valid = np.where(np.isfinite(self.fitness))[0]
        if len(valid) == 0:
            return
        best_local = np.argmin(self.fitness[valid])
        best_idx = valid[best_local]
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_x = self.population[best_idx].copy()
    
    def _mutate_current_to_pbest_batch(self, indices):
        """DE/current-to-pbest/1 mutation for sub-population"""
        target_pop = self.population[indices]
        n = len(indices)
        p_best_size = max(1, int(self.p_best * self.NP))
        p_best_pool = np.argsort(self.fitness)[:p_best_size]
        p_best = self.population[np.random.choice(p_best_pool, n)]
        r1 = np.random.randint(0, self.NP, size=n)
        r2 = np.random.randint(0, self.NP, size=n)
        donors = p_best + self.F * (target_pop - p_best) + self.F * (self.population[r1] - self.population[r2])
        return donors
    
    def _crossover_binomial_batch(self, targets, donors):
        """Binomial crossover between targets and donors"""
        n, d = donors.shape
        j_rand = np.random.randint(0, d, size=n)
        mask = np.random.uniform(0, 1, (n, d)) < self.Cr
        mask[np.arange(n), j_rand] = True
        trials = np.where(mask, donors, targets)
        return trials
    
    def _clip_to_bounds_batch(self, batch):
        """Clip population to search bounds"""
        return np.clip(batch, self.bounds[:, 0], self.bounds[:, 1])
    
    def _select_survivors_batch(self, indices, trials, trial_fitness):
        """Tournament selection: replace if trial is better"""
        current_fitness = self.fitness[indices]
        better_mask = trial_fitness < current_fitness
        better_indices = indices[better_mask]
        self.population[better_indices] = trials[better_mask]
        self.fitness[better_indices] = trial_fitness[better_mask]
        return np.sum(better_mask)
    
    def _apply_nelder_mead_local_search_batch(self, func):
        """Apply Nelder-Mead refinement to top performers"""
        n_refine = max(1, int(self.local_search_fraction * self.NP))
        top_indices = np.argsort(self.fitness)[:n_refine]
        n_refined = 0
        for idx in top_indices:
            x_base = self.population[idx].copy()
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            simplex = np.tile(x_base, (self.dim + 1, 1))
            simplex[1:] += np.random.uniform(-0.5, 0.5, (self.dim, self.dim))
            simplex = self._clip_to_bounds_batch(simplex)
            simplex_fitness = func(simplex)
            for _ in range(15):
                sort_idx = np.argsort(simplex_fitness)
                simplex, simplex_fitness = simplex[sort_idx], simplex_fitness[sort_idx]
                centroid = np.mean(simplex[:-1], axis=0)
                reflected = self._clip_to_bounds_batch(centroid + alpha * (centroid - simplex[-1]))
                r_fitness = func(reflected.reshape(1, -1))[0]
                if r_fitness < simplex_fitness[0]:
                    expanded = self._clip_to_bounds_batch(centroid + gamma * (reflected - centroid))
                    e_fitness = func(expanded.reshape(1, -1))[0]
                    if e_fitness < r_fitness:
                        simplex[-1], simplex_fitness[-1] = expanded, e_fitness
                    else:
                        simplex[-1], simplex_fitness[-1] = reflected, r_fitness
                elif r_fitness < simplex_fitness[-2]:
                    simplex[-1], simplex_fitness[-1] = reflected, r_fitness
                else:
                    contracted = self._clip_to_bounds_batch(centroid + rho * (simplex[-1] - centroid))
                    c_fitness = func(contracted.reshape(1, -1))[0]
                    if c_fitness < simplex_fitness[-1]:
                        simplex[-1], simplex_fitness[-1] = contracted, c_fitness
                    else:
                        shrink = simplex[0] + sigma * (simplex[1:] - simplex[0])
                        simplex[1:] = self._clip_to_bounds_batch(shrink)
                        simplex_fitness[1:] = func(simplex[1:])
            best_local_idx = np.argmin(simplex_fitness)
            if simplex_fitness[best_local_idx] < self.fitness[idx]:
                self.population[idx] = simplex[best_local_idx]
                self.fitness[idx] = simplex_fitness[best_local_idx]
                n_refined += 1
        return n_refined
    
    def _migrate_subpopulations(self):
        """Ring migration: best from each sub-pop replaces worst in next"""
        for i in range(self.n_subpops):
            next_i = (i + 1) % self.n_subpops
            best_sub = self.subpops[i][np.argmin(self.fitness[self.subpops[i]])]
            worst_next = self.subpops[next_i][np.argmax(self.fitness[self.subpops[next_i]])]
            self.population[worst_next] = self.population[best_sub].copy()
            self.fitness[worst_next] = self.fitness[best_sub]
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and Cr based on improvement rate"""
        if improvement_rate > 0.25:
            self.F = min(1.2, self.F * 1.1)
            self.Cr = min(0.95, self.Cr * 1.05)
        elif improvement_rate < 0.05:
            self.F = max(0.3, self.F * 0.9)
            self.Cr = max(0.3, self.Cr * 0.95)
    
    def _compute_diversity(self):
        """Compute average pairwise Euclidean distance for diversity"""
        if self.NP < 2:
            return 0.0
        sample_size = min(100, self.NP)
        sample_idx = np.random.choice(self.NP, sample_size, replace=False)
        sample = self.population[sample_idx]
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        upper_tri = distances[np.triu_indices(sample_size, k=1)]
        return np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
    
    def _restart_if_stagnant(self, gen, last_improvement_gen, func):
        """Restart population if stagnant or low diversity"""
        stagnant = gen - last_improvement_gen > self.stagnation_threshold
        low_diversity = self._compute_diversity() < self.diversity_threshold
        if stagnant or low_diversity:
            self._initialize_population(func)
            return 0
        return last_improvement_gen
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop"""
        self.best_fitness = np.inf
        self.best_x = np.zeros(self.dim)
        self._initialize_population(func)
        gen = 0
        last_improvement_gen = 0
        recent_improvements = []
        while not stopping_condition():
            trials_list = []
            for subpop in self.subpops:
                mutants = self._mutate_current_to_pbest_batch(subpop)
                trials = self._crossover_binomial_batch(self.population[subpop], mutants)
                trials = self._clip_to_bounds_batch(trials)
                trials_list.append(trials)
            all_trials = np.vstack(trials_list)
            trial_fitness = func(all_trials)
            if len(trial_fitness) < len(all_trials):
                all_trials = all_trials[:len(trial_fitness)]
            if len(trial_fitness) == 0:
                break
            all_idx = np.concatenate(self.subpops)[:len(trial_fitness)]
            n_improved = self._select_survivors_batch(all_idx, all_trials, trial_fitness)
            self._update_best()
            if stopping_condition():
                break
            if gen % self.local_search_interval == 0:
                self._apply_nelder_mead_local_search_batch(func)
                self._update_best()
            if gen % self.migration_interval == 0:
                self._migrate_subpopulations()
            recent_improvements.append(n_improved)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
            improvement_rate = np.mean(recent_improvements) / self.NP if recent_improvements else 0
            self._adapt_parameters(improvement_rate)
            if self.best_fitness < self.fitness[all_idx].min():
                last_improvement_gen = gen
            last_improvement_gen = self._restart_if_stagnant(gen, last_improvement_gen, func)
            gen += 1
        return self.best_fitness, self.best_x
```