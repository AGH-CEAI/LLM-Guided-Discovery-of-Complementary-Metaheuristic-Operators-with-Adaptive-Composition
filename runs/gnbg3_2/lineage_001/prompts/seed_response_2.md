```python
import numpy as np


class CovarianceAdaptationMemeticOptimizer:
    """
    Hybrid optimizer combining CMA-ES-style covariance adaptation with
    differential mutation bursts for exploration. Uses island-based
    subpopulations for niching and adaptive parameter control.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = kwargs.get('NP', min(200, max(50, 8 * dim)))
        self.lb = -100.0
        self.ub = 100.0
        
        # Population structure
        self.n_islands = max(2, self.NP // 40)
        self.island_size = self.NP // self.n_islands
        
        # CMA-ES parameters
        self.mu = self.island_size // 2
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        
        # Adaptation learning rates
        self.cc = 4.0 / (dim + 4.0)
        self.c1 = 2.0 / (dim ** 2)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 1) / (dim ** 2 + 2 * self.mu_eff))
        self.cs = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.damps = 1.0 + max(0, np.sqrt(self.mu_eff) - 1) + self.cs
        
        # Evolution paths
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        
        # Strategy parameters
        self.C = np.eye(dim)
        self.sigma = 0.3 * (self.ub - self.lb)
        self.mean = np.zeros(dim)
        
        # Differential mutation parameters
        self.F = 0.5
        self.CR = 0.5
        self.de_prob = 0.3
        
        # Tracking
        self.best_f = np.inf
        self.best_x = None
        self.stagnation_count = 0
        self.stagnation_max = 100 + 50 * dim
        self.generation = 0
        
        # Per-island state
        self.islands = []
        self.island_fitness = []
        self.island_means = []
        self.island_sigmas = []
        self.island_Cs = []
        
        # Budget tracking
        self.func_evals = 0
        
    def _initialize_per_island_state(self, island_idx):
        """Initialize CMA state for a single island."""
        self.island_means[island_idx] = np.random.uniform(self.lb, self.ub, self.dim)
        self.island_sigmas[island_idx] = self.sigma
        self.island_Cs[island_idx] = np.eye(self.dim)
        
    def _initialize_population(self):
        """Initialize island-based population with per-island CMA state."""
        self.islands = []
        self.island_fitness = []
        self.island_means = [np.zeros(self.dim) for _ in range(self.n_islands)]
        self.island_sigmas = [self.sigma for _ in range(self.n_islands)]
        self.island_Cs = [np.zeros((self.dim, self.dim)) for _ in range(self.n_islands)]
        
        for i in range(self.n_islands):
            pop = np.random.uniform(self.lb, self.ub, (self.island_size, self.dim))
            self.islands.append(pop)
            self.island_fitness.append(np.full(self.island_size, np.inf))
            self._initialize_per_island_state(i)
        
        self.mean = np.mean([np.mean(isl, axis=0) for isl in self.islands], axis=0)
        
    def _clip_to_bounds(self, pop):
        """Ensure population stays within bounds."""
        return np.clip(pop, self.lb, self.ub)
    
    def _sample_cma_batch(self, mean, sigma, C, n):
        """Sample n individuals from CMA distribution."""
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.eye(self.dim)
            L = np.eye(self.dim)
        z = np.random.randn(n, self.dim)
        samples = mean + sigma * (z @ L.T)
        return self._clip_to_bounds(samples)
    
    def _generate_de_mutant_batch(self, pop, n):
        """Generate differential mutation vectors."""
        indices = np.random.randint(0, len(pop), (n, 3))
        r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]
        mutants = pop[r1] + self.F * (pop[r2] - pop[r3])
        return self._clip_to_bounds(mutants)
    
    def _crossover_binomial_batch(self, targets, mutants):
        """Binomial crossover between targets and mutants."""
        mask = np.random.random((len(targets), self.dim)) < self.CR
        mask[np.arange(len(targets)), np.random.randint(0, self.dim, len(targets))] = True
        trials = np.where(mask, mutants, targets)
        return trials
    
    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        """Greedy (mu, lambda)-style selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, pop)
        new_fitness = np.where(better, trial_fitness, fitness)
        return new_pop, new_fitness
    
    def _update_island_mean(self, island_idx, pop, fitness):
        """Update island mean using weighted recombination."""
        sorted_idx = np.argsort(fitness)
        selected_idx = sorted_idx[:self.mu]
        weights = self.weights[:self.mu]
        weights = weights / np.sum(weights)
        old_mean = self.island_means[island_idx].copy()
        self.island_means[island_idx] = np.sum(pop[selected_idx] * weights[:, np.newaxis], axis=0)
        return old_mean
    
    def _adapt_step_size_batch(self, island_idx, old_mean, pop):
        """Adapt step size using evolution path."""
        y = (self.island_means[island_idx] - old_mean) / self.island_sigmas[island_idx]
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)) * y
        ps_norm = np.linalg.norm(self.ps)
        expected_norm = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))
        self.island_sigmas[island_idx] *= np.exp((self.cs / self.damps) * (ps_norm / expected_norm - 1.0))
        self.island_sigmas[island_idx] = np.clip(self.island_sigmas[island_idx], 1e-10, 10.0)
        self.sigma = np.mean(self.island_sigmas)
    
    def _adapt_covariance_batch(self, island_idx, old_mean, pop, fitness):
        """Adapt covariance matrix using rank-1 and rank-mu updates."""
        y = (self.island_means[island_idx] - old_mean) / self.island_sigmas[island_idx]
        self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc)) * y
        rank1 = self.c1 * np.outer(self.pc, self.pc)
        sorted_idx = np.argsort(fitness)
        selected_idx = sorted_idx[:self.mu]
        z = (pop[selected_idx] - old_mean) / self.island_sigmas[island_idx]
        rankmu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            rankmu += self.weights[i] * np.outer(z[i], z[i])
        rankmu *= self.cmu
        self.island_Cs[island_idx] = (1 - self.c1 - self.cmu) * self.island_Cs[island_idx] + rank1 + rankmu
        self.island_Cs[island_idx] = (self.island_Cs[island_idx] + self.island_Cs[island_idx].T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.island_Cs[island_idx])
        self.island_Cs[island_idx] = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
        self.C = np.mean(self.island_Cs, axis=0)
    
    def _compute_diversity(self, pop):
        """Compute average pairwise Euclidean distance."""
        n = len(pop)
        if n < 2:
            return 0.0
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.mean(distances[np.triu_indices(n, k=1)])
    
    def _adapt_parameters(self, combined_fitness):
        """Adapt DE parameters based on success rates."""
        if self.generation > 0 and self.generation % 10 == 0:
            improvement = np.sum(combined_fitness < self.best_f + 1e-6)
            if improvement > len(combined_fitness) * 0.1:
                self.F = np.clip(self.F * 0.9, 0.1, 2.0)
                self.CR = np.clip(self.CR * 1.1, 0.1, 1.0)
            elif improvement < len(combined_fitness) * 0.01:
                self.F = np.clip(self.F * 1.1, 0.1, 2.0)
                self.CR = np.clip(self.CR * 0.9, 0.1, 1.0)
    
    def _check_stagnation(self, combined_fitness):
        """Check for stagnation and trigger restart if needed."""
        current_best = np.min(combined_fitness)
        if current_best < self.best_f - 1e-10:
            self.best_f = current_best
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        if self.stagnation_count > self.stagnation_max:
            self.sigma = 0.5 * (self.ub - self.lb)
            self.C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
            self.stagnation_count = 0
            return True
        return False
    
    def _restart_if_stagnant(self, combined_fitness):
        """Public method wrapper for stagnation check."""
        return self._check_stagnation(combined_fitness)
    
    def _migrate_between_islands(self):
        """Ring topology migration between islands."""
        if self.n_islands < 2:
            return
        n_migrants = max(1, self.island_size // 5)
        for i in range(self.n_islands):
            source = (i - 1) % self.n_islands
            best_idx_source = np.argsort(self.island_fitness[source])[:n_migrants]
            worst_idx_target = np.argsort(self.island_fitness[i])[-n_migrants:]
            for j, (src_idx, tgt_idx) in enumerate(zip(best_idx_source, worst_idx_target)):
                self.islands[i][tgt_idx] = self.islands[source][src_idx].copy()
                self.island_fitness[i][tgt_idx] = self.island_fitness[source][src_idx]
    
    def _evolve_single_island(self, island_idx, func):
        """Evolve one island for one generation."""
        pop = self.islands[island_idx]
        fitness = self.island_fitness[island_idx]
        
        old_mean = self._update_island_mean(island_idx, pop, fitness)
        self._adapt_step_size_batch(island_idx, old_mean, pop)
        self._adapt_covariance_batch(island_idx, old_mean, pop, fitness)
        
        n_cma = int(self.island_size * (1 - self.de_prob))
        n_de = self.island_size - n_cma
        
        cma_trials = self._sample_cma_batch(
            self.island_means[island_idx], 
            self.island_sigmas[island_idx], 
            self.island_Cs[island_idx], 
            n_cma
        )
        
        de_base = pop[:n_de]
        de_mutants = self._generate_de_mutant_batch(de_base, n_de)
        de_trials = self._crossover_binomial_batch(de_base, de_mutants)
        
        trials = np.vstack([cma_trials, de_trials])
        trials = self._clip_to_bounds(trials)
        
        trial_fitness = func(trials)
        
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
            trial_fitness = trial_fitness[:len(trial_fitness)]
        
        if len(trials) > 0:
            self.islands[island_idx], self.island_fitness[island_idx] = \
                self._select_survivors_batch(pop, fitness, trials, trial_fitness)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_population()
        self.func_evals = 0
        
        while not stopping_condition():
            all_fitness = []
            for i in range(self.n_islands):
                island_fit = func(self.islands[i])
                all_fitness.append(island_fit)
                self.island_fitness[i] = island_fit
            
            self.func_evals += sum(len(f) for f in all_fitness)
            
            if len(all_fitness) == 0 or all(len(f) == 0 for f in all_fitness):
                break
            
            if stopping_condition():
                break
            
            combined_pop = np.vstack(self.islands)
            combined_fitness = np.concatenate(all_fitness)
            
            best_idx = np.argmin(combined_fitness)
            self.best_f = combined_fitness[best_idx]
            self.best_x = combined_pop[best_idx].copy()
            
            self._adapt_parameters(combined_fitness)
            
            for i in range(self.n_islands):
                self._evolve_single_island(i, func)
            
            if self.generation % 20 == 0 and self.generation > 0:
                self._migrate_between_islands()
            
            self.generation += 1
            
            if self._restart_if_stagnant(combined_fitness):
                self._initialize_population()
        
        return self.best_f, self.best_x
```