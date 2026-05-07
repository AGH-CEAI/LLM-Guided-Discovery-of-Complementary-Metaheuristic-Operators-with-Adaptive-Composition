```python
import numpy as np


class CovarianceAdaptiveSwarmOptimizer:
    """
    Hybrid optimizer combining CMA-ES-style covariance adaptation with
    PSO-inspired velocity updates across multiple subpopulations.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.np = min(max(4 * dim, 60), 8 * dim)
        self.n_subpops = max(4, dim // 15)
        self.ind_per_subpop = self.np // self.n_subpops
        self.n_elite = max(2, self.ind_per_subpop // 4)
        self.mix_interval = 7 + dim // 10
        self.vel_scale = 0.15
        self.inertia = 0.5
        self.cognitive = 1.2
        self.social = 1.5
        self.mu_eff = self.n_elite ** 2 / 4.0
        self.cc = 1.0 / (self.dim ** 0.5 + 6)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + 2 * self.mu_eff))
        self.sigma = np.full(self.n_subpops, 5.0)
        self.evolution_paths = np.zeros((self.n_subpops, self.dim))
        self.generation = 0
        self.last_improvement = np.zeros(self.n_subpops, dtype=int)
        self.stagnation_threshold = 50 + 10 * dim
        self.best_fitness = np.inf
        self.best_position = None

    def _get_subpop_bounds(self, idx):
        """Return (start, end) slice indices for subpopulation."""
        start = idx * self.ind_per_subpop
        return start, start + self.ind_per_subpop

    def _initialize_population(self):
        """Initialize population across subpopulations."""
        self.population = np.zeros((self.np, self.dim))
        for i in range(self.n_subpops):
            start, end = self._get_subpop_bounds(i)
            samples = self._sample_subpopulation(i, end - start)
            self.population[start:end] = np.clip(samples, self.lower_bound, self.upper_bound)
        self.fitness = np.full(self.np, np.inf)

    def _sample_subpopulation(self, subpop_idx, n_individuals):
        """Sample individuals using CMA-ES style Cholesky sampling."""
        mean = self.means[subpop_idx]
        cov = self.cov_matrices[subpop_idx].copy()
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov += np.eye(self.dim) * 1e-6
            L = np.linalg.cholesky(cov)
        z = np.random.randn(n_individuals, self.dim)
        return mean + self.sigma[subpop_idx] * (z @ L.T)

    def _position_update_batch(self, trials, subpop_idx, start, end):
        """Apply velocity-based position update to trial batch."""
        subpop_pop = self.population[start:end]
        global_best = self.best_position if self.best_position is not None else self.means[subpop_idx]
        subpop_best = subpop_pop[np.nanargmin(self.fitness[start:end])]
        inertia_term = self.inertia * (subpop_pop - self.prev_population[start:end])
        cognitive_term = self.cognitive * np.random.rand() * (subpop_best - subpop_pop)
        social_term = self.social * np.random.rand() * (global_best - subpop_pop)
        velocity = inertia_term + cognitive_term + social_term
        updated = subpop_pop + velocity * self.vel_scale
        return np.clip(updated, self.lower_bound, self.upper_bound)

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Elitist selection combining parent and trial populations."""
        combined_pop = np.vstack([population, trials])
        combined_fit = np.concatenate([fitness, trial_fitness])
        sorted_idx = np.argsort(combined_fit)[:self.np]
        return combined_pop[sorted_idx], combined_fit[sorted_idx]

    def _adapt_step_size(self, subpop_idx):
        """Adapt step size based on evolution path norm."""
        ps_norm = np.linalg.norm(self.evolution_paths[subpop_idx])
        expected = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        if ps_norm < 1.5 * expected:
            self.sigma[subpop_idx] *= np.exp(0.2)
        elif ps_norm > 1.5 * expected:
            self.sigma[subpop_idx] *= np.exp(-0.2)
        self.sigma[subpop_idx] = np.clip(self.sigma[subpop_idx], 1e-10, 10.0)

    def _adapt_covariance(self, subpop_idx):
        """Update covariance matrix using weighted recombination of elite individuals."""
        start, end = self._get_subpop_bounds(subpop_idx)
        subpop_pop = self.population[start:end]
        subpop_fit = self.fitness[start:end]
        elite_mask = np.argsort(subpop_fit)[:self.n_elite]
        elite = subpop_pop[elite_mask]
        ranks = np.argsort(np.argsort(subpop_fit[:self.n_elite]))
        weights = np.maximum(np.log(self.n_elite + 1) - np.log(ranks + 1), 0)
        weights /= (np.sum(weights) + 1e-30)
        mean_old = self.means[subpop_idx].copy()
        self.means[subpop_idx] = np.sum(elite * weights[:, np.newaxis], axis=0)
        y = (self.means[subpop_idx] - mean_old) / (self.sigma[subpop_idx] + 1e-30)
        hs = np.linalg.norm(self.evolution_paths[subpop_idx]) < 1.5 * np.sqrt(self.dim)
        self.evolution_paths[subpop_idx] = (1 - self.cc) * self.evolution_paths[subpop_idx] + hs * np.sqrt(self.cc * (2 - self.cc)) * np.sqrt(self.mu_eff) * y
        rank_one = np.outer(self.evolution_paths[subpop_idx], self.evolution_paths[subpop_idx])
        self.cov_matrices[subpop_idx] = (1 - self.c1 - self.cmu) * self.cov_matrices[subpop_idx] + self.c1 * rank_one + self.cmu * np.eye(self.dim) * 0.01
        eigvals, eigvecs = np.linalg.eigh(self.cov_matrices[subpop_idx])
        eigvals = np.maximum(eigvals, 1e-10)
        self.cov_matrices[subpop_idx] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _mix_subpopulations(self):
        """Exchange best individuals between neighboring subpopulations."""
        if self.generation % self.mix_interval != 0:
            return
        for i in range(self.n_subpops):
            other = (i + 1) % self.n_subpops
            start_i, end_i = self._get_subpop_bounds(i)
            start_o, end_o = self._get_subpop_bounds(other)
            best_other = self.population[start_o:end_o][np.nanargmin(self.fitness[start_o:end_o])]
            worst_i = start_i + np.nanargmax(self.fitness[start_i:end_i])
            self.population[worst_i] = best_other.copy()

    def _restart_if_stagnant(self, subpop_idx):
        """Reinitialize a stagnating subpopulation."""
        if self.generation - self.last_improvement[subpop_idx] <= self.stagnation_threshold:
            return
        self.means[subpop_idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.cov_matrices[subpop_idx] = np.eye(self.dim) * (self.sigma[subpop_idx] ** 2)
        self.sigma[subpop_idx] = np.random.uniform(2.0, 8.0)
        self.evolution_paths[subpop_idx] = np.zeros(self.dim)
        start, end = self._get_subpop_bounds(subpop_idx)
        self.population[start:end] = self._sample_subpopulation(subpop_idx, end - start)
        self.population[start:end] = np.clip(self.population[start:end], self.lower_bound, self.upper_bound)
        self.last_improvement[subpop_idx] = self.generation

    def __call__(self, func, stopping_condition):
        """Main optimization loop with batched operations."""
        self.func = func
        self._initialize_population()
        self.prev_population = self.population.copy()
        fitness = func(self.population)
        if len(fitness) < self.np:
            valid = len(fitness)
            self.population = self.population[:valid]
            self.fitness = fitness[:valid]
            self.prev_population = self.prev_population[:valid]
        else:
            self.fitness = fitness
        best_idx = np.nanargmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.population[best_idx].copy()
        while not stopping_condition():
            trials = np.zeros_like(self.population)
            for i in range(self.n_subpops):
                start, end = self._get_subpop_bounds(i)
                trials[start:end] = self._sample_subpopulation(i, end - start)
                trials[start:end] = self._position_update_batch(trials[start:end], i, start, end)
            trial_fit = func(trials)
            if len(trial_fit) < len(trials):
                valid = len(trial_fit)
                trials = trials[:valid]
                trial_fit = trial_fit[:valid]
            if stopping_condition():
                break
            fitness_repeat = np.concatenate([self.fitness, np.full(len(trial_fit), np.inf)])[:len(trials)]
            improved = trial_fit < fitness_repeat
            for i in range(self.n_subpops):
                start, end = self._get_subpop_bounds(i)
                if np.any(improved[start:end]):
                    self.last_improvement[i] = self.generation
            any_improved = np.any(improved)
            if any_improved:
                best_idx = np.nanargmin(trial_fit)
                self.best_fitness = trial_fit[best_idx]
                self.best_position = trials[best_idx].copy()
            self.prev_population = self.population.copy()
            self.population, self.fitness = self._select_survivors_batch(self.population, self.fitness, trials, trial_fit)
            for i in range(self.n_subpops):
                self._adapt_step_size(i)
                self._adapt_covariance(i)
            self._mix_subpopulations()
            for i in range(self.n_subpops):
                self._restart_if_stagnant(i)
            self.generation += 1
        return self.best_fitness, self.best_position
```