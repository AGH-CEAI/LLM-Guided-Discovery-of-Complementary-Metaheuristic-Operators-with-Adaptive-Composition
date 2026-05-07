```python
import numpy as np


class CovarianceGuidedDE:
    """
    Covariance-Guided Differential Evolution with Composite Mutation.
    Novel features:
    - Covariance matrix tracking of successful mutation vectors
    - Composite mutation: rand/1, best/1, and covariance-guided with adaptive weights
    - Dual-history adaptation for F and CR
    - Velocity-based exploration integrated with DE
    - Stagnation detection with restart from best + perturbation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(10 * dim, 300)
        self.population = np.zeros((self.np, dim))
        self.trial_pop = np.zeros((self.np, dim))
        self.velocity = np.zeros((self.np, dim))
        self.fitness = np.zeros(self.np)
        self.trial_fitness = np.zeros(self.np)
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.bounds = (-100.0, 100.0)
        self.F = 0.5
        self.CR = 0.5
        self.f_history = []
        self.cr_history = []
        self.cov_matrix = np.eye(dim) * 0.1
        self.cov_weight = 0.33
        self.rand_weight = 0.33
        self.best_weight = 0.34
        self.stagnation_count = 0
        self.prev_best = np.inf
        self.successful_vectors = []
        self.step_size = 0.5
        self._generation = 0
    
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self.fitness = self._eval_wrapper(func, self.population)
        self._update_best_batch(self.population, self.fitness)
        while not stopping_condition():
            self._generation += 1
            mutants = self._composite_mutation_batch(self.population, self.fitness)
            self.trial_pop = self._crossover_batch(self.population, mutants)
            self.trial_pop = self._clip_to_bounds_batch(self.trial_pop)
            self.trial_fitness = self._eval_wrapper(func, self.trial_pop)
            if len(self.trial_fitness) < len(self.trial_pop):
                self.trial_pop = self.trial_pop[:len(self.trial_fitness)]
                self.trial_fitness = self._eval_wrapper(func, self.trial_pop)
            if stopping_condition():
                break
            improved = self.trial_fitness < self.fitness
            self._update_successful_vectors(self.population, self.trial_pop, improved)
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, self.trial_pop, self.trial_fitness
            )
            self._update_velocity_batch()
            self._update_best_batch(self.population, self.fitness)
            self._adapt_parameters(improved)
            self._adapt_mutation_weights(improved)
            self._update_covariance_matrix()
            self._adapt_step_size()
            if self._restart_if_stagnant():
                self._initialize_population_from_best()
                self.fitness = self._eval_wrapper(func, self.population)
                self.velocity = np.zeros((self.np, self.dim))
        return self.best_f, self.best_x.copy()
    
    def _initialize_population(self):
        low, high = self.bounds
        self.population = np.random.uniform(low, high, (self.np, self.dim))
        self.velocity = np.zeros((self.np, self.dim))
        self._generation = 0
    
    def _initialize_population_from_best(self):
        low, high = self.bounds
        perturbation_scale = (high - low) * 0.1 * (self.stagnation_count + 1)
        self.population[0] = self.best_x.copy()
        for i in range(1, self.np):
            self.population[i] = self.best_x + np.random.normal(0, perturbation_scale, self.dim)
        self.population = np.clip(self.population, low, high)
        self.stagnation_count = 0
    
    def _eval_wrapper(self, func, pop):
        fit = func(pop)
        if len(fit) < len(pop):
            return fit[:len(fit)]
        return fit
    
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.bounds[0], self.bounds[1])
    
    def _composite_mutation_batch(self, pop, fit):
        low, high = self.bounds
        range_val = high - low
        mutants = np.zeros_like(pop)
        perm = np.random.permutation(self.np)
        idx_list = [perm[i::3] for i in range(3)]
        if len(idx_list[0]) > 0:
            r1, r2, r3 = idx_list[0][:len(idx_list[0])], idx_list[1][:len(idx_list[0])], idx_list[2][:len(idx_list[0])]
            min_len = min(len(r1), len(r2), len(r3))
            r1, r2, r3 = r1[:min_len], r2[:min_len], r3[:min_len]
            mutants[r1] = pop[r1] + self.F * (pop[r2] - pop[r3])
        if len(idx_list[0]) > 0:
            idx_rand = idx_list[0]
            mutants[idx_rand] += self.rand_weight * (mutants[idx_rand] - pop[idx_rand])
        best_idx = np.argmin(fit)
        if len(idx_list[1]) > 0:
            idx_best = idx_list[1][:min_len if len(idx_list[0]) > 0 else len(idx_list[1])]
            mutants[idx_best] = pop[idx_best] + self.F * (self.best_x - pop[idx_best])
            mutants[idx_best] += self.best_weight * (mutants[idx_best] - pop[idx_best])
        if len(idx_list[2]) > 0:
            idx_cov = idx_list[2][:min_len if len(idx_list[0]) > 0 else len(idx_list[2])]
            cov_mutations = np.random.multivariate_normal(
                np.zeros(self.dim), np.abs(self.cov_matrix) + np.eye(self.dim) * 0.01, len(idx_cov)
            )
            mutants[idx_cov] = pop[idx_cov] + self.cov_weight * cov_mutations * range_val * 0.1
        mutants = self._clip_to_bounds_batch(mutants)
        return mutants
    
    def _crossover_batch(self, pop, mutants):
        cr_per_dim = self._get_adaptive_cr()
        trial = np.copy(pop)
        mask = np.random.random((self.np, self.dim)) < cr_per_dim
        j_rand = np.random.randint(0, self.dim, size=self.np)
        for i in range(self.np):
            trial[i, j_rand[i]] = mutants[i, j_rand[i]]
        trial[mask] = mutants[mask]
        return trial
    
    def _get_adaptive_cr(self):
        cr = np.zeros(self.dim)
        for j in range(self.dim):
            if len(self.cr_history) > j and len(self.cr_history[j]) > 0:
                cr[j] = np.mean(self.cr_history[j][-20:])
            else:
                cr[j] = self.CR
        return np.clip(cr, 0.1, 0.95)
    
    def _select_survivors_batch(self, pop, fit, trial_pop, trial_fit):
        improved = trial_fit < fit
        new_pop = np.where(improved[:, np.newaxis], trial_pop, pop)
        new_fit = np.where(improved, trial_fit, fit)
        return new_pop, new_fit
    
    def _update_successful_vectors(self, pop, trial_pop, improved):
        for i in range(self.np):
            if improved[i]:
                diff = trial_pop[i] - pop[i]
                self.successful_vectors.append(diff)
                if len(self.successful_vectors) > 50:
                    self.successful_vectors.pop(0)
    
    def _update_covariance_matrix(self):
        if len(self.successful_vectors) < 5:
            return
        vectors = np.array(self.successful_vectors[-30:])
        mean_vec = np.mean(vectors, axis=0)
        centered = vectors - mean_vec
        self.cov_matrix = np.cov(centered.T) + np.eye(self.dim) * 1e-6
        self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if np.min(eigvals) < 1e-8:
            self.cov_matrix += np.eye(self.dim) * 1e-5
    
    def _adapt_mutation_weights(self, improved):
        n_improved = np.sum(improved)
        if n_improved < 1:
            return
        improvement_rate = n_improved / self.np
        decay = 0.95
        self.rand_weight = decay * self.rand_weight + (1 - decay) * np.random.uniform(0.15, 0.35)
        self.best_weight = decay * self.best_weight + (1 - decay) * np.random.uniform(0.25, 0.45)
        self.cov_weight = 1.0 - self.rand_weight - self.best_weight
        self.cov_weight = max(0.1, min(0.5, self.cov_weight))
    
    def _adapt_parameters(self, improved):
        n_imp = np.sum(improved)
        if n_imp > 0:
            self.f_history.append(self.F)
            if len(self.f_history) > 50:
                self.f_history.pop(0)
        if n_imp / self.np > 0.2:
            self.F = np.clip(self.F * 0.9, 0.3, 1.0)
        elif n_imp / self.np < 0.1:
            self.F = np.clip(self.F * 1.1, 0.3, 1.0)
        self.CR = np.clip(np.random.normal(0.5, 0.1), 0.3, 0.9)
        cr_per_dim = self._get_adaptive_cr()
        for j in range(self.dim):
            if len(self.cr_history) <= j:
                self.cr_history.append([])
            if np.sum(improved) > 0:
                dim_improved = np.sum(improved)
                if dim_improved > 0:
                    self.cr_history[j].append(cr_per_dim[j])
                    if len(self.cr_history[j]) > 50:
                        self.cr_history[j].pop(0)
    
    def _adapt_step_size(self):
        if len(self.f_history) < 5:
            return
        recent_f = np.mean(self.f_history[-10:])
        if recent_f < 0.5:
            self.step_size *= 0.98
        elif recent_f > 0.8:
            self.step_size *= 1.02
        self.step_size = np.clip(self.step_size, 0.01, 2.0)
    
    def _update_velocity_batch(self):
        momentum = 0.7
        self.velocity = momentum * self.velocity + self.step_size * (
            np.random.random((self.np, self.dim)) * (self.best_x - self.population)
        )
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        max_speed = (self.bounds[1] - self.bounds[0]) * 0.2
        scale = np.clip(speed / max_speed, 1.0, None)
        self.velocity /= scale
        self.population += self.velocity * 0.1
        self.population = self._clip_to_bounds_batch(self.population)
    
    def _update_best_batch(self, pop, fit):
        valid = ~np.isnan(fit)
        if not np.any(valid):
            return
        best_idx = np.argmin(fit[valid])
        all_indices = np.where(valid)[0]
        candidate_idx = all_indices[best_idx]
        if fit[candidate_idx] < self.best_f:
            self.best_f = fit[candidate_idx]
            self.best_x = pop[candidate_idx].copy()
    
    def _compute_diversity(self):
        if self.np < 2:
            return 0.0
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.std(distances) / (self.bounds[1] - self.bounds[0])
    
    def _restart_if_stagnant(self):
        improvement = self.prev_best - self.best_f
        if improvement < 1e-6 * max(1.0, abs(self.prev_best)):
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self.prev_best = self.best_f
        diversity = self._compute_diversity()
        return self.stagnation_count > 15 or (diversity < 0.01 and self.stagnation_count > 5)
```