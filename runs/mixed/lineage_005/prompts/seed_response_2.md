```python
import numpy as np


class CovarianceArchiveSwarmOptimizer:
    """
    A population-based metaheuristic combining:
    - Simplified CMA-ES covariance adaptation for exploration
    - Archive-guided mutations for exploitation
    - Velocity-based particle dynamics
    - Adaptive local search for intensification
    - Dynamic parameter self-adaptation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = kwargs.get('np', min(8 * dim, 300))
        self.ub = kwargs.get('ub', 100.0)
        self.lb = kwargs.get('lb', -100.0)
        self._init_algorithm_state()
    
    def _init_algorithm_state(self):
        self.population = None
        self.fitness = None
        self.trial_population = None
        self.trial_fitness = None
        self.cov = None
        self.step_size = None
        self.p_sigma = None
        self.archive = None
        self.archive_fitness = None
        self.best_fitness = np.inf
        self.best_position = None
        self.gen = 0
        self.succ_rate = 0.0
        self.adapt_counter = 0
        self.stagnation_count = 0
        
    def _initialize_population(self):
        range_val = self.ub - self.lb
        self.population = np.random.uniform(0, 1, (self.np, self.dim)) * range_val + self.lb
        self.population = self._clip_to_bounds_batch(self.population)
        self.fitness = np.full(self.np, np.inf)
        self._init_covariance_matrix()
        self.archive = self.population.copy()
        self.archive_fitness = self.fitness.copy()
    
    def _init_covariance_matrix(self):
        self.step_size = (self.ub - self.lb) * 0.05
        self.cov = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
    
    def _eval_wrapper(self, pop_batch):
        clipped = self._clip_to_bounds_batch(pop_batch)
        return self.func(clipped)
    
    def _update_covariance_batch(self, success_mask):
        n_success = np.sum(success_mask)
        expected_succ = 0.5 * self.np
        if n_success > 0:
            successful = self.population[success_mask]
            weights = 1.0 / (n_success + 1e-10)
            mean_shift = np.sum(successful - self.trial_population[success_mask], axis=0)
            self.p_sigma = (1 - 0.03) * self.p_sigma + np.sqrt(0.03 * (2 - 0.03)) * weights * mean_shift / (self.step_size + 1e-10)
        c1 = 1.0 / (12.0 + 0.03 * self.dim)
        c_mu = min(2.0 * n_success / (self.dim + n_success + 1e-10), 1.0 - c1)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(min(n_success, self.np // 4)):
            diff = (self.trial_population[i] - self.population[i]).reshape(-1, 1)
            rank_mu += diff @ diff.T
        if n_success > 0:
            rank_mu *= 2.0 * c_mu / (n_success * self.step_size**2 + 1e-10)
        self.cov = (1 - c1 - c_mu) * self.cov + c1 * (self.p_sigma.reshape(-1, 1) @ self.p_sigma.reshape(1, -1)) + rank_mu
        min_eig = np.linalg.eigvalsh(self.cov).min()
        if min_eig < 1e-10:
            self.cov += (1e-9 - min_eig) * np.eye(self.dim)
    
    def _update_step_size_batch(self, success_mask):
        n_success = np.sum(success_mask)
        self.succ_rate = 0.9 * self.succ_rate + 0.1 * (n_success / self.np)
        target_rate = 0.5
        if self.succ_rate > target_rate:
            self.step_size *= np.exp(0.1 * (self.succ_rate - target_rate) / (1 - target_rate + 1e-10))
        else:
            self.step_size *= np.exp(-0.1 * (target_rate - self.succ_rate) / (target_rate + 1e-10))
        self.step_size = np.clip(self.step_size, (self.ub - self.lb) * 1e-5, (self.ub - self.lb) * 0.5)
    
    def _sample_from_distribution(self, mean_vec):
        noise = np.random.randn(self.np, self.dim)
        try:
            L = np.linalg.cholesky(self.cov)
            samples = mean_vec + self.step_size * (noise @ L.T)
        except np.linalg.LinAlgError:
            samples = mean_vec + self.step_size * noise
        return samples
    
    def _mutate_current_to_pbest_with_culture(self, population, fitness, archive, archive_fitness):
        n_select = min(3, len(archive))
        if n_select == 0:
            return self._mutate_variant09(population, fitness)
        pbest_idx = np.argmin(fitness)
        pbest = population[pbest_idx]
        top_archive = archive[np.argsort(archive_fitness)[:n_select]]
        cultural = np.mean(top_archive, axis=0)
        F1, F2 = np.random.uniform(0.3, 0.9, 2)
        CR = np.random.uniform(0.5, 0.9)
        r1, r2, r3 = np.random.randint(0, self.np, 3)
        mutant = population + F1 * (pbest - population) + F2 * (cultural - population[r3])
        mask = np.random.rand(self.np, self.dim) < CR
        trials = np.where(mask, mutant, population)
        return self._clip_to_bounds_batch(trials)
    
    def _mutate_variant09(self, population, fitness):
        sorted_idx = np.argsort(fitness)
        top third = population[sorted_idx[:self.np // 3]]
        centroid = np.mean(top_third, axis=0)
        F = np.random.uniform(0.4, 0.8)
        mutant = population + F * (centroid - population)
        return self._clip_to_bounds_batch(mutant)
    
    def _build_trial_batch(self):
        if self.gen < 5:
            mean_estimate = np.mean(self.population, axis=0)
            trials = self._sample_from_distribution(mean_estimate)
        else:
            trials = self._mutate_current_to_pbest_with_culture(self.population, self.fitness, self.archive, self.archive_fitness)
        return self._clip_to_bounds_batch(trials)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        if len(trial_fitness) < len(trials):
            valid_len = len(trial_fitness)
            trials = trials[:valid_len]
            trial_fitness = trial_fitness[:valid_len]
            population = population[:valid_len]
            fitness = fitness[:valid_len]
        improved = trial_fitness < fitness
        new_pop = np.where(improved[:, np.newaxis], trials, population)
        new_fit = np.where(improved, trial_fitness, fitness)
        return new_pop, new_fit, improved
    
    def _update_archive_batch(self, population, fitness):
        combined_pop = np.vstack([self.archive, population])
        combined_fit = np.concatenate([self.archive_fitness, fitness])
        top_k = min(self.np * 2, len(combined_fit))
        top_idx = np.argsort(combined_fit)[:top_k]
        self.archive = combined_pop[top_idx]
        self.archive_fitness = combined_fit[top_idx]
    
    def _compute_reward(self, trial_fitness, fitness):
        if len(trial_fitness) == 0:
            return 0.0
        improvement = np.mean(fitness[:len(trial_fitness)] - trial_fitness)
        return np.clip(improvement / (np.std(trial_fitness) + 1e-10), -5, 5)
    
    def _adapt_parameters_batch(self, reward):
        self.adapt_counter += 1
        if self.adapt_counter >= 5:
            self.adapt_counter = 0
            adaptation_strength = 0.1 * np.tanh(reward)
            if hasattr(self, '_current_F'):
                self._current_F = np.clip(self._current_F + adaptation_strength * 0.2, 0.1, 1.5)
            else:
                self._current_F = np.random.uniform(0.5, 0.9)
    
    def _apply_adaptive_local_search(self, population, fitness):
        if self.gen % 10 != 0 or len(population) < 10:
            return population, fitness
        top_k = min(self.np // 5, len(population))
        top_idx = np.argsort(fitness)[:top_k]
        ls_radius = self.step_size * 0.5
        ls_batch = population[top_idx].copy()
        for i in range(len(ls_batch)):
            perturbation = np.random.randn(self.dim) * ls_radius
            candidate = ls_batch[i] + perturbation
            candidate = self._clip_to_bounds_batch(candidate.reshape(1, -1))[0]
            ls_batch[i] = candidate
        ls_fitness = self._eval_wrapper(ls_batch)
        improved_mask = ls_fitness < fitness[top_idx]
        new_pop = population.copy()
        new_fit = fitness.copy()
        new_pop[top_idx[improved_mask]] = ls_batch[improved_mask]
        new_fit[top_idx[improved_mask]] = ls_fitness[improved_mask]
        return new_pop, new_fit
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_if_stagnant(self, fitness):
        current_best = np.min(fitness)
        diversity = self._compute_diversity(self.population)
        threshold = (self.ub - self.lb) * 0.001
        if self.best_fitness < np.inf and (self.best_fitness - current_best) < threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        should_restart = self.stagnation_count > 50 or diversity < threshold * 0.1
        if should_restart:
            self._initialize_population()
            self.stagnation_count = 0
        return should_restart
    
    def _clip_to_bounds_batch(self, pop_batch):
        return np.clip(pop_batch, self.lb, self.ub)
    
    def _track_best(self, fitness):
        gen_best_idx = np.argmin(fitness)
        gen_best_fit = fitness[gen_best_idx]
        if gen_best_fit < self.best_fitness:
            self.best_fitness = gen_best_fit
            self.best_position = self.population[gen_best_idx].copy()
            self.stagnation_count = 0
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        self.best_fitness = np.inf
        self.best_position = None
        self.gen = 0
        
        while not stopping_condition():
            self.trial_population = self._build_trial_batch()
            self.trial_fitness = self._eval_wrapper(self.trial_population)
            
            if len(self.trial_fitness) < len(self.trial_population):
                break
            
            self.population, self.fitness, success_mask = self._select_survivors_batch(
                self.population, self.fitness, self.trial_population, self.trial_fitness
            )
            
            if len(self.fitness) == 0:
                break
            
            if stopping_condition():
                break
            
            self._update_covariance_batch(success_mask)
            self._update_step_size_batch(success_mask)
            self._update_archive_batch(self.population, self.fitness)
            self._track_best(self.fitness)
            
            self.population, self.fitness = self._apply_adaptive_local_search(self.population, self.fitness)
            
            reward = self._compute_reward(self.trial_fitness, self.fitness)
            self._adapt_parameters_batch(reward)
            
            self._restart_if_stagnant(self.fitness)
            self.gen += 1
            
            if self.gen > 5000:
                break
        
        return self.best_fitness, self.best_position
```