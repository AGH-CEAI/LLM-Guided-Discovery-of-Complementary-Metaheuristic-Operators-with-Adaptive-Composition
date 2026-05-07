import numpy as np

class CovarianceGuidedParticleSwarm:
    """Particle Swarm Optimizer with adaptive covariance and ring topology."""
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(kwargs.get('np', 5 * dim), 300)
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        
        self.step_size = kwargs.get('step_size', 0.5)
        self.cognitive = 1.496
        self.social = 1.496
        self.max_velocity = 0.2 * (self.bounds[0, 1] - self.bounds[0, 0])
        
        self.covariance = np.eye(dim) * (self.step_size ** 2)
        self.covariance_adapt_rate = 0.01
        
        self._topology_radius = 3
        self._min_diversity = 1e-6
        self._stagnation_counter = 0
        self._stagnation_threshold = 50
        
        self._success_history = []
        self._success_window = 5
        self._target_success_rate = 0.25
        
        self._trial_buffer = []
        self._trial_buffer_size = 10
        
        self._gbest = None
        self._gbest_fitness = np.inf
        self._velocity = None
        
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = self._evaluate_batch(population, func, None)
        self._update_global_best(population, fitness)
        self._velocity = np.zeros((self.np, self.dim))
        
        while not stopping_condition():
            topology = self._adapt_topology_radius(population)
            neighborhood_best_idx = self._select_neighborhood_best_batch(topology)
            
            trials = self._sample_trials_batch(population, neighborhood_best_idx)
            trial_fit = self._evaluate_batch(trials, func, trials)
            
            if len(trial_fit) < len(trials) or stopping_condition():
                break
                
            self._update_covariance_adaptive(population, trials, trial_fit)
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fit)
            self._velocity = self._update_velocity_batch(population, neighborhood_best_idx)
            self._update_global_best(population, fitness)
            
            if self._check_stagnation(trial_fit, fitness):
                population = self._restart_population(population)
                fitness = self._evaluate_batch(population, func, None)
                
            self._adapt_step_size()
            
        return self._gbest_fitness, self._gbest.copy()
    
    def _initialize_population(self):
        population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.np, self.dim)
        )
        return np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
    
    def _evaluate_batch(self, candidates, func, expected_batch):
        clipped = np.clip(candidates, self.bounds[:, 0], self.bounds[:, 1])
        fitness = func(clipped)
        
        if len(fitness) < len(clipped):
            return fitness[:len(expected_batch)] if expected_batch is not None else fitness
            
        return fitness
    
    def _update_global_best(self, population, fitness):
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < self._gbest_fitness:
            self._gbest = population[current_best_idx].copy()
            self._gbest_fitness = fitness[current_best_idx]
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1
    
    def _restart_population(self, population):
        new_pop = self._initialize_population()
        if self._gbest is not None:
            new_pop[0] = self._gbest.copy()
        self._velocity = np.zeros((self.np, self.dim))
        self._stagnation_counter = 0
        self._trial_buffer.clear()
        return new_pop
    
    def _adapt_topology_radius(self, population):
        diversity = self._compute_diversity(population)
        if diversity < self._min_diversity:
            self._topology_radius = min(self._topology_radius + 1, self.np // 2)
        else:
            shrink_factor = 0.98
            self._topology_radius = max(2, int(self._topology_radius * shrink_factor))
        return self._build_ring_topology()
    
    def _build_ring_topology(self):
        n = self.np
        topology = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(-self._topology_radius, self._topology_radius + 1):
                if j != 0:
                    topology[i, (i + j) % n] = True
        return topology
    
    def _select_neighborhood_best_batch(self, topology, pbest_fitness):
        neighborhood_best = np.zeros(self.np, dtype=int)
        for i in range(self.np):
            neighbors = np.where(topology[i])[0]
            if len(neighbors) > 0:
                neighborhood_best[i] = neighbors[np.argmin(pbest_fitness[neighbors])]
            else:
                neighborhood_best[i] = i
        return neighborhood_best
    
    def _compute_diversity(self, population):
        dists = []
        for i in range(self.np):
            diffs = population - population[i]
            dists.extend(np.sqrt(np.sum(diffs ** 2, axis=1)))
        dists = np.array(dists)
        return np.std(dists) / (np.mean(dists) + 1e-10)
    
    def _adapt_step_size(self):
        if len(self._success_history) >= 3:
            success_rate = np.mean(self._success_history[-self._success_window:])
            if success_rate > self._target_success_rate:
                self.step_size *= 1.05
            elif success_rate < self._target_success_rate * 0.5:
                self.step_size *= 0.95
            self.step_size = np.clip(self.step_size, 0.01, 10.0)
    
    def _update_covariance_adaptive(self, population, trials, trial_fit):
        perturbations = trials - population
        best_mask = trial_fit < np.mean(trial_fit)
        
        if np.sum(best_mask) >= 2:
            best_perturbations = perturbations[best_mask]
            mean_perturbation = np.mean(best_perturbations, axis=0)
            cov = np.cov(best_perturbations.T)
            
            if not np.any(np.isnan(cov)) and not np.any(np.isinf(cov)):
                cov = np.clip(cov, -1e6, 1e6)
                self.covariance = ((1 - self.covariance_adapt_rate) * self.covariance + 
                                  self.covariance_adapt_rate * cov + 
                                  1e-8 * np.eye(self.dim))
    
    def _sample_trials_batch(self, population, neighborhood_best_idx):
        lbest_positions = population[neighborhood_best_idx]
        lbest_positions = np.clip(lbest_positions, self.bounds[:, 0], self.bounds[:, 1])
        
        try:
            chol = np.linalg.cholesky(self.covariance)
            noise = np.random.randn(self.np, self.dim) @ chol.T
        except np.linalg.LinAlgError:
            noise = np.random.randn(self.np, self.dim) * self.step_size
        
        perturbations = self.step_size * noise
        trials = population + perturbations
        
        self._trial_buffer.append(trials.copy())
        if len(self._trial_buffer) > self._trial_buffer_size:
            self._trial_buffer.pop(0)
            
        return np.clip(trials, self.bounds[:, 0], self.bounds[:, 1])
    
    def _update_velocity_batch(self, population, neighborhood_best_idx):
        cognitive_component = self.cognitive * np.random.rand(self.np, self.dim) * (
            self._gbest - population
        ) if self._gbest is not None else 0
        
        social_component = self.social * np.random.rand(self.np, self.dim) * (
            population[neighborhood_best_idx] - population
        )
        
        velocity = 0.7 * self._velocity + cognitive_component + social_component
        return np.clip(velocity, -self.max_velocity, self.max_velocity)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fit):
        improvement_mask = trial_fit < fitness
        num_improved = np.sum(improvement_mask)
        
        self._success_history.append(num_improved / self.np)
        if len(self._success_history) > self._success_window:
            self._success_history.pop(0)
            
        new_population = population.copy()
        new_population[improvement_mask] = trials[improvement_mask]
        new_fitness = fitness.copy()
        new_fitness[improvement_mask] = trial_fit[improvement_mask]
        
        return new_population, new_fitness
