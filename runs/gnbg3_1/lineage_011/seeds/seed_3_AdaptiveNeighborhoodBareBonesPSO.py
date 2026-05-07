import numpy as np


class AdaptiveNeighborhoodBareBonesPSO:
    """
    Bare-Bones Particle Swarm with Adaptive Topology.
    Combines Gaussian sampling position updates with dynamic neighborhood
    adaptation based on swarm diversity. Includes DE-style mutation for
    enhanced exploration on multi-modal landscapes.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(kwargs.get('np', 5 * dim), 300)
        self.lb = -100.0
        self.ub = 100.0
        self._rng = np.random.default_rng(kwargs.get('seed', None))
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize algorithm parameters."""
        self.w = 0.729
        self.cognitive = 1.494
        self.social = 1.494
        
        self.F = 0.5
        self.CR = 0.9
        
        self.min_diversity = 1e-6
        self.topology_threshold = 0.5
        
        self.max_stagnation = 15
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        
        self.iteration = 0
        self.gbest_fitness = np.inf
        self.gbest_pos = None
        
        self.operator_names = ['barebones', 'de_mutation', 'hybrid_sample']
        self.operator_rewards = {name: 0.0 for name in self.operator_names}
        self.operator_attempts = {name: 0 for name in self.operator_names}
    
    def __call__(self, func, stopping_condition):
        """Execute optimization."""
        self.func = func
        self._initialize_population()
        self._evaluate_initial()
        
        while not stopping_condition():
            operator_name = self._select_operator()
            
            trials = self._generate_trials_batch(operator_name)
            
            if len(trials) == 0:
                break
            
            trial_fitness = self.func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if stopping_condition():
                break
            
            self._select_survivors_batch(trials, trial_fitness)
            self._update_personal_best()
            
            if self._update_global_best():
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            self._adapt_parameters()
            self._adapt_topology()
            self._update_operator_rewards(operator_name)
            
            if self._check_stagnation():
                self._restart_if_needed()
            
            self.iteration += 1
        
        return self.gbest_fitness, self.gbest_pos.copy()
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            pop[:, d] = np.linspace(self.lb, self.ub, self.np)
            self._rng.shuffle(pop[:, d])
        
        pop += self._rng.uniform(-1, 1, (self.np, self.dim))
        self.population = np.clip(pop, self.lb, self.ub)
        
        self.velocity = np.zeros((self.np, self.dim))
        self.personal_best_pos = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self._build_initial_topology()
    
    def _build_initial_topology(self):
        """Build ring topology with neighbors."""
        self.neighbors = np.zeros((self.np, 2), dtype=int)
        for i in range(self.np):
            self.neighbors[i, 0] = (i - 1) % self.np
            self.neighbors[i, 1] = (i + 1) % self.np
    
    def _evaluate_initial(self):
        """Evaluate initial population."""
        fitness = self.func(self.population)
        self.personal_best_fitness = fitness.copy()
        
        best_idx = np.argmin(fitness)
        self.gbest_fitness = fitness[best_idx]
        self.gbest_pos = self.population[best_idx].copy()
    
    def _compute_diversity(self):
        """Compute population diversity as average distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _select_operator(self):
        """Select mutation operator based on success rates."""
        for name in self.operator_names:
            self.operator_attempts[name] += 1
        
        scores = np.array([
            self.operator_rewards[name] / max(1, self.operator_attempts[name])
            for name in self.operator_names
        ])
        scores = scores / scores.sum()
        chosen = self._rng.choice(self.operator_names, p=scores)
        
        return chosen
    
    def _generate_trials_batch(self, operator_name):
        """Generate trial population using selected operator."""
        if operator_name == 'barebones':
            trials = self._barebones_sample_batch()
        elif operator_name == 'de_mutation':
            trials = self._de_mutation_batch()
        else:
            trials = self._hybrid_sample_batch()
        
        trials = np.clip(trials, self.lb, self.ub)
        return trials
    
    def _barebones_sample_batch(self):
        """
        Bare-Bones PSO: sample from Gaussian distribution
        centered at midpoint of particle and its neighborhood best.
        """
        neighborhood_best = self._get_neighborhood_best_batch()
        
        mean = 0.5 * (self.population + neighborhood_best)
        sigma = np.abs(self.population - neighborhood_best)
        sigma = np.maximum(sigma, 1e-8)
        
        samples = mean + sigma * self._rng.standard_normal((self.np, self.dim))
        return samples
    
    def _get_neighborhood_best_batch(self):
        """Get best position from each particle's neighborhood."""
        neighborhood_best = np.zeros((self.np, self.dim))
        
        for i in range(self.np):
            nb_indices = self.neighbors[i]
            nb_fitness = self.personal_best_fitness[nb_indices]
            best_nb_idx = nb_indices[np.argmin(nb_fitness)]
            neighborhood_best[i] = self.personal_best_pos[best_nb_idx]
        
        return neighborhood_best
    
    def _de_mutation_batch(self):
        """Generate trials using DE/rand/1 mutation."""
        indices = self._rng.integers(0, self.np, size=(self.np, 3))
        
        r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]
        
        trials = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        
        cr_mask = self._rng.random((self.np, self.dim)) < self.CR
        enforce_mask = np.zeros((self.np, self.dim), dtype=bool)
        enforce_idx = self._rng.integers(0, self.dim, size=self.np)
        enforce_mask[np.arange(self.np), enforce_idx] = True
        
        trials = np.where(cr_mask | enforce_mask, trials, self.population)
        
        return trials
    
    def _hybrid_sample_batch(self):
        """Hybrid: Bare-Bones with velocity influence and perturbation."""
        neighborhood_best = self._get_neighborhood_best_batch()
        
        mean = (self.population + neighborhood_best + self.gbest_pos) / 3.0
        sigma = np.abs(self.population - neighborhood_best) + np.abs(self.velocity)
        sigma = np.maximum(sigma, 1e-8)
        
        trials = mean + sigma * self._rng.standard_normal((self.np, self.dim))
        
        perturb_mask = self._rng.random((self.np, self.dim)) < 0.1
        trials = trials + perturb_mask * self._rng.uniform(-10, 10, (self.np, self.dim))
        
        return trials
    
    def _select_survivors_batch(self, trials, trial_fitness):
        """Select survivors based on fitness comparison."""
        better_mask = trial_fitness < self.personal_best_fitness
        
        self.population[better_mask] = trials[better_mask]
        self.personal_best_fitness[better_mask] = trial_fitness[better_mask]
        
        self.velocity[better_mask] = trials[better_mask] - self.population[better_mask]
    
    def _update_personal_best(self):
        """Update personal best positions and velocities."""
        pass
    
    def _update_global_best(self):
        """Update global best if improvement found."""
        improved = self.personal_best_fitness < self.gbest_fitness
        
        if np.any(improved):
            best_idx = np.argmin(self.personal_best_fitness)
            self.gbest_fitness = self.personal_best_fitness[best_idx]
            self.gbest_pos = self.personal_best_pos[best_idx].copy()
            return True
        
        return False
    
    def _adapt_parameters(self):
        """Adapt PSO parameters based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.min_diversity:
            self.w = min(0.9, self.w * 1.1)
            self.F = min(1.0, self.F * 1.2)
        else:
            self.w = max(0.3, self.w * 0.95)
            self.F = max(0.3, self.F * 0.95)
    
    def _adapt_topology(self):
        """Adapt neighborhood topology based on diversity."""
        diversity = self._compute_diversity()
        max_diversity_estimate = (self.ub - self.lb) * np.sqrt(self.dim / 2)
        normalized_diversity = diversity / max_diversity_estimate
        
        if normalized_diversity < self.topology_threshold:
            self._expand_topology()
        elif normalized_diversity > 0.8:
            self._contract_topology()
    
    def _expand_topology(self):
        """Expand neighborhood to include more particles."""
        new_size = min(5, self.np // 4)
        new_neighbors = np.zeros((self.np, new_size), dtype=int)
        
        for i in range(self.np):
            indices = [(i + offset) % self.np for offset in range(-new_size // 2, new_size // 2 + 1)]
            new_neighbors[i] = indices
        
        self.neighbors = new_neighbors
    
    def _contract_topology(self):
        """Contract neighborhood to local neighborhood only."""
        self.neighbors = np.zeros((self.np, 2), dtype=int)
        for i in range(self.np):
            self.neighbors[i, 0] = (i - 1) % self.np
            self.neighbors[i, 1] = (i + 1) % self.np
    
    def _update_operator_rewards(self, operator_name):
        """Update operator rewards based on recent performance."""
        improvement_count = np.sum(self.personal_best_fitness < np.inf)
        improvement_rate = improvement_count / self.np
        
        self.operator_rewards[operator_name] += improvement_rate
    
    def _check_stagnation(self):
        """Check if optimization has stagnated."""
        if self.gbest_fitness >= self.prev_best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.prev_best_fitness = self.gbest_fitness
        return self.stagnation_counter >= self.max_stagnation
    
    def _restart_if_needed(self):
        """Reinitialize part of population if stagnant."""
        n_restart = max(2, self.np // 4)
        restart_idx = self._rng.choice(self.np, n_restart, replace=False)
        
        for d in range(self.dim):
            restart_vals = np.linspace(self.lb, self.ub, n_restart)
            self._rng.shuffle(restart_vals)
            self.population[restart_idx, d] = restart_vals
        
        self.population[restart_idx] = np.clip(
            self.population[restart_idx] + self._rng.uniform(-5, 5, (n_restart, self.dim)),
            self.lb, self.ub
        )
        
        self.velocity[restart_idx] = 0.0
        self.stagnation_counter = 0
