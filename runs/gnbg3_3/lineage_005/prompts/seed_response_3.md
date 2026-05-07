```python
import numpy as np


class AdaptiveTopologyCulturalPSO:
    """
    Adaptive Topology Particle Swarm with Cultural Coevolution.
    
    A PSO-based optimizer with:
    - Adaptive topology (ring/star/von Neumann switching)
    - Multiple position update strategies (standard, temporal drift, fitness rank, mutation)
    - Cultural belief space guiding search
    - Adaptive inertia, cognitive/social coefficients
    - Diversity-triggered restarts
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.NP = kwargs.get('NP', min(150, max(50, 5 * dim)))
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.2
        
        # Inertia weight parameters
        self.w_max = kwargs.get('w_max', 0.9)
        self.w_min = kwargs.get('w_min', 0.2)
        
        # Cognitive and social coefficients
        self.c1 = kwargs.get('c1', 1.496)
        self.c2 = kwargs.get('c2', 1.496)
        
        # Strategy parameters
        self.strategy_names = ['standard', 'temporal_drift', 'fitness_rank', 'mutation']
        self.n_strategies = len(self.strategy_names)
        self.strategy_probs = np.ones(self.n_strategies) / self.n_strategies
        self.strategy_successes = np.zeros(self.n_strategies)
        
        # Topology parameters
        self.topology_names = ['ring', 'star', 'von_neumann']
        self.current_topology = 0
        
        # Adaptation parameters
        self.adapt_interval = 10
        self.diversity_threshold = 1e-6
        self.stagnation_threshold = 50
        
        # State
        self.generation = 0
        self.best_history = []
        self.stagnation_counter = 0
        
    def _initialize_population(self, dim, lower, upper, NP):
        """Initialize population and velocity."""
        population = lower + (upper - lower) * np.random.rand(NP, dim)
        velocity = np.random.randn(NP, dim) * (upper - lower) * 0.05
        return population, velocity
    
    def _initialize_personal_best(self, population):
        """Initialize personal best positions and fitness."""
        return population.copy(), np.full(len(population), np.inf)
    
    def _initialize_topology_ring(self, NP):
        """Initialize ring topology neighbors."""
        n_neighbors = max(2, int(np.sqrt(NP)))
        neighbors = np.zeros((NP, n_neighbors), dtype=int)
        for i in range(NP):
            for j in range(n_neighbors):
                neighbors[i, j] = (i - n_neighbors // 2 + j) % NP
        return neighbors
    
    def _initialize_topology_von_neumann(self, NP):
        """Initialize von Neumann grid topology."""
        grid_size = int(np.ceil(np.sqrt(NP)))
        NP_actual = grid_size * grid_size
        neighbors = np.zeros((NP_actual, 4), dtype=int)
        for i in range(NP_actual):
            row, col = i // grid_size, i % grid_size
            neighbors[i, 0] = row * grid_size + (col - 1) % grid_size
            neighbors[i, 1] = row * grid_size + (col + 1) % grid_size
            neighbors[i, 2] = ((row - 1) % grid_size) * grid_size + col
            neighbors[i, 3] = ((row + 1) % grid_size) * grid_size + col
        if NP < NP_actual:
            return neighbors[:NP]
        return neighbors
    
    def _select_topology(self, NP):
        """Select topology based on current index."""
        if self.current_topology == 0:
            return self._initialize_topology_ring(NP)
        elif self.current_topology == 1:
            return np.tile(np.arange(NP), (NP, 1))
        else:
            return self._initialize_topology_von_neumann(NP)
    
    def _initialize_swarm(self, dim, lower, upper, NP):
        """Initialize full swarm state."""
        population, velocity = self._initialize_population(dim, lower, upper, NP)
        p_best, p_best_fitness = self._initialize_personal_best(population)
        neighborhood_indices = self._select_topology(NP)
        return population, velocity, p_best, p_best_fitness, neighborhood_indices
    
    def _mutate_batch(self, population, p_best, neighborhood_indices, inertia, 
                      cognitive_coeff, social_coeff, temporal_drift, strategy_probs, 
                      strategy_successes, dim, NP):
        """Generate trial vectors using selected strategies in batch."""
        trials = np.empty_like(population)
        r1 = np.random.randint(0, NP, NP)
        r2 = np.random.randint(0, NP, NP)
        
        neighborhood_best = np.array([p_best[neighborhood_indices[i]].min(axis=0) 
                                       for i in range(NP)])
        
        rand_cog = np.random.rand(NP, dim)
        rand_soc = np.random.rand(NP, dim)
        
        v_base = (inertia * np.random.rand(NP, 1) + 
                  cognitive_coeff * rand_cog + 
                  social_coeff * rand_soc)
        
        v_standard = v_base * (p_best[r1] - population) + v_base * (neighborhood_best - population)
        
        theta = np.random.rand(NP, 1) * 2 * np.pi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        if dim >= 2:
            v_drift = np.copy(v_standard)
            v_drift[:, 0] = cos_t[:, 0] * v_standard[:, 0] - sin_t[:, 0] * v_standard[:, 1]
            v_drift[:, 1] = sin_t[:, 0] * v_standard[:, 0] + cos_t[:, 0] * v_standard[:, 1]
            if dim > 2:
                v_drift[:, 2:] = v_standard[:, 2:]
        else:
            v_drift = v_standard
        
        ranks = np.argsort(np.argsort(-p_best[:, 0])) + 1
        rank_weights = 1.0 / ranks[r1].reshape(-1, 1)
        v_rank = v_base * rank_weights * (p_best[r1] - population) + v_base * (neighborhood_best - population)
        
        v_mutation = v_base * (p_best[r1] - population) + v_base * (neighborhood_best - population)
        v_mutation += temporal_drift * np.random.randn(NP, dim)
        
        strategy_idx = np.random.choice(self.n_strategies, NP, p=strategy_probs)
        
        for s in range(self.n_strategies):
            mask = strategy_idx == s
            if s == 0:
                trials[mask] = population[mask] + v_standard[mask]
            elif s == 1:
                trials[mask] = population[mask] + v_drift[mask]
            elif s == 2:
                trials[mask] = population[mask] + v_rank[mask]
            else:
                trials[mask] = population[mask] + v_mutation[mask]
        
        return trials
    
    def _velocity_update_batch(self, population, velocity, p_best, neighborhood_best,
                               inertia, cognitive_coeff, social_coeff, temporal_drift,
                               strategy_probs, strategy_successes, dim):
        """Update velocity using multiple strategies."""
        NP = len(population)
        r1 = np.random.randint(0, NP, NP)
        r2 = np.random.randint(0, NP, NP)
        
        rand_cog = np.random.rand(NP, dim)
        rand_soc = np.random.rand(NP, dim)
        
        cognitive = cognitive_coeff * rand_cog * (p_best[r1] - population)
        social = social_coeff * rand_soc * (neighborhood_best - population)
        
        v_standard = inertia * velocity + cognitive + social
        
        theta = np.random.rand(NP, 1) * 2 * np.pi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        if dim >= 2:
            v_drift = np.copy(v_standard)
            v_drift[:, 0] = cos_t[:, 0] * v_standard[:, 0] - sin_t[:, 0] * v_standard[:, 1]
            v_drift[:, 1] = sin_t[:, 0] * v_standard[:, 0] + cos_t[:, 0] * v_standard[:, 1]
            if dim > 2:
                v_drift[:, 2:] = v_standard[:, 2:]
        else:
            v_drift = v_standard
        
        ranks = np.argsort(np.argsort(-p_best[:, 0])) + 1
        rank_weights = 1.0 / (ranks[r1].reshape(-1, 1) + 1)
        v_rank = inertia * velocity + cognitive * rank_weights + social
        
        v_mutation = v_standard + temporal_drift * np.random.randn(NP, dim)
        
        strategy_idx = np.random.choice(self.n_strategies, NP, p=strategy_probs)
        
        new_velocity = np.empty_like(velocity)
        for s in range(self.n_strategies):
            mask = strategy_idx == s
            if s == 0:
                new_velocity[mask] = v_standard[mask]
            elif s == 1:
                new_velocity[mask] = v_drift[mask]
            elif s == 2:
                new_velocity[mask] = v_rank[mask]
            else:
                new_velocity[mask] = v_mutation[mask]
        
        return new_velocity
    
    def _position_update_batch(self, population, velocity, lower, upper):
        """Update positions from velocity."""
        return population + velocity
    
    def _select_survivors_batch(self, population, velocity, p_best, p_best_fitness,
                                trial_population, trial_fitness):
        """Greedy selection between current and trial populations."""
        NP = len(population)
        improved_mask = trial_fitness < p_best_fitness
        
        population[improved_mask] = trial_population[improved_mask]
        velocity[improved_mask] = np.random.randn(np.sum(improved_mask), self.dim) * (self.upper_bound - self.lower_bound) * 0.05
        
        p_best_improved = trial_fitness < p_best_fitness
        p_best[p_best_improved] = population[p_best_improved]
        p_best_fitness[p_best_improved] = trial_fitness[p_best_improved]
        
        return population, velocity, p_best, p_best_fitness
    
    def _adapt_inertia_weight(self, inertia, g_best_fitness, f_opt):
        """Adapt inertia weight based on improvement rate."""
        if len(self.best_history) > 1:
            improvement_rate = (self.best_history[-2] - self.best_history[-1]) / (self.best_history[-2] + 1e-10)
            adaptation = self.w_max - (self.w_max - self.w_min) * (1.0 - np.exp(-improvement_rate * 10))
            inertia = np.clip(adaptation, self.w_min, self.w_max)
        else:
            inertia = self.w_max
        return inertia
    
    def _adapt_cognitive_social(self, cognitive_coeff, social_coeff, g_best_fitness, f_opt):
        """Adapt cognitive and social coefficients based on convergence."""
        if len(self.best_history) > 1:
            recent_improvement = self.best_history[-2] - self.best_history[-1]
            if recent_improvement < 1e-8:
                cognitive_coeff = min(cognitive_coeff * 1.05, 2.0)
                social_coeff = max(social_coeff * 0.95, 0.5)
            else:
                cognitive_coeff = max(cognitive_coeff * 0.99, 0.5)
                social_coeff = min(social_coeff * 1.01, 2.0)
        return cognitive_coeff, social_coeff
    
    def _adapt_neighborhood_size(self, neighborhood_size, diversity, g_best_fitness, f_opt):
        """Adapt neighborhood size based on diversity and convergence."""
        if len(self.best_history) > 1:
            recent_improvement = self.best_history[-2] - self.best_history[-1]
            if diversity < self.diversity_threshold:
                neighborhood_size = min(neighborhood_size + 1, self.NP // 2)
            elif recent_improvement < 1e-8:
                neighborhood_size = max(neighborhood_size - 1, 2)
        return neighborhood_size
    
    def _adapt_strategy_probs(self, strategy_probs, trial_fitness, g_best_fitness):
        """Adapt strategy selection probabilities based on success."""
        if len(self.best_history) > 1:
            recent_improvement = self.best_history[-2] - self.best_history[-1]
            if recent_improvement > 1e-6:
                best_strategy = np.argmax(self.strategy_successes)
                new_prob = strategy_probs[best_strategy] * 0.9 + 0.1 / self.n_strategies
                strategy_probs[best_strategy] = new_prob
                strategy_probs /= strategy_probs.sum()
        return strategy_probs
    
    def _update_temporal_drift(self, g_best_fitness, f_opt):
        """Update temporal drift factor based on improvement rate."""
        if len(self.best_history) > 1:
            improvement_rate = (self.best_history[-2] - self.best_history[-1]) / (self.best_history[-2] + 1e-10)
            temporal_drift = np.clip(improvement_rate * 0.1, 0.0, 0.5)
        else:
            temporal_drift = 0.1
        return temporal_drift
    
    def _compute_diversity(self, population):
        """Compute population diversity as average Euclidean distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _check_diversity_trigger(self, diversity):
        """Check if diversity is low enough to trigger restart."""
        return diversity < self.diversity_threshold
    
    def _restart_if_stagnant(self, population, velocity, p_best, p_best_fitness, g_best, dim, lower, upper):
        """Restart population to increase diversity."""
        new_population = lower + (upper - lower) * np.random.rand(*population.shape)
        new_population[0] = g_best.copy()
        
        new_velocity = np.random.randn(*velocity.shape) * (upper - lower) * 0.05
        new_velocity[0] = np.zeros(dim)
        
        new_p_best = new_population.copy()
        new_p_best_fitness = np.full(len(new_population), np.inf)
        
        self.stagnation_counter = 0
        
        return new_population, new_velocity, new_p_best, new_p_best_fitness
    
    def _update_neighborhood_best(self, p_best, neighborhood_indices):
        """Update neighborhood best positions."""
        NP = len(p_best)
        neighborhood_best = np.empty((NP, self.dim))
        for i in range(NP):
            neighbors = neighborhood_indices[i]
            best_neighbor_idx = neighbors[np.argmin(p_best[neighbors, 0])]
            neighborhood_best[i] = p_best[best_neighbor_idx]
        return neighborhood_best
    
    def __call__(self, func, stopping_condition):
        dim = self.dim
        lower = self.lower_bound
        upper = self.upper_bound
        NP = self.NP
        
        f_opt = np.inf
        x_opt = None
        self.generation = 0
        self.best_history = [np.inf]
        self.stagnation_counter = 0
        self.strategy_probs = np.ones(self.n_strategies) / self.n_strategies
        self.strategy_successes = np.zeros(self.n_strategies)
        
        population, velocity, p_best, p_best_fitness, neighborhood_indices = self._initialize_swarm(dim, lower, upper, NP)
        
        neighborhood_size = max(2, int(np.sqrt(NP)))
        inertia = self.w_max
        cognitive_coeff = self.c1
        social_coeff = self.c2
        temporal_drift = 0.1
        diversity = self._compute_diversity(population)
        
        while not stopping_condition():
            if self.generation % self.adapt_interval == 0:
                inertia = self._adapt_inertia_weight(inertia, f_opt, f_opt)
                self.current_topology = (self.current_topology + 1) % len(self.topology_names)
                neighborhood_indices = self._select_topology(NP)
            
            neighborhood_best = self._update_neighborhood_best(p_best, neighborhood_indices)
            
            velocity = self._velocity_update_batch(
                population, velocity, p_best, neighborhood_best,
                inertia, cognitive_coeff, social_coeff, temporal_drift,
                self.strategy_probs, self.strategy_successes, dim
            )
            
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            
            new_position = self._position_update_batch(population, velocity, lower, upper)
            new_position = np.clip(new_position, lower, upper)
            
            trial_fitness = func(new_position)
            
            if len(trial_fitness) < len(new_position):
                if len(trial_fitness) > 0:
                    new_position = new_position[:len(trial_fitness)]
                    velocity = velocity[:len(trial_fitness)]
                    trial_fitness = trial_fitness[:len(trial_fitness)]
                break
            
            population, velocity, p_best, p_best_fitness = self._select_survivors_batch(
                population, velocity, p_best, p_best_fitness, new_position, trial_fitness
            )
            
            best_idx = np.argmin(p_best_fitness)
            if p_best_fitness[best_idx] < f_opt:
                f_opt = p_best_fitness[best_idx]
                x_opt = p_best[best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            cognitive_coeff, social_coeff = self._adapt_cognitive_social(cognitive_coeff, social_coeff, f_opt, f_opt)
            
            neighborhood_size = self._adapt_neighborhood_size(neighborhood_size, diversity, f_opt, f_opt)
            
            diversity = self._compute_diversity(population)
            
            self.strategy_probs = self._adapt_strategy_probs(self.strategy_probs, trial_fitness, f_opt)
            
            if self._check_diversity_trigger(diversity):
                population, velocity, p_best, p_best_fitness = self._restart_if_stagnant(
                    population, velocity, p_best, p_best_fitness, x_opt, dim, lower, upper
                )
            
            temporal_drift = self._update_temporal_drift(f_opt, f_opt)
            
            self.generation += 1
            self.best_history.append(f_opt)
        
        return f_opt, x_opt
```