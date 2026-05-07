```python
import numpy as np


class QuantumSwarmOppositionRingOptimizer:
    """
    Quantum-inspired Swarm Optimizer with Adaptive Opposition-based Learning
    and Ring Topology Neighborhoods.
    
    Key features:
    - Ring topology with adaptive neighborhood size
    - Quantum rotation angles for exploration
    - Opposition-based learning for exploitation
    - Inertia weight adaptation
    - Fitness rank transformation for position updates
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.np = min(max(4 * dim, 60), 300)

        self.inertia_weight = 0.729
        self.cognitive_param = 1.494
        self.social_param = 1.494
        self.quantum_rotation_base = 0.1
        self.opposition_rate = 0.3
        self.diversity_threshold = 0.05

        self.population = None
        self.velocity = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.neighborhood_size = 3
        self.stagnation_count = 0
        self.gen = 0
        self.best_history = []

    def _initialize_population(self):
        """Initialize population uniformly within bounds."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )

    def _initialize_velocity(self):
        """Initialize velocity vectors scaled by search range."""
        range_val = (self.upper_bound - self.lower_bound) * 0.1
        self.velocity = np.random.uniform(
            -range_val, range_val, (self.np, self.dim)
        )

    def _bound_candidates(self, candidates):
        """Clip candidates to search bounds."""
        return np.clip(candidates, self.lower_bound, self.upper_bound)

    def _evaluate_batch(self, func, candidates):
        """Evaluate fitness for a batch of candidates."""
        bounded = self._bound_candidates(candidates)
        fitness = func(bounded)
        if len(fitness) < len(candidates):
            valid_len = len(fitness)
            fitness = np.pad(fitness, (0, len(candidates) - valid_len),
                           constant_values=np.inf)
        return fitness

    def _initialize_best_tracking(self):
        """Initialize personal and global best tracking."""
        self.personal_best = self.population.copy()
        self.personal_best_fitness = self.fitness.copy()
        best_idx = np.argmin(self.fitness)
        self.global_best = self.population[best_idx].copy()
        self.global_best_fitness = self.fitness[best_idx]
        self.best_history.append(self.global_best_fitness)

    def _update_personal_best_batch(self):
        """Update personal best positions using vectorized comparison."""
        improved_mask = self.fitness < self.personal_best_fitness
        self.personal_best[improved_mask] = self.population[improved_mask]
        self.personal_best_fitness[improved_mask] = self.fitness[improved_mask]

    def _update_global_best_batch(self):
        """Update global best position."""
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.fitness[best_idx]
            self.global_best = self.population[best_idx].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

    def _compute_ring_topology_batch(self):
        """Compute ring neighborhood indices for all particles as 2D array."""
        half = self.neighborhood_size // 2
        indices = np.arange(self.np)
        neighborhood_indices = np.zeros((self.np, self.neighborhood_size), dtype=int)
        for offset in range(-half, half + 1):
            neighborhood_indices[:, offset + half] = (indices + offset) % self.np
        return neighborhood_indices

    def _compute_neighborhood_best_batch(self):
        """Compute neighborhood best positions using batched operations."""
        neighborhood_indices = self._compute_ring_topology_batch()
        n_rows = self.neighborhood_size
        best_neighbor_idx = np.full(self.np, -1, dtype=int)
        
        for c in range(n_rows):
            col_indices = neighborhood_indices[:, c]
            col_fitness = self.personal_best_fitness[col_indices]
            if c == 0:
                current_best_fit = col_fitness.copy()
            else:
                better_mask = col_fitness < current_best_fit
                current_best_fit[better_mask] = col_fitness[better_mask]
                best_neighbor_idx[better_mask] = col_indices[better_mask]
        
        best_neighbor_idx[best_neighbor_idx == -1] = neighborhood_indices[:, 0][best_neighbor_idx == -1]
        
        neighborhood_best = np.zeros((self.np, self.dim))
        for i in range(self.np):
            neighborhood_best[i] = self.personal_best[best_neighbor_idx[i]]
        return neighborhood_best

    def _compute_fitness_rank_batch(self):
        """Compute normalized fitness ranks [0, 1] for all particles."""
        sorted_order = np.argsort(self.fitness)
        ranks = np.empty(self.np)
        ranks[sorted_order] = np.arange(self.np) / max(1, self.np - 1)
        return ranks

    def _compute_quantum_perturbation_batch(self):
        """Compute quantum-inspired perturbation based on fitness ranks."""
        ranks = self._compute_fitness_rank_batch()
        
        quantum_angles = self.quantum_rotation_base * (1.0 - ranks * 0.5)
        
        tan_samples = np.tan(np.random.uniform(0, np.pi / 2, (self.np, self.dim)))
        perturbation = quantum_angles[:, np.newaxis] * tan_samples
        
        sign_matrix = np.random.choice([-1, 1], size=(self.np, self.dim))
        perturbation = perturbation * sign_matrix
        
        return perturbation

    def _velocity_update_batch(self, neighborhood_best):
        """Update velocity using PSO formula with neighborhood influence."""
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.cognitive_param * r1 * (self.personal_best - self.population)
        social = self.social_param * r2 * (neighborhood_best - self.population)
        
        self.velocity = self.inertia_weight * self.velocity + cognitive + social

    def _position_update_batch(self):
        """Update positions using velocity."""
        self.population = self.population + self.velocity

    def _position_update_temporal_drift(self):
        """Add temporal drift component to positions for exploration."""
        drift_strength = 0.05 * (1.0 - min(self.gen / 200, 1.0))
        drift = np.random.randn(self.np, self.dim) * drift_strength
        self.population = self.population + drift

    def _position_update_fitness_rank(self):
        """Modify position update strength based on fitness rank."""
        ranks = self._compute_fitness_rank_batch()
        rank_adjustment = 1.0 + ranks[:, np.newaxis] * 0.2
        self.population = self.population * rank_adjustment

    def _apply_quantum_rotation_batch(self):
        """Apply quantum rotation perturbation to population."""
        perturbation = self._compute_quantum_perturbation_batch()
        self.population = self.population + perturbation

    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on stagnation and diversity."""
        if self.stagnation_count > 10:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif self.stagnation_count == 0:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.98)

    def _adapt_quantum_rotation(self):
        """Adapt quantum rotation base angle based on progress."""
        if len(self.best_history) >= 10:
            recent_improvement = self.best_history[-10] - self.global_best_fitness
            if recent_improvement < 1e-6:
                self.quantum_rotation_base = min(0.3, self.quantum_rotation_base * 1.1)
            else:
                self.quantum_rotation_base = max(0.01, self.quantum_rotation_base * 0.97)

    def _adapt_neighborhood_size(self):
        """Adapt neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        if diversity < self.diversity_threshold:
            self.neighborhood_size = min(self.np // 3, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold * 3:
            self.neighborhood_size = max(3, self.neighborhood_size - 1)

    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        if self.population is None or len(self.population) < 2:
            return 1.0
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances) / (self.upper_bound - self.lower_bound)

    def _generate_opposition_batch(self):
        """Generate opposition-based candidates."""
        if np.random.random() < self.opposition_rate:
            center = (self.lower_bound + self.upper_bound) / 2.0
            opp_pop = 2.0 * center - self.population
            return self._bound_candidates(opp_pop)
        return None

    def _select_survivors_batch(self, func):
        """Select survivors combining current and opposition population."""
        opp_pop = self._generate_opposition_batch()
        if opp_pop is not None:
            opp_fitness = self._evaluate_batch(func, opp_pop)
            
            combined_pop = np.vstack([self.population, opp_pop])
            combined_fit = np.concatenate([self.fitness, opp_fitness])
            
            sorted_indices = np.argsort(combined_fit)[:self.np]
            self.population = combined_pop[sorted_indices]
            self.fitness = combined_fit[sorted_indices]

    def _restart_if_stagnant(self):
        """Reinitialize portion of population if stagnated."""
        if self.stagnation_count > 30:
            n_replace = self.np // 3
            worst_indices = np.argsort(self.fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                -50, 50, (n_replace, self.dim)
            )
            self.stagnation_count = 0

    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        self._initialize_population()
        self._initialize_velocity()
        self.fitness = self._evaluate_batch(func, self.population)
        self._initialize_best_tracking()
        
        while not stopping_condition():
            self.gen += 1
            
            self._adapt_inertia_weight()
            self._adapt_quantum_rotation()
            self._adapt_neighborhood_size()
            
            neighborhood_best = self._compute_neighborhood_best_batch()
            
            self._velocity_update_batch(neighborhood_best)
            self._position_update_batch()
            self._apply_quantum_rotation_batch()
            self._position_update_temporal_drift()
            self._position_update_fitness_rank()
            
            self.population = self._bound_candidates(self.population)
            
            if stopping_condition():
                break
                
            self.fitness = self._evaluate_batch(func, self.population)
            
            if len(self.fitness) < self.np:
                break
            
            self._update_personal_best_batch()
            self._update_global_best_batch()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch(func)
            
            self.best_history.append(self.global_best_fitness)
            self._restart_if_stagnant()
            
            if len(self.best_history) > 100:
                self.best_history = self.best_history[-100:]
        
        return self.global_best_fitness, self.global_best
```