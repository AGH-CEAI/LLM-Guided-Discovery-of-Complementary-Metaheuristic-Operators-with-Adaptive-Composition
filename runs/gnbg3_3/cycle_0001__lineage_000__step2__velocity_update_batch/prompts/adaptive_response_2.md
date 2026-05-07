```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Adaptive Topology Particle Swarm Optimizer with PROBE-AND-COMMIT Variant Selection.
    
    MECHANISM: Contextual Bandit / Probe-and-Commit (#4 from menu)
    
    WHY THIS FITS THE GAP TABLE:
    - Task 5 has a 13917× gap to median competitor → BLEND-HOSTILE
    - Task 3 has a 9.3× gap to median → BLEND-HOSTILE
    - Ensemble blending CANNOT preserve per-task advantages when gaps are this large.
      A 50/50 blend of winner (3.12e+01) and runner-up (6.03e+01) yields ~4.57e+01,
      which is 1.5× worse than winner alone.
    - PROBE-AND-COMMIT dispatches entirely to the probe winner (weight 1.0),
      preserving the per-task advantage with no blending penalty.
    
    PHASE A (PROBE, ~10% of budget):
      - Run each of 6 velocity-update variants for a short burst (8 gens, small sub-pop)
      - The variant with best rank-normalized convergence rate is selected
      
    PHASE B (COMMIT):
      - Run selected variant for remaining budget with full population
      - Small ε probability of re-probing if stagnation detected
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Probe state
        self.selected_variant = 10  # Default to most-winning variant
        
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
        
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on swarm diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        
        return cognitive, social
    
    def _velocity_update_variant_0_original(self, pop, vel, pbest, lbest, pbest_fit, gbest, stagnation, gen, probe_np, n_size):
        """Original velocity update with DE/rand/1 mutation."""
        improvement = 1.0 / (1.0 + stagnation)
        cogn = self.cognitive_base * (1.0 + 0.5 * improvement)
        soc = self.social_base * (2.0 - improvement)
        
        r1 = np.random.uniform(0, 1, (probe_np, self.dim))
        r2 = np.random.uniform(0, 1, (probe_np, self.dim))
        
        cogn_comp = cogn * r1 * (pbest - pop)
        soc_comp = soc * r2 * (lbest - pop)
        
        mut_mask = np.random.uniform(0, 1, (probe_np, self.dim))
        mut_thresh = 0.1 * (1.0 - gen / 5000)
        mut_active = mut_mask < mut_thresh
        
        mut_vecs = np.zeros((probe_np, self.dim))
        for i in range(probe_np):
            idx = np.random.choice([j for j in range(probe_np) if j != i], 3, replace=False)
            mut_vecs[i] = pop[idx[0]] + 0.5 * (pop[idx[1]] - pop[idx[2]])
        
        mut_comp = np.where(mut_active, 0.3 * (mut_vecs - pop), 0.0)
        
        new_vel = 0.729 * vel + cogn_comp + soc_comp + mut_comp
        return np.clip(new_vel, self.v_min, self.v_max)
    
    def _velocity_update_variant_1(self, pop, vel, pbest, lbest, pbest_fit, gbest, stagnation, gen, probe_np, n_size):
        """Geometric centroid + k-NN repulsion."""
        centroid = np.mean(pop, axis=0)
        min_pos = np.min(pop, axis=0)
        max_pos = np.max(pop, axis=0)
        spread = max_pos - min_pos + 1e-10
        dist_to_centroid = np.linalg.norm(pop - centroid, axis=1, keepdims=True)
        
        r1 = np.random.uniform(0, 1, (probe_np, self.dim))
        r2 = np.random.uniform(0, 1, (probe_np, self.dim))
        
        cogn = self.cognitive_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        soc = self.social_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        
        cogn_comp = cogn * r1 * (pbest - pop)
        soc_comp = soc * r2 * (lbest - pop)
        
        centroid_dir = centroid - pop
        centroid_strength = np.exp(-dist_to_centroid / 100.0)
        centroid_comp = 0.3 * centroid_strength * centroid_dir
        
        k = min(5, probe_np - 1)
        knn_repulsion = np.zeros((probe_np, self.dim))
        for i in range(probe_np):
            diffs = pop - pop[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nn_idx = np.argpartition(dists, k)[:k]
            nn_dists = dists[nn_idx]
            for j, nb in enumerate(nn_idx):
                if nn_dists[j] > 1e-10:
                    knn_repulsion[i] += (pop[i] - pop[nb]) / (nn_dists[j] ** 2 + 1e-10)
        knn_comp = 0.1 * knn_repulsion
        
        spread_mod = np.mean(spread) / (spread + 1e-10)
        
        new_vel = (0.729 * vel + cogn_comp + soc_comp + centroid_comp + knn_comp) * spread_mod
        return np.clip(new_vel, self.v_min, self.v_max)
    
    def _velocity_update_variant_3(self, pop, vel, pbest, lbest, pbest_fit, gbest, stagnation, gen, probe_np, n_size):
        """Entropy-modulated distribution-guided exploration."""
        improvement = 1.0 / (1.0 + stagnation)
        cogn = self.cognitive_base * (1.0 + 0.5 * improvement)
