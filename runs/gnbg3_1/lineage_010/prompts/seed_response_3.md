```python
import numpy as np


class AdaptiveTopologyParticleSwarmOptimizer:
    """
    AT-PSO-CL: Adaptive Topology Particle Swarm with Covariance Learning.
    
    A PSO-based optimizer with:
    - Ring neighborhood topology for structured information flow
    - Adaptive topology switching (ring vs fully-connected)
    - Covariance learning from successful personal bests
    - Adaptive inertia weight and step size
    - Operator selection with reward tracking
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(4 * dim, 200)
        self.lb = -100.0
        self.ub = 100.0
        
        # Inertia and acceleration parameters
        self.chi = 0.729
        self.c1_init = 1.496
        self.c2_init = 1.496
        self.c1 = self.c1_init
        self.c2 = self.c2_init
        
        # Step size control
        self.step_size = 1.0
        self.step_size_min = 0.01
        self.step_size_max = 2.0
        
        # Diversity and stagnation
        self.diversity_target = 0.1
        self.stagnation_tol = 1e-8
        self.stagnation_counter = 0
        self.max_stagnation = 20
        
        # Topology: 0=ring, 1=fully_connected
        self.topology = 0
        self._build_ring_topology()
        
        # Operator selection (3 operators)
        self.operator_weights = np.ones(3)
        self.operator_rewards = np.zeros(3)
        self.learning_rate = 0.1
        
        # Covariance adaptation
        self.cov_adapt_enabled = True
        self.cov_dimension = min(dim, 30)
        self.success_history = []
        
        # Tracking
        self.gbest_history = []
        
    def _build_ring_topology(self):
        """Build ring neighborhood connections."""
        self.topology_neighbors = np.zeros((self.np, 2), dtype=int)
        for i in range(self.np):
            self.topology_neighbors[i, 0] = (i - 1) % self.np
            self.topology_neighbors[i, 1] = (i + 1) % self.np
    
    def _initialize_population(self):
        """Initialize swarm uniformly in bounds."""
        return np.random.uniform(self.lb, self.ub, (self.np, self.dim))
    
    def _initialize_personal_best(self, population, fitness):
        """Set initial personal bests."""
        return population.copy(), fitness.copy()
    
    def _evaluate_batch(self, population, func, stopping_condition):
        """Evaluate entire population in one batch call."""
        if stopping_condition():
            return np.array([])
        fitness = func(population)
        if len(fitness) < len(population):
            return np.array([])
        return fitness
    
    def _update_velocity_ring_batch(self, population, velocities, pbest, gbest_neighbor):
        """Update velocities using ring neighborhood structure."""
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.c1 * r1 * (pbest - population)
        
        social = np.zeros((self.np, self.dim))
        for i in range(self.np):
            left, right = self.topology_neighbors[i]
            better = pbest[left] if pbest[left] is not None and pbest[right] is not None else None
            if pbest[left] is not None:
                social[i] = pbest[left] if pbest[left] is not None else gbest_neighbor
            if pbest[right] is not None:
                if social[i] is None or (pbest[right] < pbest[left]).all():
                    social[i] = pbest[right]
            if social[i] is None:
                social[i] = gbest_neighbor
        
        social = self.c2 * r2 * (social - population)
        velocities[:] = self.chi * (velocities + cognitive + social)
        
        max_vel = 0.2 * (self.ub - self.lb)
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities = np.where(norms > max_vel, velocities * max_vel / norms, velocities)
    
    def _update_velocity_fullyconnected_batch(self, population, velocities, pbest, gbest_pos):
        """Update velocities with fully-connected social structure."""
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.c1 * r1 * (pbest - population)
        social = self.c2 * r2 * (gbest_pos - population)
        
        velocities[:] = self.chi * (velocities + cognitive + social)
        
        max_vel = 0.2 * (self.ub - self.lb)
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities = np.where(norms > max_vel, velocities * max_vel / norms, velocities)
    
    def _update_position_batch(self, population, velocities):
        """Update positions with velocity and clip to bounds."""
        population[:] += velocities
        np.clip(population, self.lb, self.ub, out=population)
    
    def _update_personal_best_batch(self, population, fitness, pbest, pbest_fitness):
        """Update personal bests where improvement occurred."""
        improved = fitness < pbest_fitness
        pbest_fitness[improved] = fitness[improved]
        pbest[improved] = population[improved].copy()
        return improved
    
    def _update_global_best_batch(self, pbest_fitness, pbest, gbest_pos, gbest_fitness):
        """Update global best from personal bests."""
        idx = np.argmin(pbest_fitness)
        if pbest_fitness[idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[idx]
            gbest_pos[:] = pbest[idx]
        return gbest_fitness, gbest_pos, idx
    
    def _compute_diversity(self, population):
        """Compute normalized swarm diversity measure."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances) / (self.ub - self.lb)
    
    def _check_stagnation(self, old_gbest, new_gbest):
        """Detect stagnation based on global best improvement."""
        improvement = old_gbest - new_gbest
        self.stagnation_counter = self.stagnation_counter + 1 if improvement < self.stagnation_tol else 0
        return self.stagnation_counter >= self.max_stagnation
    
    def _adapt_parameters(self, fitness, gbest_fitness, gbest_init):
        """Adapt inertia weight and step size based on progress."""
        progress_ratio = max(0.0, min(1.0, (gbest_init - gbest_fitness) / (gbest_init + 1e-10)))
        
        self.chi = 0.729 - 0.2 * progress_ratio
        
        self.c1 = self.c1_init * (1 - 0.3 * progress_ratio)
        self.c2 = self.c2_init * (1 + 0.3 * progress_ratio)
        
        fitness_variance = np.var(fitness)
        if fitness_variance < 1e-6:
            self.step_size = min(self.step_size * 1.2, self.step_size_max)
        else:
            self.step_size = max(self.step_size * 0.95, self.step_size_min)
    
    def _adapt_topology(self, diversity):
        """Switch topology based on diversity level."""
        if diversity < self.diversity_target * 0.5 and self.topology == 0:
            self.topology = 1
        elif diversity > self.diversity_target * 2.0 and self.topology == 1:
            self.topology = 0
    
    def _generate_covariance_trials(self, pbest, trial_count):
        """Generate trials using covariance learning from personal bests."""
        if len(self.success_history) < 5:
            return None
        
        samples = np.array(self.success_history[-50:])
        if len(samples) < 5:
            return None
        
        centroid = np.mean(samples, axis=0)
        
        centered = samples - centroid
        cov = np.cov(centered.T)
        cov += 0.001 * np.eye(self.dim)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            return None
        
        perturbation = sqrt_cov @ np.random.randn(self.dim, trial_count)
        perturbation = perturbation.T * self.step_size * 0.5
        
        base_idx = np.random.choice(len(pbest), trial_count)
        trials = pbest[base_idx] + perturbation
        
        return trials
    
    def _generate_operator_trials(self, population, pbest, gbest_pos, operator_idx):
        """Generate trials using selected operator strategy."""
        if operator_idx == 0:
            base_idx = np.random.choice(self.np, self.np)
            r = np.random.uniform(0.5, 1.5, (self.np, 1))
            trials = pbest[base_idx] + r * (np.random.rand(self.np, 1) * (pbest - population) +
                                            np.random.rand(self.np, 1) * (gbest_pos - pbest[base_idx]))
        elif operator_idx == 1:
            trials = gbest_pos + self.step_size * np.random.randn(self.np, self.dim)
        else:
            cov_trials = self._generate_covariance_trials(pbest, self.np)
            if cov_trials is not None:
                trials = cov_trials
            else:
                trials = population + self.step_size * np.random.randn(self.np, self.dim)
        
        np.clip(trials, self.lb, self.ub, out=trials)
        return trials
    
    def _select_operator(self):
        """Select operator using weighted probability."""
        weights = np.maximum(self.operator_weights, 0.01)
        probs = weights / np.sum(weights)
        return np.random.choice(len(self.operator_weights), p=probs)
    
    def _update_operator_rewards(self, fitness, trial_fitness, operator_idx):
        """Update operator weights based on performance."""
        improvement = np.mean(fitness) - np.mean(trial_fitness)
        if improvement > 0:
            self.operator_rewards[operator_idx] += improvement * 10
        else:
            self.operator_rewards[operator_idx] -= 0.01
        
        self.operator_weights += self.learning_rate * self.operator_rewards
        self.operator_weights = np.maximum(self.operator_weights, 0.01)
    
    def _restart_if_stagnant(self, population, pbest, pbest_fitness, gbest_pos, gbest_fitness, func):
        """Reinitialize swarm while preserving best solution."""
        n_replace = max(5, self.np // 5)
        replace_idx = np.random.choice(self.np, n_replace, replace=False)
        
        new_particles = np.random.randn(n_replace, self.dim)
        new_particles = gbest_pos + new_particles * self.step_size * 2.0
        np.clip(new_particles, self.lb, self.ub, out=new_particles)
        
        population[replace_idx] = new_particles
        pbest[replace_idx] = population[replace_idx].copy()
        
        if len(pbest_fitness) > 0 and len(population) > 0:
            eval_batch = population[replace_idx]
            new_fitness = func(eval_batch)
            if len(new_fitness) == len(replace_idx):
                pbest_fitness[replace_idx] = new_fitness
        
        self.stagnation_counter = 0
        self.success_history = []
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        population = self._initialize_population()
        velocities = np.zeros((self.np, self.dim))
        
        fitness = self._evaluate_batch(population, func, stopping_condition)
        if len(fitness) == 0:
            return np.inf, np.zeros(self.dim)
        
        pbest, pbest_fitness = self._initialize_personal_best(population, fitness)
        gbest_pos = pbest[np.argmin(pbest_fitness)].copy()
        gbest_fitness = np.min(pbest_fitness)
        gbest_init = gbest_fitness
        
        self.gbest_history = [gbest_fitness]
        old_gbest = gbest_fitness
        
        while not stopping_condition():
            if self.topology == 0:
                self._update_velocity_ring_batch(population, velocities, pbest, gbest_pos)
            else:
                self._update_velocity_fullyconnected_batch(population, velocities, pbest, gbest_pos)
            
            self._update_position_batch(population, velocities)
            
            operator_idx = self._select_operator()
            trials = self._generate_operator_trials(population, pbest, gbest_pos, operator_idx)
            
            trial_fitness = self._evaluate_batch(trials, func, stopping_condition)
            if len(trial_fitness) < self.np:
                break
            
            if stopping_condition():
                break
            
            improved = self._update_personal_best_batch(trials, trial_fitness, pbest, pbest_fitness)
            
            for idx in np.where(improved)[0]:
                self.success_history.append(pbest[idx].copy())
            if len(self.success_history) > 200:
                self.success_history = self.success_history[-200:]
            
            gbest_fitness, gbest_pos, gbest_idx = self._update_global_best_batch(
                pbest_fitness, pbest, gbest_pos, gbest_fitness)
            
            self._update_operator_rewards(fitness, trial_fitness, operator_idx)
            
            self._adapt_parameters(fitness, gbest_fitness, gbest_init)
            
            diversity = self._compute_diversity(population)
            self._adapt_topology(diversity)
            
            if self._check_stagnation(old_gbest, gbest_fitness):
                self._restart_if_stagnant(population, pbest, pbest_fitness,
                                         gbest_pos, gbest_fitness, func)
            
            old_gbest = gbest_fitness
            self.gbest_history.append(gbest_fitness)
            fitness = trial_fitness
        
        return gbest_fitness, gbest_pos
```