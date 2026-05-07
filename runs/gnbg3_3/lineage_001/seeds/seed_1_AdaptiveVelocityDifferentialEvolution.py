import numpy as np


class AdaptiveVelocityDifferentialEvolution:
    """
    Adaptive Velocity-guided Differential Evolution (AVDE).
    
    Key features:
    - DE/current-to-pbest/1 mutation backbone with velocity-guided perturbation
    - Ring topology for neighborhood information
    - Adaptive F, CR, and inertia weight parameters
    - Stagnation-triggered reinitialization
    - Fully batched operations for efficiency
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = kwargs.get('np', 5 * dim)
        self.np = min(self.np, 500)
        self.max_gen = kwargs.get('max_gen', 1000)
        self.bounds = np.array([-100.0, 100.0])
        self.archive = []
        
        self.F = 0.5
        self.CR = 0.5
        self.inertia = 0.9
        self.p_best = 0.1
        self.tau_F = 0.1
        self.tau_CR = 0.1
        self.stagnation_count = 0
        self.stagnation_best_fitness = np.inf
        self.prev_positions = None
        self.velocity = None
        
    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling for space-filling."""
        pop = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            bins = np.linspace(0, 1, self.np + 1)
            samples = np.random.uniform(bins[:-1], bins[1:])
            np.random.shuffle(samples)
            pop[:, d] = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return pop
    
    def _build_ring_topology(self):
        """Build ring topology indices for neighborhood-based mutation."""
        topology = np.zeros((self.np, 3), dtype=np.int32)
        for i in range(self.np):
            left = (i - 1) % self.np
            right = (i + 1) % self.np
            opposite = (i + self.np // 2) % self.np
            topology[i] = [left, right, opposite]
        return topology
    
    def _mutate_current_to_pbest_batch(self, population, fitness, topology, gen):
        """
        Generate current-to-pbest/1/bin mutation vectors using ring topology.
        Uses archive and ring neighbors for base selection.
        """
        np_pop = len(population)
        archive_size = len(self.archive)
        
        p_indices = np.random.randint(0, max(1, int(self.p_best * np_pop)), size=np_pop)
        sorted_idx = np.argsort(fitness)
        pbest_indices = sorted_idx[p_indices]
        
        base_indices = np.array([topology[i, 0] for i in range(np_pop)])
        
        r1_indices = np.array([topology[i, 1] for i in range(np_pop)])
        r2_indices = np.array([topology[i, 2] for i in range(np_pop)])
        
        if archive_size >= 2:
            archive_arr = np.array(self.archive)
            arch_rand = np.random.randint(0, archive_size, size=np_pop)
            archive_indices = arch_rand
            valid_archive = archive_arr[archive_indices]
            archive_vec = valid_archive - population[r1_indices]
        else:
            archive_vec = population[r2_indices] - population[r1_indices]
        
        pbest_vec = population[pbest_indices] - population
        ring_vec = population[r2_indices] - population[base_indices]
        
        F_adapted = self.F * (0.9 + 0.2 * np.random.rand(np_pop))
        
        mutation = population[base_indices] + F_adapted.reshape(-1, 1) * pbest_vec
        mutation += self.F * archive_vec + 0.5 * self.F * ring_vec
        
        return mutation
    
    def _apply_velocity_perturbation_batch(self, population, mutation):
        """
        Apply velocity-guided perturbation inspired by PSO.
        Maintains search momentum from previous successful directions.
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(population)
        
        vel_decay = 0.7
        self.velocity = vel_decay * self.velocity + (mutation - population)
        
        if self.prev_positions is not None:
            direction = mutation - self.prev_positions
            self.velocity = 0.5 * self.velocity + 0.3 * direction
        
        vel_perturbation = self.inertia * self.velocity
        mutation = mutation + vel_perturbation
        
        max_vel = 0.5 * (self.bounds[1] - self.bounds[0])
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
        
        self.prev_positions = population.copy()
        
        return mutation
    
    def _crossover_batch(self, population, mutation):
        """Binomial crossover between target and mutation vectors."""
        np_pop, dim = population.shape
        trials = np.copy(population)
        
        j_rand = np.random.randint(0, dim)
        mask = np.random.rand(np_pop, dim) < self.CR
        
        trials = np.where(mask, mutation, population)
        trials[:, j_rand] = mutation[:, j_rand]
        
        return trials
    
    def _clip_to_bounds_batch(self, trials):
        """Ensure all trial vectors are within search bounds."""
        return np.clip(trials, self.bounds[0], self.bounds[1])
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection between target and trial vectors."""
        better = trial_fitness < fitness
        
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        
        for i in range(len(population)):
            if better[i] and len(self.archive) < 5 * self.np:
                self.archive.append(population[i].copy())
        
        best_idx = np.argmin(new_fit)
        
        return new_pop, new_fit, new_fit[best_idx], new_pop[best_idx], best_idx
    
    def _adapt_parameters(self, gen):
        """Adapt F, CR, and inertia weight based on generation progress."""
        self.F = 0.5 if np.random.rand() > self.tau_F else np.clip(0.1 + 0.9 * np.random.rand(), 0.1, 0.9)
        self.CR = 0.5 if np.random.rand() > self.tau_CR else np.clip(np.random.rand(), 0.3, 0.7)
        
        progress = gen / max(1, self.max_gen)
        self.inertia = 0.9 - 0.7 * (progress ** 0.5)
        self.inertia = np.clip(self.inertia, 0.1, 0.9)
    
    def _adapt_velocity(self, improvement_rate):
        """Adapt inertia weight based on recent improvement rate."""
        if improvement_rate > 0.1:
            self.inertia = np.clip(self.inertia * 1.05, 0.1, 0.9)
        elif improvement_rate < 0.01:
            self.inertia = np.clip(self.inertia * 0.95, 0.1, 0.9)
    
    def _check_stagnation_and_restart(self, population, fitness, best_fitness):
        """Check for stagnation and reinitialize part of the population if needed."""
        if best_fitness >= self.stagnation_best_fitness:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.stagnation_best_fitness = best_fitness
            return population, fitness
        
        if self.stagnation_count > 50:
            stagnation_ratio = min(0.5, self.stagnation_count / 500)
            n_replace = max(1, int(stagnation_ratio * self.np))
            
            worst_indices = np.argsort(fitness)[-n_replace:]
            best_idx = np.argmin(fitness)
            best_pos = population[best_idx]
            
            for idx in worst_indices:
                scale = np.random.uniform(0.5, 2.0)
                noise = np.random.randn(self.dim) * scale
                population[idx] = best_pos + noise
                population[idx] = np.clip(population[idx], self.bounds[0], self.bounds[1])
                fitness[idx] = np.inf
            
            self.stagnation_count = 0
            self.velocity = None
            self.prev_positions = None
            self.archive = []
        
        return population, fitness
    
    def _compute_diversity(self, population):
        """Compute population diversity as average distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop with batched operations."""
        population = self._initialize_population()
        population = self._clip_to_bounds_batch(population)
        
        fitness = func(population)
        if len(fitness) < self.np:
            population = population[:len(fitness)]
            fitness = fitness[:len(fitness)]
            self.np = len(population)
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_position = population[best_idx].copy()
        
        topology = self._build_ring_topology()
        self.velocity = np.zeros_like(population)
        self.prev_positions = population.copy()
        
        gen = 0
        while not stopping_condition():
            gen += 1
            
            self._adapt_parameters(gen)
            
            mutation = self._mutate_current_to_pbest_batch(population, fitness, topology, gen)
            mutation = self._apply_velocity_perturbation_batch(population, mutation)
            
            trials = self._crossover_batch(population, mutation)
            trials = self._clip_to_bounds_batch(trials)
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                break
            
            if stopping_condition():
                break
            
            population, fitness, curr_best_fit, curr_best_pos, curr_best_idx = \
                self._select_survivors_batch(population, fitness, trials, trial_fitness)
            
            if curr_best_fit < best_fitness:
                improvement = best_fitness - curr_best_fit
                improvement_rate = improvement / max(1e-10, abs(best_fitness))
                self._adapt_velocity(improvement_rate)
                best_fitness = curr_best_fit
                best_position = curr_best_pos.copy()
            
            population, fitness = self._check_stagnation_and_restart(
                population, fitness, best_fitness
            )
            
            if gen % 100 == 0:
                diversity = self._compute_diversity(population)
        
        return best_fitness, best_position
