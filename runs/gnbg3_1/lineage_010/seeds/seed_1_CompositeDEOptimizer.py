import numpy as np


class CompositeDEOptimizer:
    """
    Composite Mutation Differential Evolution with Adaptive Operator Selection.
    
    Key features:
    - Composite mutation blending current-to-pbest, best, and rand strategies
    - Adaptive operator weights via success-based reinforcement learning
    - SHADE-style F/CR adaptation with per-operator memory
    - Archive of rejected solutions for enhanced diversity
    - Stagnation detection with opposition-based reinitialization
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_init = max(20, 4 * dim)
        self.max_generations = kwargs.get('max_generations', 10000)
        self.target_fitness = kwargs.get('target_fitness', 1e-8)
        
    def __call__(self, func, stopping_condition):
        population, fitness = self._initialize_population(func)
        
        while not stopping_condition():
            # Generate trial batch (vectorized)
            trials, op_idx = self._generate_trial_batch(population, fitness)
            
            # Evaluate trial batch
            trial_fit = func(trials)
            
            # Handle budget exhaustion gracefully
            if len(trial_fit) < len(trials):
                trials = trials[:len(trial_fit)]
                trial_fit = trial_fit[:len(trial_fit)]
            
            # Check stopping after evaluation
            if stopping_condition():
                break
                
            # Select survivors and update archives
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fit
            )
            
            # Update operator rewards based on success
            self._update_operator_rewards(op_idx, improved_mask)
            
            # Adapt F/CR parameters
            F_new, CR_new = self._adapt_parameters_batch(op_idx, improved_mask, fitness, trial_fit)
            self.current_F = F_new
            self.current_CR = CR_new
            
            # Check stagnation and diversity
            self.stagnation_count, diversity = self._check_stagnation(fitness)
            self._compute_diversity(population)
            
            # Reinitialize if stuck
            if self.stagnation_count >= self.max_stagnation or diversity < self.min_diversity:
                population, fitness = self._restart_if_stagnant(population, fitness, func)
            
            self.prev_fitness = fitness.copy()
        
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx].copy()
    
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling for better spread."""
        self.np = self.np_init
        segments = np.linspace(0, 1, self.np + 1)
        
        population = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            offsets = np.random.rand(self.np)
            population[:, d] = self.lb + (segments[:-1] + offsets) * (self.ub - self.lb) / self.np * self.np
        
        # Shuffle each dimension independently
        for d in range(self.dim):
            np.random.shuffle(population[:, d])
        
        self.population = np.clip(population, self.lb, self.ub)
        self.population_fitness = func(self.population)
        
        # Track best solution
        best_idx = np.argmin(self.population_fitness)
        self.best_fitness = self.population_fitness[best_idx]
        self.best_solution = self.population[best_idx].copy()
        
        # Initialize archive
        self.archive = []
        self.archive_max = self.np * 2
        
        # Operator configuration
        self.num_operators = 3
        self.operator_names = ['current_to_pbest', 'best', 'rand']
        self.operator_weights = np.array([0.8, 0.1, 0.1])
        self.operator_success = np.zeros(self.num_operators)
        self.operator_counts = np.zeros(self.num_operators)
        
        # Per-operator F/CR memory (SHADE-style)
        self.h_F = np.ones(self.num_operators) * 0.5
        self.h_CR = np.ones(self.num_operators) * 0.9
        self.success_F_lists = [[] for _ in range(self.num_operators)]
        self.success_CR_lists = [[] for _ in range(self.num_operators)]
        
        # Current parameters
        self.current_F = np.full(self.np, 0.5)
        self.current_CR = np.full(self.np, 0.9)
        
        # Stagnation tracking
        self.prev_fitness = np.full(self.np, np.inf)
        self.stagnation_count = 0
        self.max_stagnation = 15
        self.min_diversity = 0.05 * (self.ub - self.lb) * np.sqrt(self.dim) / self.np
        self.diversity_history = []
        
        return self.population, self.population_fitness
    
    def _select_mutation_indices_batch(self, target_indices):
        """Select 3 distinct indices excluding target for each individual."""
        np_op = len(target_indices)
        all_indices = np.arange(self.np)
        
        mutation_indices = np.zeros((np_op, 3), dtype=int)
        for i, tidx in enumerate(target_indices):
            mask = all_indices != tidx
            candidates = all_indices[mask]
            selected = np.random.choice(candidates, size=3, replace=False)
            mutation_indices[i] = selected
        
        return mutation_indices
    
    def _select_pbest_batch(self, fitness, p_percent=0.05):
        """Select p-best indices for current-to-pbest mutation."""
        p_size = max(2, int(self.np * p_percent))
        sorted_indices = np.argsort(fitness)
        pbest_pool = sorted_indices[:p_size]
        pbest_indices = np.random.choice(pbest_pool, size=self.np)
        return pbest_indices
    
    def _sample_archive_batch(self, size):
        """Sample indices from archive for mutation vectors."""
        if len(self.archive) == 0:
            return None
        archive_arr = np.array(self.archive)
        if len(archive_arr) >= size:
            indices = np.random.choice(len(archive_arr), size=size, replace=False)
        else:
            indices = np.random.choice(len(archive_arr), size=size, replace=True)
        return archive_arr[indices]
    
    def _generate_mutant_batch(self, population, fitness, op_idx):
        """Composite mutation blending multiple strategies with adaptive weights."""
        np_op, dim = population.shape
        
        # Generate indices for all mutation strategies
        target_indices = np.arange(np_op)
        mut_indices = self._select_mutation_indices_batch(target_indices)
        r1, r2, r3 = mut_indices[:, 0], mut_indices[:, 1], mut_indices[:, 2]
        
        pbest_indices = self._select_pbest_batch(fitness)
        archive_samples = self._sample_archive_batch(np_op)
        
        # Generate F values for each individual
        F = self.current_F.copy()
        F = np.clip(F, 0.0, 2.0)
        
        # Strategy 1: Current-to-pbest with archive
        if archive_samples is not None:
            strategy1 = population[target_indices] + F[:, np.newaxis] * (
                population[pbest_indices] - archive_samples
            )
        else:
            strategy1 = population[target_indices] + F[:, np.newaxis] * (
                population[pbest_indices] - population[r1]
            )
        
        # Strategy 2: Best/1-style
        best_idx = np.argmin(fitness)
        F2 = np.clip(F * 0.8, 0.0, 2.0)
        strategy2 = population[best_idx] + F2[:, np.newaxis] * (
            population[r1] - population[r2]
        )
        
        # Strategy 3: Rand/1-style
        F3 = np.clip(F * 1.2, 0.0, 2.0)
        strategy3 = population[r1] + F3[:, np.newaxis] * (
            population[r2] - population[r3]
        )
        
        # Composite blend based on operator weights
        mutant = (
            self.operator_weights[0] * strategy1 +
            self.operator_weights[1] * strategy2 +
            self.operator_weights[2] * strategy3
        )
        
        mutant = np.clip(mutant, self.lb, self.ub)
        return mutant
    
    def _crossover_batch(self, target_pop, mutant_pop):
        """Binomial crossover with dimension guarantee."""
        cr = np.clip(self.current_CR, 0.0, 1.0)
        dim = target_pop.shape[1]
        
        mask = np.random.rand(self.np, dim) < cr[:, np.newaxis]
        forced_dims = np.arange(dim)
        forced_indices = np.random.randint(0, dim, size=self.np)
        mask[forced_indices, forced_dims] = True
        
        trial = np.where(mask, mutant_pop, target_pop)
        return trial
    
    def _generate_trial_batch(self, population, fitness):
        """Generate full trial batch for current generation."""
        op_idx = self._select_operator_batch()
        mutant = self._generate_mutant_batch(population, fitness, op_idx)
        trial = self._crossover_batch(population, mutant)
        trial = np.clip(trial, self.lb, self.ub)
        return trial, op_idx
    
    def _select_operator_batch(self):
        """Select mutation operator for each individual using weighted roulette."""
        weights = self.operator_weights
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
        
        cumsum = np.cumsum(weights)
        r = np.random.rand(self.np)
        op_indices = np.zeros(self.np, dtype=int)
        
        for i in range(self.np):
            op_indices[i] = np.searchsorted(cumsum, r[i])
        
        return op_indices
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fit):
        """Selection with archive management for rejected individuals."""
        improved = trial_fit < fitness
        
        # Update population with improvements
        new_pop = population.copy()
        new_fit = fitness.copy()
        new_pop[improved] = trials[improved]
        new_fit[improved] = trial_fit[improved]
        
        # Update archive with rejected solutions
        rejected = population[~improved]
        if len(rejected) > 0:
            self.archive.extend(rejected.tolist())
            if len(self.archive) > self.archive_max:
                self.archive = self.archive[-self.archive_max:]
        
        # Update best solution tracking
        best_idx = np.argmin(new_fit)
        if new_fit[best_idx] < self.best_fitness:
            self.best_fitness = new_fit[best_idx]
            self.best_solution = new_pop[best_idx].copy()
        
        return new_pop, new_fit, improved
    
    def _update_operator_rewards(self, op_idx, improved):
        """Update operator weights based on success rates."""
        for op in range(self.num_operators):
            mask = op_idx == op
            successes = np.sum(improved & mask)
            trials = np.sum(mask)
            
            self.operator_success[op] += successes
            self.operator_counts[op] += max(1, trials)
        
        # Compute new weights from success rates
        success_rates = np.array([
            self.operator_success[op] / max(1, self.operator_counts[op])
            for op in range(self.num_operators)
        ])
        
        # Blend historical weights with current success rates
        alpha = 0.7
        self.operator_weights = alpha * success_rates + (1 - alpha) * self.operator_weights
        self.operator_weights = np.maximum(self.operator_weights, 0.05)
        self.operator_weights /= np.sum(self.operator_weights)
    
    def _adapt_parameters_batch(self, op_idx, improved, fitness, trial_fit):
        """Adapt F/CR using SHADE-style success history per operator."""
        delta_fit = fitness - trial_fit
        delta_fit = np.maximum(delta_fit, 1e-12)
        
        new_F = self.current_F.copy()
        new_CR = self.current_CR.copy()
        
        for op in range(self.num_operators):
            op_mask = op_idx == op
            success_mask = op_mask & improved
            
            if np.any(success_mask):
                success_F = self.current_F[success_mask]
                success_CR = self.current_CR[success_mask]
                weights = delta_fit[success_mask]
                weights = weights / np.sum(weights)
                
                self.success_F_lists[op].extend(success_F.tolist())
                self.success_CR_lists[op].extend(success_CR.tolist())
        
        # Update historical means with decay
        for op in range(self.num_operators):
            if len(self.success_F_lists[op]) > 0:
                self.h_F[op] = 0.1 * self.h_F[op] + 0.9 * np.mean(self.success_F_lists[op])
            if len(self.success_CR_lists[op]) > 0:
                self.h_CR[op] = 0.1 * self.h_CR[op] + 0.9 * np.mean(self.success_CR_lists[op])
            
            # Keep lists bounded
            if len(self.success_F_lists[op]) > 50:
                self.success_F_lists[op] = self.success_F_lists[op][-50:]
            if len(self.success_CR_lists[op]) > 50:
                self.success_CR_lists[op] = self.success_CR_lists[op][-50:]
        
        # Generate new F/CR from historical means
        for op in range(self.num_operators):
            op_mask = op_idx == op
            if np.any(op_mask):
                op_count = np.sum(op_mask)
                
                # F: use Lehmer mean for positive selection pressure
                if len(self.success_F_lists[op]) > 0:
                    f_vals = np.array(self.success_F_lists[op][-20:])
                    new_F[op_mask] = np.mean(f_vals ** 2) / (np.mean(f_vals) + 1e-12)
                else:
                    new_F[op_mask] = self.h_F[op] + 0.1 * np.random.randn(op_count)
                
                # CR: use weighted mean
                if len(self.success_CR_lists[op]) > 0:
                    cr_vals = np.array(self.success_CR_lists[op][-20:])
                    new_CR[op_mask] = np.mean(cr_vals)
                else:
                    new_CR[op_mask] = self.h_CR[op] + 0.1 * np.random.randn(op_count)
        
        # Ensure no complete failure mode
        if not np.any(improved):
            new_F = np.clip(new_F * 1.2, 0.3, 1.0)
        
        return new_F, new_CR
    
    def _check_stagnation(self, fitness):
        """Detect stagnation via fitness plateau and track diversity."""
        best_current = np.min(fitness)
        improvement = self.prev_fitness.max() - best_current
        
        if improvement < 1e-6:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        return self.stagnation_count, 0.0
    
    def _compute_diversity(self, population):
        """Compute population diversity as average distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        diversity = np.mean(distances)
        
        self.diversity_history.append(diversity)
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        
        return diversity
    
    def _restart_if_stagnant(self, population, fitness, func):
        """Opposition-based reinitialization to escape local optima."""
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        
        # Generate opposition candidates
        center = (self.lb + self.ub) / 2
        range_val = (self.ub - self.lb) / 2
        opposition = center + (center - population)
        
        # Mix current population with opposition
        combined = np.vstack([population, opposition])
        combined_fitness = func(combined)
        
        # Select best NP individuals
        sorted_indices = np.argsort(combined_fitness)
        selected_indices = sorted_indices[:self.np]
        
        new_pop = np.clip(combined[selected_indices], self.lb, self.ub)
        new_fitness = combined_fitness[selected_indices]
        
        # Reset stagnation counter
        self.stagnation_count = 0
        
        # Reset operator weights to favor exploration after restart
        self.operator_weights = np.array([0.5, 0.25, 0.25])
        
        return new_pop, new_fitness
