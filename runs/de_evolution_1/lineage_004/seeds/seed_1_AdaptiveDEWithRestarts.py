import numpy as np


class AdaptiveDEWithRestarts:
    """
    Adaptive Differential Evolution with multi-strategy mutation,
    parameter adaptation (SHADE-like), stagnation detection, and restarts.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 60), 300)
        self.min_np = max(4, dim)
        
        # SHADE-like memory for F and CR
        self.memory_size = 6
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        
        # Archive for external archive (used in current-to-pbest mutation)
        self.archive_max = self.np_size
        self.archive = []
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        self.best_fitness_history = []
        self.restart_count = 0

    def _initialize_population(self):
        """Create initial random population within bounds."""
        pop = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        return pop

    def _clip_to_bounds(self, pop):
        """Clip population to search bounds."""
        return np.clip(pop, self.lb, self.ub)

    def _generate_parameters_batch(self, size):
        """Generate F and CR parameters using Cauchy/Normal from memory (SHADE-style)."""
        ri = np.random.randint(0, self.memory_size, size=size)
        
        # F from Cauchy distribution
        F = np.zeros(size)
        for i in range(size):
            while True:
                f_val = self.memory_F[ri[i]] + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    break
            F[i] = min(f_val, 1.0)
        
        # CR from Normal distribution
        CR = np.random.normal(self.memory_CR[ri], 0.1, size=size)
        CR = np.clip(CR, 0.0, 1.0)
        
        return F, CR

    def _select_pbest_indices(self, fitness, p_min=0.05, p_max=0.25):
        """Select p-best indices for current-to-pbest mutation."""
        NP = len(fitness)
        p = np.random.uniform(p_min, p_max)
        p_count = max(2, int(NP * p))
        sorted_indices = np.argsort(fitness)
        pbest_pool = sorted_indices[:p_count]
        chosen = pbest_pool[np.random.randint(0, p_count, size=NP)]
        return chosen

    def _mutate_batch(self, population, fitness, F):
        """
        Multi-strategy mutation: current-to-pbest/1 with archive.
        v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2_or_archive)
        """
        NP, dim = population.shape
        pbest_idx = self._select_pbest_indices(fitness)
        
        # Random distinct indices r1
        r1 = np.random.randint(0, NP, size=NP)
        for i in range(NP):
            while r1[i] == i:
                r1[i] = np.random.randint(0, NP)
        
        # r2 from population + archive
        archive_arr = np.array(self.archive) if len(self.archive) > 0 else np.empty((0, dim))
        combined = np.vstack([population, archive_arr]) if len(archive_arr) > 0 else population.copy()
        n_combined = len(combined)
        
        r2 = np.random.randint(0, n_combined, size=NP)
        for i in range(NP):
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, n_combined)
        
        F_expanded = F[:, np.newaxis]
        
        mutants = (population 
                   + F_expanded * (population[pbest_idx] - population) 
                   + F_expanded * (population[r1] - combined[r2]))
        
        return mutants

    def _crossover_batch(self, population, mutants, CR):
        """Binomial crossover applied batch-wise."""
        NP, dim = population.shape
        rand_matrix = np.random.rand(NP, dim)
        j_rand = np.random.randint(0, dim, size=NP)
        
        # Create mask: True where we take from mutant
        mask = rand_matrix < CR[:, np.newaxis]
        # Ensure at least one dimension from mutant
        for i in range(NP):
            mask[i, j_rand[i]] = True
        
        trials = np.where(mask, mutants, population)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection: keep better of parent vs trial."""
        valid = np.isfinite(trial_fitness)
        improved = np.zeros(len(fitness), dtype=bool)
        
        for i in range(len(fitness)):
            if i < len(trial_fitness) and valid[i] and trial_fitness[i] <= fitness[i]:
                improved[i] = True
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        improved_indices = np.where(improved)[0]
        new_pop[improved_indices] = trials[improved_indices]
        new_fit[improved_indices] = trial_fitness[improved_indices]
        
        return new_pop, new_fit, improved

    def _update_archive(self, population, improved):
        """Add replaced parents to archive, trim if needed."""
        replaced = population[improved]
        for row in replaced:
            self.archive.append(row.copy())
        
        while len(self.archive) > self.archive_max:
            idx = np.random.randint(0, len(self.archive))
            self.archive.pop(idx)

    def _update_memory(self, F, CR, fitness, trial_fitness, improved):
        """Update SHADE parameter memory with successful F and CR values."""
        if not np.any(improved):
            return
        
        s_F = F[improved]
        s_CR = CR[improved]
        
        # Weight by fitness improvement
        deltas = np.abs(fitness[improved] - trial_fitness[improved])
        if np.sum(deltas) == 0:
            weights = np.ones(len(deltas)) / len(deltas)
        else:
            weights = deltas / np.sum(deltas)
        
        # Lehmer mean for F
        numerator = np.sum(weights * s_F ** 2)
        denominator = np.sum(weights * s_F)
        if denominator > 0:
            new_F = numerator / denominator
        else:
            new_F = self.memory_F[self.memory_idx]
        
        # Weighted mean for CR
        new_CR = np.sum(weights * s_CR)
        
        self.memory_F[self.memory_idx] = new_F
        self.memory_CR[self.memory_idx] = new_CR
        self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def _detect_stagnation(self, best_fitness):
        """Detect if the optimization has stagnated."""
        self.best_fitness_history.append(best_fitness)
        
        if len(self.best_fitness_history) < self.stagnation_threshold:
            return False
        
        recent = self.best_fitness_history[-self.stagnation_threshold:]
        if len(recent) < 2:
            return False
        
        improvement = abs(recent[0] - recent[-1])
        scale = max(abs(recent[-1]), 1e-12)
        relative_improvement = improvement / scale
        
        if relative_improvement < 1e-12:
            self.stagnation_counter += 1
            return True
        
        self.stagnation_counter = 0
        return False

    def _restart(self, population, fitness):
        """
        Partial restart: keep best individuals, reinitialize rest.
        Also reset parameter memory partially.
        """
        NP = len(fitness)
        n_keep = max(2, NP // 5)
        
        sorted_idx = np.argsort(fitness)
        best_pop = population[sorted_idx[:n_keep]].copy()
        best_fit = fitness[sorted_idx[:n_keep]].copy()
        
        # Reinitialize the rest with opposition-based learning or random
        n_new = NP - n_keep
        new_pop = np.random.uniform(self.lb, self.ub, (n_new, self.dim))
        
        # Some new individuals near best with perturbation
        n_near = min(n_new // 2, n_new)
        if n_near > 0:
            best_x = best_pop[0]
            scale = max(10.0 / (1 + self.restart_count), 1.0)
            perturbation = np.random.normal(0, scale, (n_near, self.dim))
            new_pop[:n_near] = best_x + perturbation
        
        new_pop = self._clip_to_bounds(new_pop)
        
        population = np.vstack([best_pop, new_pop])
        fitness = np.concatenate([best_fit, np.full(n_new, np.inf)])
        
        # Partially reset memory
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        self.archive = []
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.restart_count += 1
        
        return population, fitness

    def _compute_diversity(self, population):
        """Compute population diversity as mean std across dimensions."""
        return np.mean(np.std(population, axis=0))

    def _opposition_based_init(self, population):
        """Generate opposition-based candidates for diversity."""
        opp = self.lb + self.ub - population
        return self._clip_to_bounds(opp)

    def _local_search_batch(self, population, fitness, best_idx, n_local=5):
        """
        Small batch local search around the best individual to refine solutions.
        Returns candidate batch for evaluation.
        """
        best_x = population[best_idx]
        diversity = self._compute_diversity(population)
        sigma = max(diversity * 0.1, 0.01)
        
        perturbations = np.random.normal(0, sigma, (n_local, self.dim))
        candidates = best_x + perturbations
        candidates = self._clip_to_bounds(candidates)
        return candidates

    def _handle_evaluation_result(self, trial_fitness, expected_size):
        """Handle cases where func returns fewer results than expected."""
        if trial_fitness is None:
            return np.full(expected_size, np.inf), False
        
        actual = len(trial_fitness)
        if actual < expected_size:
            padded = np.full(expected_size, np.inf)
            padded[:actual] = trial_fitness[:actual]
            # Replace NaN with inf
            padded[np.isnan(padded)] = np.inf
            return padded, True
        
        result = trial_fitness.copy()
        result[np.isnan(result)] = np.inf
        return result, False

    def _adapt_population_size(self, generation, max_gen_estimate):
        """
        Linear population size reduction (L-SHADE style).
        Returns target NP for next generation.
        """
        ratio = generation / max(max_gen_estimate, 1)
        ratio = min(ratio, 1.0)
        target_np = int(self.np_size - (self.np_size - self.min_np) * ratio)
        return max(target_np, self.min_np)

    def _reduce_population(self, population, fitness, target_np):
        """Reduce population to target size, keeping the best."""
        if len(fitness) <= target_np:
            return population, fitness
        
        sorted_idx = np.argsort(fitness)
        keep = sorted_idx[:target_np]
        return population[keep].copy(), fitness[keep].copy()

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population = self._initialize_population()
        fitness = func(population)
        fitness, truncated = self._handle_evaluation_result(fitness, self.np_size)
        
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return fitness[best_idx], population[best_idx].copy()
        
        # Also evaluate opposition
        opp_pop = self._opposition_based_init(population)
        opp_fitness = func(opp_pop)
        opp_fitness, _ = self._handle_evaluation_result(opp_fitness, self.np_size)
        
        if stopping_condition():
            all_pop = np.vstack([population, opp_pop])
            all_fit = np.concatenate([fitness, opp_fitness])
            best_idx = np.argmin(all_fit)
            return all_fit[best_idx], all_pop[best_idx].copy()
        
        # Merge and keep best NP
        combined_pop = np.vstack([population, opp_pop])
        combined_fit = np.concatenate([fitness, opp_fitness])
        sorted_idx = np.argsort(combined_fit)
        population = combined_pop[sorted_idx[:self.np_size]].copy()
        fitness = combined_fit[sorted_idx[:self.np_size]].copy()
        
        # Track global best
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()
        
        generation = 0
        max_gen_estimate = 1000  # rough estimate, adjusted dynamically
        
        while not stopping_condition():
            NP = len(population)
            generation += 1
            
            # Generate adaptive parameters
            F, CR = self._generate_parameters_batch(NP)
            
            # Mutation
            mutants = self._mutate_batch(population, fitness, F)
            mutants = self._clip_to_bounds(mutants)
            
            # Crossover
            trials = self._crossover_batch(population, mutants, CR)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trials
            trial_fitness = func(trials)
            trial_fitness, truncated = self._handle_evaluation_result(trial_fitness, NP)
            
            if stopping_condition():
                # Check if any trial is better
                for i in range(NP):
                    if np.isfinite(trial_fitness[i]) and trial_fitness[i] < f_opt:
                        f_opt = trial_fitness[i]
                        x_opt = trials[i].copy()
                break
            
            # Archive old population members that will be replaced
            self._update_archive(population, np.zeros(NP, dtype=bool))
            
            # Selection
            new_pop, new_fit, improved = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update archive with replaced parents
            self._update_archive(population, improved)
            
            # Update memory
            self._update_memory(F, CR, fitness, trial_fitness, improved)
            
            population = new_pop
            fitness = new_fit
            
            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
            # Local search every few generations
            if generation % 10 == 0 and not stopping_condition():
                n_local = min(8, max(4, NP // 10))
                local_candidates = self._local_search_batch(
                    population, fitness, current_best_idx, n_local
                )
                local_fitness = func(local_candidates)
                local_fitness, _ = self._handle_evaluation_result(local_fitness, n_local)
                
                if stopping_condition():
                    for i in range(len(local_fitness)):
                        if np.isfinite(local_fitness[i]) and local_fitness[i] < f_opt:
                            f_opt = local_fitness[i]
                            x_opt = local_candidates[i].copy()
                    break
                
                # Replace worst in population if local search found better
                for i in range(len(local_fitness)):
                    if np.isfinite(local_fitness[i]):
                        worst_idx = np.argmax(fitness)
                        if local_fitness[i] < fitness[worst_idx]:
                            population[worst_idx] = local_candidates[i].copy()
                            fitness[worst_idx] = local_fitness[i]
                            if local_fitness[i] < f_opt:
                                f_opt = local_fitness[i]
                                x_opt = local_candidates[i].copy()
            
            # Stagnation detection and restart
            stagnated = self._detect_stagnation(f_opt)
            if stagnated and self.stagnation_counter >= 2:
                population, fitness = self._restart(population, fitness)
                
                # Evaluate newly initialized individuals
                needs_eval = np.isinf(fitness)
                if np.any(needs_eval):
                    eval_pop = population[needs_eval]
                    eval_fit = func(eval_pop)
                    eval_fit, _ = self._handle_evaluation_result(eval_fit, len(eval_pop))
                    
                    if stopping_condition():
                        for i in range(len(eval_fit)):
                            if np.isfinite(eval_fit[i]) and eval_fit[i] < f_opt:
                                f_opt = eval_fit[i]
                                x_opt = eval_pop[i].copy()
                        break
                    
                    fitness[needs_eval] = eval_fit
                    
                    current_best_idx = np.argmin(fitness)
                    if fitness[current_best_idx] < f_opt:
                        f_opt = fitness[current_best_idx]
                        x_opt = population[current_best_idx].copy()
            
            # Optional population size reduction (L-SHADE style, gentle)
            if generation % 50 == 0 and generation > 100:
                target_np = self._adapt_population_size(generation, max_gen_estimate)
                target_np = max(target_np, self.min_np)
                if len(population) > target_np:
                    population, fitness = self._reduce_population(
                        population, fitness, target_np
                    )
                    self.archive_max = len(population)
        
        return f_opt, x_opt
