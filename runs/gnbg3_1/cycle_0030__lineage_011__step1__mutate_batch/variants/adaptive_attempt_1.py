import numpy as np


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Success-Based Operator Pool.
    
    Key features:
    - Pool of DE mutation strategies selected via Thompson Sampling (from benchmark winners)
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5*dim (balance between exploration and efficiency)
        self.np = min(max(5 * dim, 20), 200)
        
        # Mutation strategies pool (from benchmark winners)
        self.mutation_strategies = ['variant_01', 'variant_04', 'variant_07', 'variant_09', 'variant_10']
        self.n_strategies = len(self.mutation_strategies)
        
        # Thompson Sampling parameters
        self.operator_rewards = np.zeros(self.n_strategies)
        self.operator_counts = np.ones(self.n_strategies)
        self.reward_history = [list() for _ in range(self.n_strategies)]
        self.reward_window = 20
        
        # DE parameters
        self.cr = 0.9
        self.f_base = 0.5
        self.f_adapt = 0.1
        
        # Local search parameters
        self.local_search_interval = 5
        self.local_search_budget = 0.02
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE variants
        self.archive = None
        self.archive_max_size = self.np
        
        # Per-variant state tracking
        self.variant_04_archive = None
        self.variant_10_archive = None
        self.variant_10_scale_mode = 0
    
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.np, self.dim))
        
        grid = np.linspace(0, 1, self.np + 1)[:-1] + np.random.uniform(0, 1/self.np, self.np)
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.lower_bound + (self.upper_bound - self.lower_bound) * grid[perm]
        
        fitness = func(pop)
        fitness = self._handle_budget_truncation(fitness, pop)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Select mutation operator using Thompson Sampling with sliding window."""
        # Compute posterior mean reward for each operator
        mean_rewards = np.zeros(self.n_strategies)
        for i in range(self.n_strategies):
            if len(self.reward_history[i]) > 0:
                mean_rewards[i] = np.mean(self.reward_history[i])
            else:
                mean_rewards[i] = 0.0
        
        # Thompson Sampling: sample from posterior
        # Use Beta distribution approximation for rewards in [0, 1]
        alpha = np.maximum(mean_rewards * 10 + 1, 1.0)
        beta = np.maximum((1 - mean_rewards) * 10 + 1, 1.0)
        
        samples = np.random.beta(alpha, beta)
        
        # Add small random tie-breaker
        samples = samples + np.random.uniform(0, 0.01, size=self.n_strategies)
        
        selected = int(np.argmax(samples))
        return selected
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using selected DE mutation strategy from benchmark winners."""
        np_pop = len(population)
        dim = self.dim
        
        mutants = np.empty_like(population)
        
        if selected_operator == 0:  # variant_01: current-to-rand/1 with archive
            all_indices = np.arange(np_pop)
            r = np.random.randint(0, np_pop, size=(np_pop, 4))
            for i in range(np_pop):
                while len(np.unique(r[i])) < 4:
                    r[i] = np.random.randint(0, np_pop, size=4)
            r1, r2, r3, r4 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
            
            if self.archive is None:
                self.archive = population.copy()
            else:
                if len(self.archive) > self.archive_max_size:
                    self.archive = self.archive[np.random.choice(len(self.archive), self.archive_max_size, replace=False)]
                self.archive = np.vstack([self.archive, population])
                if len(self.archive) > self.archive_max_size * 2:
                    self.archive = self.archive[np.random.choice(len(self.archive), self.archive_max_size, replace=False)]
            
            archive_size = len(self.archive)
            if archive_size > np_pop:
                r_archive = np.random.randint(0, archive_size, size=np_pop)
                archive_donors = self.archive[r_archive]
            else:
                archive_donors = population
            
            sorted_idx = np.argsort(fitness)
            p = 0.1 + 0.1 * 0
            pbest_count = max(1, int(np_pop * p))
            pbest_idx = sorted_idx[:pbest_count]
            
            f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
            f_values = np.clip(f_values, 0.1, 2.0)
            
            mutants = (population + f_values[:, np.newaxis] * 
                      (population[r1] - population + population[r2] - population[r3]))
            
        elif selected_operator == 1:  # variant_04: covariance-adaptive with archive
            if self.variant_04_archive is None:
                self.variant_04_archive = population[np.argsort(fitness)[:min(10, np_pop)]].copy()
                self._v04_archive_fitness = np.sort(fitness)[:min(10, np_pop)]
            
            current_best_idx = np.argmin(fitness)
            n_archive = len(self.variant_04_archive)
            if fitness[current_best_idx] < self._v04_archive_fitness[0]:
                if n_archive < 50:
                    self.variant_04_archive = np.vstack([self.variant_04_archive, population[current_best_idx]])
                    self._v04_archive_fitness = np.append(self._v04_archive_fitness, fitness[current_best_idx])
                else:
                    worst_arch_idx = np.argmax(self._v04_archive_fitness)
                    self.variant_04_archive[worst_arch_idx] = population[current_best_idx]
                    self._v04_archive_fitness[worst_arch_idx] = fitness[current_best_idx]
            
            mean_pop = np.mean(population, axis=0)
            centered = population - mean_pop
            cov_matrix = np.cov(centered.T)
            cov_matrix += np.eye(dim) * 1e-8
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
            except np.linalg.LinAlgError:
                eigenvalues = np.ones(dim)
                eigenvectors = np.eye(dim)
            
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-12)
            sqrt_eigen = np.sqrt(eigenvalues_norm)
            
            r1 = np.random.randint(0, np_pop, size=np_pop)
            r2 = np.random.randint(0, n_archive, size=np_pop)
            
            f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
            f_values = np.clip(f_values, 0.1, 2.0)
            
            base_pop = population[r1]
            target_archive = self.variant_04_archive[r2]
            
            diff = target_archive - base_pop
            scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
            
            noise = np.random.randn(np_pop, dim)
            noise = noise @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
            noise *= 0.1 * f_values[:, np.newaxis]
            
            mutants = base_pop + f_values[:, np.newaxis] * scaled_diff + noise
            
        elif selected_operator == 2:  # variant_07: multi-strategy ensemble
            indices = np.arange(np_pop)
            ring_prev = np.roll(indices, 1)
            ring_next = np.roll(indices, -1)
            
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]
            
            f_min, f_max = np.min(fitness), np.max(fitness)
            f_range = max(f_max - f_min, 1e-10)
            f_variance = np.var(fitness)
            best_worst_ratio = (f_min + 1e-10) / (f_max + 1e-10)
            
            valid_mask = np.ones((np_pop, np_pop), dtype=bool)
            np.fill_diagonal(valid_mask, False)
            
            r1 = np.array([np.random.choice(np.where(valid_mask[i])[0]) for i in range(np_pop)])
            r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) for i in range(np_pop)])
            r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) for i in range(np_pop)])
            
            base_f = self.f_base * (1.0 + 0.5 * np.log1p(f_variance) / (np_pop + 1))
            f_values = base_f + self.f_adapt * np.random.randn(np_pop)
            f_values = np.clip(f_values, 0.05, 2.5)
            
            mutant_rand = population[r1] + f_values[:, np.newaxis] * (population[r2] - population[r3])
            
            pbest = population[np.random.choice(pbest_idx, np_pop)]
            mutant_gradient = population + f_values[:, np.newaxis] * (pbest - population[r1] + population[r2] - population[r3])
            
            best_idx = sorted_idx[0]
            mutant_oppose = population[r1] + f_values[:, np.newaxis] * (population[best_idx] - population[r2] + population[r3] - population[r1])
            
            elite = population[sorted_idx[:max(1, int(np_pop * 0.05))]]
            elite_choice = elite[np.random.randint(0, len(elite), size=np_pop)]
            mutant_elite = elite_choice + f_values[:, np.newaxis] * (population[r1] - population[r2])
            
            if best_worst_ratio < 0.1:
                probs = np.array([0.35, 0.25, 0.25, 0.15])
            elif best_worst_ratio > 0.5:
                probs = np.array([0.2, 0.35, 0.15, 0.3])
            else:
                probs = np.array([0.3, 0.3, 0.2, 0.2])
            
            probs = probs / probs.sum()
            strategy_choices = np.random.choice(4, size=np_pop, p=probs)
            
            for i in range(np_pop):
                if strategy_choices[i] == 0:
                    mutants[i] = mutant_rand[i]
                elif strategy_choices[i] == 1:
                    mutants[i] = mutant_gradient[i]
                elif strategy_choices[i] == 2:
                    mutants[i] = mutant_oppose[i]
                else:
                    mutants[i] = mutant_elite[i]
            
        elif selected_operator == 3:  # variant_09: ring-guided adaptive mutation
            indices = np.arange(np_pop)
            ring_prev = np.roll(indices, 1)
            ring_next = np.roll(indices, -1)
            
            r1 = np.array([np.random.choice(np.delete(np.arange(np_pop), i)) for i in range(np_pop)])
            r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) for i in range(np_pop)])
            r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) for i in range(np_pop)])
            
            f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
            f_values = np.clip(f_values, 0.3, 1.5)
            
            for i in range(np_pop):
                f_curr = float(fitness[i])
                f_prev = float(fitness[ring_prev[i]])
                f_next = float(fitness[ring_next[i]])
                
                if f_prev <= f_next and f_prev < f_curr:
                    ring_dir = population[ring_prev[i]] - population[i]
                    ring_weight = 0.5
                elif f_next < f_prev and f_next < f_curr:
                    ring_dir = population[ring_next[i]] - population[i]
                    ring_weight = 0.5
                else:
                    ring_dir = population[ring_next[i]] - population[ring_prev[i]]
                    ring_weight = 0.3
                
                if 0 == 0:
                    base = population[r1[i]]
                    de_vec = population[r2[i]] - population[r3[i]]
                    mutants[i] = base + f_values[i] * de_vec + ring_weight * ring_dir
                elif 1 == 0:
                    base = population[i]
                    de_vec = population[r1[i]] - population[r2[i]]
                    if f_prev <= f_next:
                        target = population[ring_prev[i]]
                    else:
                        target = population[ring_next[i]]
                    mutants[i] = base + f_values[i] * de_vec + 0.7 * (target - base) + 0.3 * ring_dir
                elif 2 == 0:
                    sorted_idx = np.argsort(fitness)
                    pbest_idx = sorted_idx[0]
                    base = population[i]
                    pbest_vec = population[pbest_idx] - population[i]
                    de_vec = population[r1[i]] - population[r2[i]]
                    mutants[i] = base + f_values[i] * de_vec + 0.5 * pbest_vec + 0.2 * ring_dir
                else:
                    best_idx = np.argmin(fitness)
                    base = population[r1[i]]
                    best_vec = population[best_idx] - population[r1[i]]
                    de_vec = population[r2[i]] - population[r3[i]]
                    mutants[i] = base + f_values[i] * de_vec + 0.4 * best_vec + 0.3 * ring_dir
            
        else:  # variant_10: archive-guided multi-scale mutation
            sorted_idx = np.argsort(fitness)
            
            if self.variant_10_archive is None:
                self.variant_10_archive = population[sorted_idx[:min(10, np_pop)]].copy()
            else:
                combined = np.vstack([self.variant_10_archive, population[sorted_idx[:5]]])
                combined_fitness = np.array(list(np.zeros(len(self.variant_10_archive)) * 0.1) + list(fitness[sorted_idx[:5]]))
                top_indices = np.argsort(combined_fitness)[:min(15, len(combined))]
                self.variant_10_archive = combined[top_indices]
            
            r1 = np.array([np.random.choice(np.delete(np.arange(np_pop), i)) for i in range(np_pop)])
            r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) for i in range(np_pop)])
            
            archive_size = len(self.variant_10_archive)
            arch_idx = np.random.randint(0, archive_size, size=np_pop)
            
            self.variant_10_scale_mode = (self.variant_10_scale_mode + 1) % 3
            if self.variant_10_scale_mode == 0:
                f_values = self.f_base * np.ones(np_pop) * 0.4
            elif self.variant_10_scale_mode == 1:
                f_values = self.f_base * np.ones(np_pop) * 0.8
            else:
                f_values = self.f_base * np.ones(np_pop) * 1.5
            
            f_values = f_values + 0.1 * np.random.randn(np_pop)
            f_values = np.clip(f_values, 0.1, 2.0)
            
            if 0 == 0:
                mutants = (self.variant_10_archive[arch_idx] + f_values[:, np.newaxis] * 
                          (population[r1] - population[r2]))
            elif 1 == 0:
                best_idx = np.argmin(fitness)
                mutants = (self.variant_10_archive[arch_idx] + f_values[:, np.newaxis] * 
                          (population[best_idx] - self.variant_10_archive[arch_idx] + 
                           population[r1] - population[r2]))
            elif 2 == 0:
                arch_pbest = self.variant_10_archive[arch_idx]
                mutants = (population + f_values[:, np.newaxis] * 
                          (arch_pbest - population[r1] + population[r1] - population[r2]))
            else:
                arch_idx2 = np.random.randint(0, archive_size, size=np_pop)
                mutants = (self.variant_10_archive[arch_idx] + f_values[:, np.newaxis] * 
                          (population[r1] - population[r2] + 
                           population[r2] - self.variant_10_archive[arch_idx2]))
        
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        
        nan_mask = np.any(np.isnan(mutants) | np.isinf(mutants), axis=1)
        if np.any(nan_mask):
            mutants[nan_mask] = population[nan_mask] + np.random.uniform(-0.1, 0.1, (np.sum(nan_mask), self.dim))
            mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        np_pop = len(population)
        
        cr_mask = np.random.rand(np_pop, self.dim) < self.cr
        
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        cr_mask[np.arange(np_pop), j_rand] = True
        
        trials = np.where(cr_mask, mutants, population)
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Selection: greedy replacement based on fitness."""
        better_mask = trial_fitness < fitness
        
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness, better_mask
    
    def _apply_local_refinement(self, population, fitness, func):
        """Apply Nelder-Mead-like local search to top individuals."""
        if len(population) < self.dim + 1:
            return population, fitness
        
        n_elite = min(3, len(population))
        
        for i in range(n_elite):
            x_best = population[i].copy()
            f_best = float(fitness[i])
            
            simplex = np.tile(x_best, (self.dim + 1, 1))
            simplex[0] = x_best
            
            step_size = 0.5
            for j in range(1, self.dim + 1):
                simplex[j] = x_best.copy()
                simplex[j, (j-1) % self.dim] += step_size * (self.upper_bound - self.lower_bound)
                simplex[j] = np.clip(simplex[j], self.lower_bound, self.upper_bound)
            
            simplex_fitness = func(simplex)
            simplex_fitness = self._handle_budget_truncation(simplex_fitness, simplex)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            sort_idx = np.argsort(simplex_fitness)
            simplex = simplex[sort_idx]
            simplex_fitness = simplex_fitness[sort_idx]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
            f_r = float(func(x_r.reshape(1, -1))[0])
            
            if f_r < simplex_fitness[0]:
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                f_e = float(func(x_e.reshape(1, -1))[0])
                
                if f_e < f_r:
                    new_point, new_f = x_e, f_e
                else:
                    new_point, new_f = x_r, f_r
            elif f_r < simplex_fitness[-2]:
                new_point, new_f = x_r, f_r
            else:
                new_point = simplex[0] + rho * (simplex[-1] - simplex[0])
                new_point = np.clip(new_point, self.lower_bound, self.upper_bound)
                new_f = float(func(new_point.reshape(1, -1))[0])
            
            if new_f < f_best:
                population[i] = new_point
                fitness[i] = new_f
        
        return population, fitness
    
    def _update_operator_rewards(self, selected_operator, success_mask):
        """Update operator rewards using sliding window credit assignment."""
        success_rate = float(np.mean(success_mask))
        
        self.reward_history[selected_operator].append(success_rate)
        if len(self.reward_history[selected_operator]) > self.reward_window:
            self.reward_history[selected_operator].pop(0)
        
        total_reward = float(np.sum(self.reward_history[selected_operator]))
        self.operator_rewards[selected_operator] = total_reward
        self.operator_counts[selected_operator] = float(len(self.reward_history[selected_operator]))
    
    def _adapt_step_size(self, success_rate):
        """Adapt F and CR based on recent success."""
        if success_rate > 0.2:
            self.f_adapt = min(0.5, self.f_adapt * 1.1)
        elif success_rate < 0.05:
            self.f_adapt = max(0.01, self.f_adapt * 0.9)
        
        if success_rate > 0.3:
            self.cr = min(0.95, self.cr * 1.01)
        elif success_rate < 0.1:
            self.cr = max(0.5, self.cr * 0.99)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        avg_distance = float(np.mean(distances[mask]))
        
        return avg_distance
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = float(np.min(fitness))
        self.best_history.append(current_best)
        
        if len(self.best_history) > self.stagnation_limit:
            self.best_history.pop(0)
        
        if len(self.best_history) >= self.stagnation_limit:
            improvement = self.best_history[-1] - self.best_history[0]
            if abs(improvement) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            return self.stagnation_counter >= 2
        
        return False
    
    def _restart_if_needed(self, population, fitness, func, force=False):
        """Restart population if stagnation or diversity loss detected."""
        diversity = self._compute_diversity(population)
        
        if force or diversity < self.diversity_threshold or self.stagnation_counter >= 2:
            n_reinit = int(0.7 * self.np)
            reinit_idx = np.argsort(fitness)[-n_reinit:]
            
            for d in range(self.dim):
                grid = np.linspace(0, 1, n_reinit + 1)[:-1] + np.random.uniform(0, 1/n_reinit, n_reinit)
                perm = np.random.permutation(n_reinit)
                population[reinit_idx, d] = (self.lower_bound + 
                    (self.upper_bound - self.lower_bound) * grid[perm])
            
            new_fitness = func(population[reinit_idx])
            if len(new_fitness) < len(reinit_idx):
                new_fitness = np.pad(new_fitness, (0, len(reinit_idx) - len(new_fitness)), 
                                     constant_values=np.inf)
            fitness[reinit_idx] = new_fitness
            
            self.stagnation_counter = 0
            self.best_history = []
            
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Ensure all individuals are within bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def __call__(self, func, stopping_condition):
        """Run the optimizer."""
        population, fitness = self._initialize_population(func)
        
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return float(fitness[best_idx]), population[best_idx].copy()
        
        generation = 0
        local_search_counter = 0
        
        while not stopping_condition():
            selected_operator = self._select_mutation_operator_batch()
            
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            trials = self._crossover_batch(population, mutants)
            
            trials = self._clip_to_bounds(trials)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                population = population[:valid_trials]
                fitness = fitness[:valid_trials]
                self.np = valid_trials
                if self.np < 10:
                    break
            
            if stopping_condition():
                break
            
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            self._update_operator_rewards(selected_operator, success_mask)
            
            success_rate = float(np.mean(success_mask))
            self._adapt_step_size(success_rate)
            
            local_search_counter += 1
            if local_search_counter >= self.local_search_interval:
                population, fitness = self._apply_local_refinement(population, fitness, func)
                local_search_counter = 0
                
                if stopping_condition():
                    break
            
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            if generation % 10 == 0:
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            generation += 1
        
        best_idx = np.argmin(fitness)
        return float(fitness[best_idx]), population[best_idx].copy()
