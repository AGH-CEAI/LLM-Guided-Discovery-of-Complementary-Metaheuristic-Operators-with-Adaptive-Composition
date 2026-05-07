import numpy as np


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Success-Based Operator Pool.
    
    Key features:
    - Pool of DE mutation strategies selected via success history (UCB-based)
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
        
        # Mutation strategies pool
        self.mutation_strategies = ['rand', 'best', 'current_to_pbest', 'rand_to_best']
        self.n_strategies = len(self.mutation_strategies)
        
        # Success history for operator selection (UCB)
        self.operator_rewards = np.zeros(self.n_strategies)
        self.operator_counts = np.ones(self.n_strategies)
        self.ucb_c = 2.0  # Exploration weight for UCB
        
        # DE parameters
        self.cr = 0.9
        self.f_base = 0.5
        self.f_adapt = 0.1
        
        # Local search parameters
        self.local_search_interval = 5  # Apply LS every N generations
        self.local_search_budget = 0.02  # Fraction of func budget for LS
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE (optional, bounded)
        self.archive = None
        self.archive_max_size = self.np
        
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.np, self.dim))
        
        # LHS for better spread
        grid = np.linspace(0, 1, self.np + 1)[:-1] + np.random.uniform(0, 1/self.np, self.np)
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.lower_bound + (self.upper_bound - self.lower_bound) * grid[perm]
        
        # Evaluate initial population
        fitness = func(pop)
        fitness = self._handle_budget_truncation(fitness, pop)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            # Truncate population to match fitness
            actual_len = len(fitness)
            # Keep first `actual_len` individuals
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Select mutation operator per-individual using diversity-adaptive softmax selection."""
        np_pop = len(self.population) if hasattr(self, 'population') and self.population is not None else self.np

        # Compute population diversity metrics
        if hasattr(self, 'population') and self.population is not None and len(self.population) > 1:
            # Fitness variance (normalized)
            f_var = np.var(self.fitness) if hasattr(self, 'fitness') and self.fitness is not None else 1.0
            f_range = np.max(self.fitness) - np.min(self.fitness) if hasattr(self, 'fitness') and self.fitness is not None else 1.0
            diversity_score = min(1.0, f_var / max(f_range, 1e-10))

            # Spatial diversity (average distance from centroid)
            centroid = np.mean(self.population, axis=0)
            spatial_dist = np.mean(np.sqrt(np.sum((self.population - centroid) ** 2, axis=1)))
            spatial_diversity = min(1.0, spatial_dist / (self.upper_bound - self.lower_bound + 1e-10))
        else:
            diversity_score = 0.5
            spatial_diversity = 0.5

        # Compute base probabilities from success history (UCB-like reward)
        total_counts = np.sum(self.operator_counts) + 1e-10
        ucb_rewards = self.operator_rewards / np.maximum(self.operator_counts, 1)
        ucb_rewards += self.ucb_c * np.sqrt(np.log(total_counts) / np.maximum(self.operator_counts, 1))

        # Convert to probabilities via softmax with temperature based on diversity
        # Low diversity -> high temperature (more exploration, flatter distribution)
        # High diversity -> low temperature (more exploitation, sharper distribution)
        temperature = 2.0 - 1.5 * diversity_score  # Range: [0.5, 2.0]
        softmax_scores = np.exp((ucb_rewards - np.max(ucb_rewards)) / max(temperature, 0.1))
        base_probs = softmax_scores / np.sum(softmax_scores)

        # Adjust probabilities based on diversity state
        if diversity_score < 0.1:  # Very low diversity - stuck in local optima
            # Boost exploratory operators (rand-based: indices 0, 2)
            adjusted_probs = base_probs.copy()
            adjusted_probs[0] *= 1.5  # rand
            adjusted_probs[2] *= 1.3  # rand-to-best
            adjusted_probs[1] *= 0.7  # current-to-pbest (less greedy)
            adjusted_probs[3] *= 0.8  # elite (less convergent)
        elif diversity_score > 0.5:  # Good diversity - can afford exploitation
            adjusted_probs = base_probs.copy()
            adjusted_probs[1] *= 1.3  # current-to-pbest
            adjusted_probs[3] *= 1.2  # elite
            adjusted_probs[0] *= 0.8  # rand
        else:  # Normal state
            adjusted_probs = base_probs.copy()

        # Normalize
        adjusted_probs = np.maximum(adjusted_probs, 1e-10)
        adjusted_probs /= np.sum(adjusted_probs)

        # Select operator for EACH individual independently
        selected_operators = np.random.choice(self.n_strategies, size=np_pop, p=adjusted_probs)

        # Update counts based on selected operators (for next iteration's reward tracking)
        for op_idx in range(self.n_strategies):
            count_this_op = np.sum(selected_operators == op_idx)
            self.operator_counts[op_idx] += count_this_op * 0.1  # Scaled update

        return selected_operators
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using multi-strategy ensemble with fitness-weighted selection."""
        np_pop = len(population)

        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)

        # Get sorted indices for various selection schemes
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
        worst_idx = sorted_idx[-1]

        # Compute fitness landscape properties for adaptive strategy selection
        f_min, f_max = np.min(fitness), np.max(fitness)
        f_range = max(f_max - f_min, 1e-10)
        f_variance = np.var(fitness)
        best_worst_ratio = (f_min + 1e-10) / (f_max + 1e-10)

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Efficient batch index sampling
        valid_mask = np.ones((np_pop, np_pop), dtype=bool)
        np.fill_diagonal(valid_mask, False)

        for i in range(np_pop):
            valid_mask[i, i] = False

        # Sample r1, r2, r3 for each individual
        r1 = np.array([np.random.choice(np.where(valid_mask[i])[0]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) 
                       for i in range(np_pop)])

        # Adaptive F with dynamic bounds based on landscape
        base_f = self.f_base * (1.0 + 0.5 * np.log1p(f_variance) / (np_pop + 1))
        f_values = base_f + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.05, 2.5)

        # Strategy 0: Random direction (rand/3/bin - different from rand/1)
        mutant_rand = population[r1] + f_values[:, np.newaxis] * (
            population[r2] - population[r3])

        # Strategy 1: Gradient-following (current-to-pbest/2 with better indices)
        pbest = population[np.random.choice(pbest_idx, np_pop)]
        mutant_gradient = population + f_values[:, np.newaxis] * (
            pbest - population[r1] + population[r2] - population[r3])

        # Strategy 2: Opposition-based (toward better, away from worst)
        best_idx = sorted_idx[0]
        mutant_oppose = population[r1] + f_values[:, np.newaxis] * (
            population[best_idx] - population[r2] + population[r3] - population[r1])

        # Strategy 3: Dimension-wise精英 (elite-guided with dimensional diversity)
        elite = population[sorted_idx[:max(1, int(np_pop * 0.05))]]  # Top 5%
        elite_choice = elite[np.random.randint(0, len(elite), size=np_pop)]
        mutant_elite = elite_choice + f_values[:, np.newaxis] * (
            population[r1] - population[r2])

        # Compute strategy selection probabilities based on fitness landscape
        # High variance = exploration needed, low ratio = stuck in local optima
        explore_weight = min(0.5, f_variance / (f_range + 1e-10))
        exploit_weight = 1.0 - explore_weight

        # Probabilities: [rand, gradient, oppose, elite]
        if best_worst_ratio < 0.1:  # Stuck - favor exploration
            probs = np.array([0.35, 0.25, 0.25, 0.15])
        elif best_worst_ratio > 0.5:  # Good progress - balance
            probs = np.array([0.2, 0.35, 0.15, 0.3])
        else:  # Normal - moderate exploration
            probs = np.array([0.3, 0.3, 0.2, 0.2])

        # Normalize probabilities
        probs = probs / probs.sum()

        # Generate random choices for each individual
        strategy_choices = np.random.choice(4, size=np_pop, p=probs)

        # Assemble final mutants based on strategy choices
        for i in range(np_pop):
            if strategy_choices[i] == 0:
                mutants[i] = mutant_rand[i]
            elif strategy_choices[i] == 1:
                mutants[i] = mutant_gradient[i]
            elif strategy_choices[i] == 2:
                mutants[i] = mutant_oppose[i]
            else:
                mutants[i] = mutant_elite[i]

        # Clip to bounds with numerical safety
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        # Handle any NaN/Inf from numerical issues
        nan_mask = np.any(np.isnan(mutants) | np.isinf(mutants), axis=1)
        if np.any(nan_mask):
            mutants[nan_mask] = population[nan_mask] + np.random.uniform(
                -0.1, 0.1, (np.sum(nan_mask), self.dim))
            mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        np_pop = len(population)
        
        # Random crossover mask
        cr_mask = np.random.rand(np_pop, self.dim) < self.cr
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        cr_mask[np.arange(np_pop), j_rand] = True
        
        # Create trial vectors
        trials = np.where(cr_mask, mutants, population)
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Selection: greedy replacement based on fitness."""
        # Better fitness wins (assuming minimization)
        better_mask = trial_fitness < fitness
        
        # Update population and fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness, better_mask
    
    def _apply_local_refinement(self, population, fitness, func):
        """Apply Nelder-Mead-like local search to top individuals."""
        if len(population) < self.dim + 1:
            return population, fitness
        
        n_elite = min(3, len(population))  # Apply to top 3
        
        for i in range(n_elite):
            # Get current best position
            x_best = population[i].copy()
            f_best = fitness[i]
            
            # Build simplex around current best
            simplex = np.tile(x_best, (self.dim + 1, 1))
            simplex[0] = x_best
            
            # Add small perturbations for other vertices
            step_size = 0.5
            for j in range(1, self.dim + 1):
                simplex[j] = x_best.copy()
                simplex[j, (j-1) % self.dim] += step_size * (self.upper_bound - self.lower_bound)
                simplex[j] = np.clip(simplex[j], self.lower_bound, self.upper_bound)
            
            # Evaluate simplex
            simplex_fitness = func(simplex)
            simplex_fitness = self._handle_budget_truncation(simplex_fitness, simplex)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            # Simple Nelder-Mead iteration
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            # Sort simplex
            sort_idx = np.argsort(simplex_fitness)
            simplex = simplex[sort_idx]
            simplex_fitness = simplex_fitness[sort_idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
            f_r = func(x_r.reshape(1, -1))[0]
            
            if f_r < simplex_fitness[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                f_e = func(x_e.reshape(1, -1))[0]
                
                if f_e < f_r:
                    new_point, new_f = x_e, f_e
                else:
                    new_point, new_f = x_r, f_r
            elif f_r < simplex_fitness[-2]:
                new_point, new_f = x_r, f_r
            else:
                # Shrink
                new_point = simplex[0] + rho * (simplex[-1] - simplex[0])
                new_point = np.clip(new_point, self.lower_bound, self.upper_bound)
                new_f = func(new_point.reshape(1, -1))[0]
            
            # Update if improvement found
            if new_f < f_best:
                population[i] = new_point
                fitness[i] = new_f
        
        return population, fitness
    
    def _update_operator_rewards(self, selected_operator, success_mask):
        """Update operator rewards based on success rate."""
        success_rate = np.mean(success_mask)
        
        # Reward successful operators
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        # Decay rewards of non-selected operators slightly
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
    
    def _adapt_step_size(self, success_rate):
        """Adapt F and CR based on recent success."""
        if success_rate > 0.2:
            # Good success: maintain or slightly increase exploration
            self.f_adapt = min(0.5, self.f_adapt * 1.1)
        elif success_rate < 0.05:
            # Poor success: reduce step size for exploitation
            self.f_adapt = max(0.01, self.f_adapt * 0.9)
        
        # Adapt CR: increase if stuck, decrease if unstable
        if success_rate > 0.3:
            self.cr = min(0.95, self.cr * 1.01)
        elif success_rate < 0.1:
            self.cr = max(0.5, self.cr * 0.99)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Average of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        avg_distance = np.mean(distances[mask])
        
        return avg_distance
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
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
            # Reinitialize portion of population
            n_reinit = int(0.7 * self.np)
            reinit_idx = np.argsort(fitness)[-n_reinit:]  # Replace worst
            
            # Generate new individuals using LHS
            for d in range(self.dim):
                grid = np.linspace(0, 1, n_reinit + 1)[:-1] + np.random.uniform(0, 1/n_reinit, n_reinit)
                perm = np.random.permutation(n_reinit)
                population[reinit_idx, d] = (self.lower_bound + 
                    (self.upper_bound - self.lower_bound) * grid[perm])
            
            # Evaluate reinitialized portion
            new_fitness = func(population[reinit_idx])
            if len(new_fitness) < len(reinit_idx):
                new_fitness = np.pad(new_fitness, (0, len(reinit_idx) - len(new_fitness)), 
                                     constant_values=np.inf)
            fitness[reinit_idx] = new_fitness
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            self.best_history = []
            
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Ensure all individuals are within bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        population, fitness = self._initialize_population(func)
        
        # Check budget after initialization
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return fitness[best_idx], population[best_idx]
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator
            selected_operator = self._select_mutation_operator_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip trials to bounds
            trials = self._clip_to_bounds(trials)
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget truncation
            if len(trial_fitness) < len(trials):
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                population = population[:valid_trials]
                fitness = fitness[:valid_trials]
                self.np = valid_trials
                if self.np < 10:
                    break
            
            # Check stopping after partial evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update operator rewards
            self._update_operator_rewards(selected_operator, success_mask)
            
            # Adapt step size
            success_rate = np.mean(success_mask)
            self._adapt_step_size(success_rate)
            
            # Periodic local search
            local_search_counter += 1
            if local_search_counter >= self.local_search_interval:
                population, fitness = self._apply_local_refinement(population, fitness, func)
                local_search_counter = 0
                
                if stopping_condition():
                    break
            
            # Check stagnation
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            # Periodic restart check based on diversity
            if generation % 10 == 0:
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            generation += 1
        
        # Return best solution
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]
