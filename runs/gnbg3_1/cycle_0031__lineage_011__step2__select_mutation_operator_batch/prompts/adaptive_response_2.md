```python
import numpy as np


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Meta-Adaptive Operator Selection.
    
    Key features:
    - Pool of DE mutation strategies
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    - Meta-adaptive selection of operator selection strategy (learns from benchmark patterns)
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
        self.local_search_interval = 5
        self.local_search_budget = 0.02
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE (optional, bounded)
        self.archive = None
        self.archive_max_size = self.np
        
        # ========== META-ADAPTIVE OPERATOR SELECTION ==========
        # Track performance of each strategy selector using sliding window
        self.n_selectors = 10  # 9 variants + 1 original
        
        # Meta-bandit state: successes and failures for each selector
        self.selector_successes = np.ones(self.n_selectors, dtype=float)
        self.selector_failures = np.ones(self.n_selectors, dtype=float)
        
        # Sliding window credit assignment
        self.selector_window_size = 20
        self.selector_history = [[] for _ in range(self.n_selectors)]  # List of (reward, count) tuples
        
        # Current best selector based on empirical performance
        self.current_best_selector = 0
        self.selector_call_counts = np.zeros(self.n_selectors)
        
        # Performance tracking for adaptive switching
        self.recent_rewards = []
        self.generation = 0
        
        # Initialize tracking arrays used by some selector variants
        self.operator_diversity = np.ones(self.n_strategies)
        self.operator_momentum = np.zeros(self.n_strategies)
        self.op_success_history = None
        self.op_history_idx = 0
        self.last_selected = 0
        self.last_success_rate = 0.0
        self.call_counter = 0
        self.operator_success_ema = np.zeros(self.n_strategies)
        self.operator_successes = np.ones(self.n_strategies, dtype=float)
        self.operator_failures = np.ones(self.n_strategies, dtype=float)
        self.last_fitness = None
        
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
        self.last_fitness = np.array([float(np.min(fitness))], dtype=float)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Meta-adaptive operator selection: learns best strategy during optimization."""
        # Use Thompson Sampling to select which selector strategy to use
        total_observations = np.sum(self.selector_successes) + np.sum(self.selector_failures)
        
        # Sample from Beta distributions for each selector
        beta_samples = np.array([
            np.random.beta(self.selector_successes[i] + 0.1, self.selector_failures[i] + 0.1)
            for i in range(self.n_selectors)
        ])
        
        # Add exploration bonus for under-explored selectors
        exploration_bonus = self.ucb_c * np.sqrt(np.log(total_observations + 1) / 
                                                  np.maximum(self.selector_call_counts + 1, 1))
        exploration_bonus = np.clip(exploration_bonus, 0, 5.0)
        
        final_scores = beta_samples + exploration_bonus
        
        # Select which selector strategy to use
        selected_selector = int(np.argmax(final_scores))
        self.selector_call_counts[selected_selector] += 1
        
        # Track which selector we're using
        self.current_best_selector = selected_selector
        
        # Call the selected operator selection method
        if selected_selector == 0:
            result = self._selector_original()
        elif selected_selector == 1:
            result = self._selector_variant_01()
        elif selected_selector == 2:
            result = self._selector_variant_03()
        elif selected_selector == 3:
            result = self._selector_variant_04()
        elif selected_selector == 4:
            result = self._selector_variant_06()
        elif selected_selector == 5:
            result = self._selector_variant_07()
        elif selected_selector == 6:
            result = self._selector_variant_08()
        elif selected_selector == 7:
            result = self._selector_variant_09()
        elif selected_selector == 8:
            result = self._selector_variant_10()
        else:
            result = self._selector_variant_07()  # Default to best variant
        
        return result
    
    def _update_meta_selector(self, selected_operator, success_mask):
        """Update meta-selector based on success of operator selection."""
        success_rate = float(np.mean(success_mask))
        success_rate = np.clip(success_rate, 0.0, 1.0)
        
        # Update sliding window history
        selector_idx = self.current_best_selector
        if len(self.selector_history[selector_idx]) >= self.selector_window_size:
            self.selector_history[selector_idx].pop(0)
        self.selector_history[selector_idx].append(success_rate)
        
        # Compute average reward from sliding window
        if len(self.selector_history[selector_idx]) > 0:
            avg_reward = float(np.mean(self.selector_history[selector_idx]))
        else:
            avg_reward = 0.5
        
        # Update Beta distribution parameters
        if success_rate > 0.2:
            self.selector_successes[selector_idx] += success_rate
        else:
            self.selector_failures[selector_idx] += (1.0 - success_rate)
        
        # Decay all selector parameters slightly to favor recent performance
        decay_factor = 0.999
        self.selector_successes *= decay_factor
        self.selector_failures *= decay_factor
        
        # Ensure minimum values to avoid collapse
        self.selector_successes = np.maximum(self.selector_successes, 0.1)
        self.selector_failures = np.maximum(self.selector_failures, 0.1)
        
        # Track recent rewards for adaptive switching
        self.recent_rewards.append(success_rate)
        if len(self.recent_rewards) > 50:
            self.recent_rewards.pop(0)
    
    # ========== SELECTOR VARIANT IMPLEMENTATIONS ==========
    
    def _selector_original(self):
        """Original UCB-based operator selection."""
        ucb_values = (self.operator_rewards / np.maximum(self.operator_counts, 1) + 
                      self.ucb_c * np.sqrt(np.log(sum(self.operator_counts)) / 
                      np.maximum(self.operator_counts, 1)))
        
        selected = np.argmax(ucb_values) + np.random.randint(-1, 2)
        selected = np.clip(selected, 0, self.n_strategies - 1)
        return int(selected)
    
    def _selector_variant_01(self):
        """Thompson Sampling with Beta posterior distributions (variant_01)."""
        successes = np.clip(self.operator_rewards, 0, self.operator_counts)
        failures = np.maximum(self.operator_counts - successes, 0)
        
        beta_samples = np.random.beta(successes + 1, failures + 1)
        
        if np.sum(beta_samples) == 0 or np.any(np.isnan(beta_samples)):
            return np.random.randint(0, self.n_strategies)
        
        return int(np.argmax(beta_samples))
    
    def _selector_variant_03(self):
        """Thompson Sampling with fitness-aware exploration boost (variant_03)."""
        success_rates = self.operator_rewards / np.maximum(self.operator_counts, 1)
        
        rate_variance = float(np.var(success_rates))
        rate_mean = float(np.mean(success_rates))
        
        alpha = np.clip(self.operator_rewards + 1.0, 1e-10, 1e10)
        beta = np.clip(self.operator_counts - self.operator_rewards + 1.0, 1e-10, 1e10)
        
        sampled_values = np.random.beta(alpha, beta)
        
        if rate_variance < 0.01 and rate_mean < 0.3:
            exploration_noise = np.random.uniform(0, 0.5, size=self.n_strategies)
            sampled_values = sampled_values + exploration_noise
        
        novelty_bonus = np.random.uniform(0, 0.2) * (1.0 / np.maximum(self.operator_counts, 1))
        sampled_values = sampled_values + novelty_bonus
        
        selected = int(np.argmax(sampled_values))
        return int(np.clip(selected, 0, self.n_strategies - 1))
    
    def _selector_variant_04(self):
        """Fitness-rank-aware adaptive selection (variant_04)."""
        perf_scores = self.operator_rewards / np.maximum(self.operator_counts, 1)
        
        if perf_scores.max() > perf_scores.min():
            perf_norm = (perf_scores - perf_scores.min()) / (perf_scores.max() - perf_scores.min() + 1e-10)
        else:
            perf_norm = np.ones(self.n_strategies) / self.n_strategies
        
        exploration_bonus = 1.0 / np.log(np.maximum(self.operator_counts, 2))
        exploration_bonus = exploration_bonus / (np.sum(exploration_bonus) + 1e-10)
        
        blended_scores = 0.6 * perf_norm + 0.4 * exploration_bonus
        
        probs = np.clip(blended_scores, 1e-10, None)
        probs = probs / probs.sum()
        
        selected = np.random.choice(self.n_strategies, p=probs)
        return int(np.clip(selected, 0, self.n_strategies - 1))
    
    def _selector_variant_06(self):
        """Simulated annealing schedule selection (variant_06)."""
        total_counts = float(np.sum(self.operator_counts))
        temperature = max(0.01, 10.0 / np.log2(total_counts + 2))
        
        reward_offset = float(np.max(self.operator_rewards)) + 1.0
        positive_rewards = self.operator_rewards - reward_offset
        
        exp_scores = np.exp(positive_rewards / max(temperature, 1e-10))
        softmax_probs = exp_scores / np.sum(exp_scores)
        
        softmax_probs = np.clip(softmax_probs, 1e-10, 1.0)
        softmax_probs = softmax_probs / np.sum(softmax_probs)
        
        selected = np.random.choice(self.n_strategies, p=softmax_probs)
        return int(selected)
    
    def _selector_variant_07(self):
        """Multi-Objective Thompson Sampling with entropy bonus (variant_07)."""
        total_counts = float(np.sum(self.operator_counts))
        
        avg_rewards = self.operator_rewards / np.maximum(self.operator_counts, 1)
        
        success_component = avg_rewards.copy()
        
        diversity_component = self.operator_diversity / (np.maximum(np.sum(self.operator_diversity), 1e-10))
        
        momentum_component = np.clip(self.operator_momentum, 0, 1)
        
        combined_reward = (0.5 * success_component + 
                          0.3 * diversity_component + 
                          0.2 * momentum_component)
        
        combined_reward = np.nan_to_num(combined_reward, nan=0.0, posinf=0.0, neginf=0.0)
        
        shape_param = combined_reward + 1e-6
        sampled_scores = np.random.gamma(shape_param, 1.0)
        
        exploration_bonus = self.ucb_c * np.sqrt(np.log(total_counts + 1) / np.maximum(self.operator_counts, 1))
        exploration_bonus = np.clip(exploration_bonus, 0, 10.0)
        
        final_scores = sampled_scores + exploration_bonus
        
        selected = int(np.argmax(final_scores))
        
        self.operator_momentum *= 0.9
        self.operator_diversity *= 0.95
        
        return int(np.clip(selected, 0, self.n_strategies - 1))
    
    def _selector_variant_08(self):
        """Entropy-driven adaptive operator selection (variant_08)."""
        n_ops = self.n_strategies
        
        if self.op_success_history is None:
            self.op_success_history = np.zeros((n_ops, 10))
            self.op_history_idx = 0
            self.last_selected = 0
        
        if hasattr(self, 'last_success_rate'):
            self.op_success_history[self.last_selected, self.op_history_idx] = self.last_success_rate
            self.op_history_idx = (self.op_history_idx + 1) % 10
        
        success_rates = np.mean(self.op_success_history, axis=1) + 1e-10
        
        p_normalized = success_rates / (np.sum(success_rates) + 1e-10)
        entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
        max_entropy = np.log(n_ops + 1e-10)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        success_variance = float(np.var(success_rates))
        
        explore_factor = max(0.1, 1.0 - normalized_entropy + min(0.4, success_variance * 10))
        exploit_factor = 1.0 - explore_factor
        
        exploit_probs = success_rates / np.sum(success_rates)
        explore_probs = np.ones(n_ops) / n_ops
        
        mixed_probs = explore_factor * explore_probs + exploit_factor * exploit_probs
        mixed_probs = mixed_probs / np.sum(mixed_probs)
        
        mixed_probs = np.clip(mixed_probs, 1e-10, 1.0)
        mixed_probs = mixed_probs / np.sum(mixed_probs)
        
        selected = np.random.choice(n_ops, p=mixed_probs)
        
        self.call_counter += 1
        
        if self.call_counter % 20 == 0 and normalized_entropy < 0.5:
            least_used = np.argmin(np.sum(self.op_success_history, axis=1))
            if np.random.random() < 0.3:
                selected = int(least_used)
        
        self.last_selected = int(selected)
        return int(selected)
    
    def _selector_variant_09(self):
        """Thompson Sampling with landscape-aware perturbation (variant_09)."""
        samples = np.array([np.random.beta(self.operator_successes[i] + 0.1, 
                                            self.operator_failures[i] + 0.1) 
                            for i in range(self.n_strategies)])
        
        if self.last_fitness is not None and len(self.last_fitness) > 1:
            f_range = float(np.max(self.last_fitness) - np.min(self.last_fitness))
            if f_range > 1e3:
                samples = samples + np.random.uniform(0, 0.3, size=self.n_strategies)
            elif f_range > 1e1:
                samples = samples + np.random.uniform(0, 0.15, size=self.n_strategies)
        
        return int(np.argmax(samples))
    
    def _selector_variant_10(self):
        """Entropy-adaptive softmax selection (variant_10)."""
        decay = 0.9
        
        current_rates = self.operator_rewards / np.maximum(self.operator_counts, 1)
        
        self.operator_success_ema = decay * self.operator_success_ema + (1 - decay) * current_rates
        
        rate_variance = float(np.var(self.operator_success_ema))
        rate_max = float(np.max(self.operator_success_ema))
        rate_min = float(np.min(self.operator_success_ema))
        
        if rate_max > rate_min + 1e-10:
            normalized_rates = (self.operator_success_ema - rate_min) / (rate_max - rate_min + 1e-10)
            normalized_rates = np.clip(normalized_rates, 1e-10, 1.0)
            entropy = -np.sum(normalized_rates * np.log(normalized_rates + 1e-10))
            max_entropy = np.log(self.n_strategies + 1e-10)
            entropy_factor = max(0.1, entropy / (max_entropy + 1e-10))
        else:
            entropy_factor = 1.0
        
        temperature = max(0.1, 2.0 * entropy_factor / (rate_variance * 10 + 0.1))
        
        scaled = self.operator_success_ema / (temperature + 1e-10)
        scaled = scaled - np.max(scaled)
        exp_rates = np.exp(scaled)
        probs = exp_rates / (np.sum(exp_rates) + 1e-10)
        
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (np.sum(probs) + 1e-10)
        
        return int(np.random.choice(self.n_strategies, p=probs))
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using multi-strategy ensemble with fitness-weighted selection."""
        np_pop = len(population)

        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)

        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]
        worst_idx = sorted_idx[-1]

        f_min, f_max = np.min(fitness), np.max(fitness)
        f_range = max(f_max - f_min, 1e-10)
        f_variance = np.var(fitness)
        best_worst_ratio = (f_min + 1e-10) / (f_max + 1e-10)

        mutants = np.empty_like(population)

        valid_mask = np.ones((np_pop, np_pop), dtype=bool)
        np.fill_diagonal(valid_mask, False)

        for i in range(np_pop):
            valid_mask[i, i] = False

        r1 = np.array([np.random.choice(np.where(valid_mask[i])[0]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) 
                       for i in range(np_pop)])

        base_f = self.f_base * (1.0 + 0.5 * np.log1p(f_variance) / (np_pop + 1))
        f_values = base_f + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.05, 2.5)

        mutant_rand = population[r1] + f_values[:, np.newaxis] * (
            population[r2] - population[r3])

        pbest = population[np.random.choice(pbest_idx, np_pop)]
        mutant_gradient = population + f_values[:, np.newaxis] * (
            pbest - population[r1] + population[r2] - population[r3])

        best_idx = sorted_idx[0]
        mutant_oppose = population[r1] + f_values[:, np.newaxis] * (
            population[best_idx] - population[r2] + population[r3] - population[r1])

        elite = population[sorted_idx[:max(1, int(np_pop * 0.05))]]
        elite_choice = elite[np.random.randint(0, len(elite), size=np_pop)]
        mutant_elite = elite_choice + f_values[:, np.newaxis] * (
            population[r1] - population[r2])

        explore_weight = min(0.5, f_variance / (f_range + 1e-10))
        exploit_weight = 1.0 - explore_weight

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

        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        nan_mask = np.any(np.isnan(mutants) | np.isinf(mutants), axis=1)
        if np.any(nan_mask):
            mutants[nan_mask] = population[nan_mask] + np.random.uniform(
                -0.1, 0.1, (np.sum(nan_mask), self.dim))
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
        """Update operator rewards based on success rate."""
        success_rate = float(np.mean(success_mask))
        
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
        
        # Update tracking for variant_08
        self.last_success_rate = float(success_rate)
        
        # Update tracking for variant_09
        self.operator_successes[selected_operator] += success_rate
        self.operator_failures[selected_operator] += (1.0 - success_rate)
        
        # Update tracking for variant_07
        if success_rate > 0:
            self.operator_diversity[selected_operator] += 0.1
            self.operator_momentum[selected_operator] = min(1.0, self.operator_momentum[selected_operator] + success_rate)
    
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
        
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return float(fitness[best_idx]), population[best_idx].copy()
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator using meta-adaptive selection
            selected_operator = self._select_mutation_operator_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip trials to bounds
            trials = self._clip_to_bounds(trials)
            
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
            
            if stopping_condition():
                break
            
            # Selection
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update operator rewards
            self._update_operator_rewards(selected_operator, success_mask)
            
            # Update meta-selector based on performance
            self._update_meta_selector(selected_operator, success_mask)
            
            # Adapt step size
            success_rate = float(np.mean(success_mask))
            self._adapt_step_size(success_rate)
            
            # Update last fitness for variant_09
            self.last_fitness = np.array([float(np.min(fitness))], dtype=float)
            
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
            self.generation = generation
        
        # Return best solution
        best_idx = np.argmin(fitness)
        return float(fitness[best_idx]), population[best_idx].copy()
```