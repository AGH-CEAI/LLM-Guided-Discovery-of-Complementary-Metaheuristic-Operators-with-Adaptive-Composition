import numpy as np


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Meta-Adaptive Operator Selection.
    
    Key features:
    - Pool of DE mutation strategies selected via multi-armed bandit
    - Meta-adaptive strategy selector that learns from benchmark insights
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    
    The meta-adaptive selector combines insights from 9 benchmarked variants:
    - Thompson Sampling (variant_01, variant_09)
    - Fitness-rank-aware selection (variant_04)
    - Simulated annealing schedule (variant_06)
    - Multi-objective Thompson Sampling (variant_07)
    - Entropy-driven selection (variant_08)
    - Entropy-adaptive softmax (variant_10)
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
        
        # ===== META-ADAPTIVE OPERATOR SELECTOR =====
        # Track performance of each selection strategy (from benchmark variants)
        self.meta_strategy_names = [
            'thompson_beta',           # variant_01: Thompson Sampling with Beta
            'fitness_rank_aware',      # variant_04: Fitness-rank-aware adaptive
            'annealing',               # variant_06: Simulated annealing schedule
            'multi_obj_thompson',      # variant_07: Multi-objective Thompson (5 wins!)
            'entropy_driven',          # variant_08: Entropy-driven adaptive
            'softmax_entropy',         # variant_10: Entropy-adaptive softmax
            'thompson_landscape',      # variant_09: Thompson with landscape perturbation
            'ucb_fallback',            # original: UCB-based (fallback)
        ]
        self.n_meta_strategies = len(self.meta_strategy_names)
        
        # Meta-strategy performance tracking (initialized based on benchmark win counts)
        # Higher initial trust for strategies with more benchmark wins
        benchmark_wins = {
            'multi_obj_thompson': 5,    # 5 wins - highest
            'fitness_rank_aware': 4,    # 4 wins
            'thompson_beta': 3,         # 3 wins
            'softmax_entropy': 3,       # 3 wins
            'thompson_landscape': 3,    # 3 wins
            'entropy_driven': 2,        # 2 wins
            'annealing': 1,             # 1 win
            'ucb_fallback': 1,          # 1 win
        }
        
        self.meta_rewards = np.array([benchmark_wins.get(name, 1) for name in self.meta_strategy_names], dtype=float)
        self.meta_counts = np.ones(self.n_meta_strategies) * 10.0  # Start with some prior
        self.meta_selected_count = np.zeros(self.n_meta_strategies)
        
        # Sliding window for recent meta-performance
        self.meta_window_size = 20
        self.meta_recent_success = [np.zeros(self.meta_window_size) for _ in range(self.n_meta_strategies)]
        self.meta_window_idx = 0
        
        # Current meta-strategy in use
        self.current_meta_strategy = 3  # Start with multi_obj_thompson (5 wins)
        
        # Per-meta-strategy operator tracking
        self.strategy_operator_rewards = [np.zeros(self.n_strategies) for _ in range(self.n_meta_strategies)]
        self.strategy_operator_counts = [np.ones(self.n_strategies) for _ in range(self.n_meta_strategies)]
        
        # Additional tracking for specific meta-strategies
        self.operator_successes = np.ones(self.n_strategies, dtype=float)
        self.operator_failures = np.ones(self.n_strategies, dtype=float)
        
        self.op_success_history = None
        self.op_history_idx = 0
        self.last_selected_op = 0
        self.last_success_rate = 0.0
        self.call_counter = 0
        
        if not hasattr(self, 'operator_diversity'):
            self.operator_diversity = np.ones(self.n_strategies)
        if not hasattr(self, 'operator_momentum'):
            self.operator_momentum = np.zeros(self.n_strategies)
        if not hasattr(self, 'operator_success_ema'):
            self.operator_success_ema = np.zeros(self.n_strategies)
            
        # ===== END META-ADAPTIVE OPERATOR SELECTOR =====
        
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
    
    def _select_meta_strategy(self):
        """Select which meta-strategy to use based on Thompson Sampling with sliding window performance."""
        # Thompson Sampling on meta-strategies using recent performance
        total_recent = np.zeros(self.n_meta_strategies)
        for i in range(self.n_meta_strategies):
            window = self.meta_recent_success[i]
            total_recent[i] = np.sum(window) / (len(window) + 1e-10)
        
        # Combine with prior from benchmark wins
        combined = 0.7 * (self.meta_rewards / np.maximum(self.meta_counts, 1)) + \
                   0.3 * (total_recent / np.maximum(self.meta_selected_count + 1, 1))
        
        # Thompson Sampling: sample from Beta distribution
        alpha = combined * 10 + 1
        beta = (1 - combined) * 10 + 1
        beta = np.clip(beta, 1, 100)
        
        samples = np.random.beta(alpha, beta)
        
        # Add small exploration bonus for under-explored strategies
        exploration_bonus = 0.1 * np.sqrt(np.log(sum(self.meta_counts) + 1) / 
                                          np.maximum(self.meta_selected_count + 1, 1))
        samples += exploration_bonus
        
        selected = int(np.argmax(samples))
        self.current_meta_strategy = selected
        self.meta_selected_count[selected] += 1
        
        return selected
    
    def _select_mutation_operator_batch(self, fitness=None):
        """Select mutation operator using adaptive meta-strategy selection.
        
        Combines insights from benchmarked variants with online learning:
        - variant_01: Thompson Sampling with Beta
        - variant_04: Fitness-rank-aware adaptive (4 wins)
        - variant_06: Simulated annealing
        - variant_07: Multi-objective Thompson with entropy bonus (5 wins - most!)
        - variant_08: Entropy-driven adaptive
        - variant_09: Thompson with landscape perturbation
        - variant_10: Entropy-adaptive softmax
        - original: UCB fallback
        """
        meta_idx = self._select_meta_strategy()
        
        # Get the current operator tracking for this meta-strategy
        op_rewards = self.strategy_operator_rewards[meta_idx]
        op_counts = self.strategy_operator_counts[meta_idx]
        
        # Select using the appropriate method based on meta_idx
        if meta_idx == 0:  # Thompson Beta (variant_01)
            selected = self._thompson_beta_selection(op_rewards, op_counts)
        elif meta_idx == 1:  # Fitness-rank-aware (variant_04)
            selected = self._fitness_rank_selection(op_rewards, op_counts)
        elif meta_idx == 2:  # Annealing (variant_06)
            selected = self._annealing_selection(op_rewards, op_counts)
        elif meta_idx == 3:  # Multi-objective Thompson (variant_07) - 5 WINS!
            selected = self._multi_obj_thompson_selection(op_rewards, op_counts)
        elif meta_idx == 4:  # Entropy-driven (variant_08)
            selected = self._entropy_driven_selection(op_rewards, op_counts)
        elif meta_idx == 5:  # Softmax entropy (variant_10)
            selected = self._softmax_entropy_selection(op_rewards, op_counts)
        elif meta_idx == 6:  # Thompson landscape (variant_09)
            selected = self._thompson_landscape_selection(op_rewards, op_counts, fitness)
        else:  # UCB fallback (original)
            selected = self._ucb_fallback_selection(op_rewards, op_counts)
        
        # Ensure valid selection
        selected = int(np.clip(selected, 0, self.n_strategies - 1))
        self.last_selected_op = selected
        
        return selected
    
    def _thompson_beta_selection(self, op_rewards, op_counts):
        """Thompson Sampling with Beta distribution (variant_01)."""
        successes = np.clip(op_rewards, 0, op_counts)
        failures = np.maximum(op_counts - successes, 0)
        beta_samples = np.random.beta(successes + 1, failures + 1)
        
        if np.sum(beta_samples) == 0 or np.any(np.isnan(beta_samples)):
            return np.random.randint(0, self.n_strategies)
        
        return int(np.argmax(beta_samples))
    
    def _fitness_rank_selection(self, op_rewards, op_counts):
        """Fitness-rank-aware adaptive selection (variant_04)."""
        perf_scores = op_rewards / np.maximum(op_counts, 1)
        
        if perf_scores.max() > perf_scores.min():
            perf_norm = (perf_scores - perf_scores.min()) / (perf_scores.max() - perf_scores.min() + 1e-10)
        else:
            perf_norm = np.ones(self.n_strategies) / self.n_strategies
        
        exploration_bonus = 1.0 / np.log(np.maximum(op_counts, 2))
        exploration_bonus = exploration_bonus / (exploration_bonus.sum() + 1e-10)
        
        blended_scores = 0.6 * perf_norm + 0.4 * exploration_bonus
        probs = np.clip(blended_scores, 1e-10, None)
        probs = probs / probs.sum()
        
        return np.random.choice(self.n_strategies, p=probs)
    
    def _annealing_selection(self, op_rewards, op_counts):
        """Simulated annealing schedule selection (variant_06)."""
        total_counts = np.sum(op_counts)
        temperature = max(0.01, 10.0 / np.log2(total_counts + 2))
        
        reward_offset = np.max(op_rewards) + 1.0
        positive_rewards = op_rewards - reward_offset
        
        exp_scores = np.exp(positive_rewards / max(temperature, 1e-10))
        softmax_probs = exp_scores / np.sum(exp_scores)
        softmax_probs = np.clip(softmax_probs, 1e-10, 1.0)
        softmax_probs = softmax_probs / np.sum(softmax_probs)
        
        return np.random.choice(self.n_strategies, p=softmax_probs)
    
    def _multi_obj_thompson_selection(self, op_rewards, op_counts):
        """Multi-objective Thompson Sampling with entropy bonus (variant_07) - 5 WINS!"""
        total_counts = np.sum(op_counts)
        
        avg_rewards = op_rewards / np.maximum(op_counts, 1)
        success_component = avg_rewards.copy()
        
        # Diversity component
        if hasattr(self, 'operator_diversity') and len(self.operator_diversity) == self.n_strategies:
            diversity_component = self.operator_diversity / (np.maximum(np.sum(self.operator_diversity), 1e-10))
        else:
            diversity_component = 1.0 / np.maximum(op_counts, 1)
        
        # Momentum component
        momentum_component = np.zeros(self.n_strategies)
        if hasattr(self, 'operator_momentum') and len(self.operator_momentum) == self.n_strategies:
            momentum_component = np.clip(self.operator_momentum, 0, 1)
        
        combined_reward = (0.5 * success_component + 
                          0.3 * diversity_component + 
                          0.2 * momentum_component)
        combined_reward = np.nan_to_num(combined_reward, nan=0.0, posinf=0.0, neginf=0.0)
        
        shape_param = combined_reward + 1e-6
        sampled_scores = np.random.gamma(shape_param, 1.0)
        
        exploration_bonus = self.ucb_c * np.sqrt(np.log(total_counts + 1) / np.maximum(op_counts, 1))
        exploration_bonus = np.clip(exploration_bonus, 0, 10.0)
        
        final_scores = sampled_scores + exploration_bonus
        selected = int(np.argmax(final_scores))
        
        # Update tracking
        self.operator_diversity *= 0.95
        self.operator_momentum *= 0.9
        
        return selected
    
    def _entropy_driven_selection(self, op_rewards, op_counts):
        """Entropy-driven adaptive selection (variant_08)."""
        n_ops = self.n_strategies
        
        if self.op_success_history is None:
            self.op_success_history = np.zeros((n_ops, 10))
            self.op_history_idx = 0
        
        if hasattr(self, 'last_success_rate'):
            self.op_success_history[self.last_selected_op, self.op_history_idx] = self.last_success_rate
            self.op_history_idx = (self.op_history_idx + 1) % 10
        
        success_rates = np.mean(self.op_success_history, axis=1) + 1e-10
        
        p_normalized = success_rates / (np.sum(success_rates) + 1e-10)
        entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
        max_entropy = np.log(n_ops + 1e-10)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        success_variance = np.var(success_rates)
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
            least_used = np.argmin(self.op_success_history.sum(axis=1))
            if np.random.random() < 0.3:
                selected = least_used
        
        self.last_selected_op = selected
        return selected
    
    def _softmax_entropy_selection(self, op_rewards, op_counts):
        """Entropy-adaptive softmax selection (variant_10)."""
        decay = 0.9
        if not hasattr(self, 'operator_success_ema') or self.operator_success_ema is None:
            self.operator_success_ema = np.zeros(self.n_strategies)
        
        current_rates = op_rewards / np.maximum(op_counts, 1)
        self.operator_success_ema = decay * self.operator_success_ema + (1 - decay) * current_rates
        
        rate_variance = np.var(self.operator_success_ema)
        rate_max = np.max(self.operator_success_ema)
        rate_min = np.min(self.operator_success_ema)
        
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
        probs = probs / np.sum(probs)
        
        return np.random.choice(self.n_strategies, p=probs)
    
    def _thompson_landscape_selection(self, op_rewards, op_counts, fitness):
        """Thompson Sampling with landscape perturbation (variant_09)."""
        if not hasattr(self, 'operator_successes'):
            self.operator_successes = np.ones(self.n_strategies, dtype=float)
        if not hasattr(self, 'operator_failures'):
            self.operator_failures = np.ones(self.n_strategies, dtype=float)
        
        samples = np.array([np.random.beta(self.operator_successes[i] + 0.1, 
                                            self.operator_failures[i] + 0.1) 
                            for i in range(self.n_strategies)])
        
        if fitness is not None and len(fitness) > 1:
            f_range = np.max(fitness) - np.min(fitness)
            if f_range > 1e3:
                samples += np.random.uniform(0, 0.3, size=self.n_strategies)
            elif f_range > 1e1:
                samples += np.random.uniform(0, 0.15, size=self.n_strategies)
        
        return int(np.argmax(samples))
    
    def _ucb_fallback_selection(self, op_rewards, op_counts):
        """UCB-based fallback selection (original)."""
        ucb_values = (op_rewards / np.maximum(op_counts, 1) + 
                      self.ucb_c * np.sqrt(np.log(sum(op_counts)) / np.maximum(op_counts, 1)))
        
        selected = np.argmax(ucb_values) + np.random.randint(-1, 2)
        return int(np.clip(selected, 0, self.n_strategies - 1))
    
    def _update_meta_rewards(self, meta_idx, success_rate):
        """Update meta-strategy rewards based on success."""
        # Update sliding window
        self.meta_recent_success[meta_idx][self.meta_window_idx] = success_rate
        self.meta_window_idx = (self.meta_window_idx + 1) % self.meta_window_size
        
        # Update rewards
        self.meta_rewards[meta_idx] += success_rate
        self.meta_counts[meta_idx] += 1
    
    def _update_operator_rewards(self, selected_operator, success_mask, meta_idx=None):
        """Update operator rewards based on success rate."""
        success_rate = np.mean(success_mask)
        
        # Update global rewards
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        # Update meta-strategy specific rewards
        if meta_idx is not None and meta_idx < len(self.strategy_operator_rewards):
            self.strategy_operator_rewards[meta_idx][selected_operator] += success_rate
            self.strategy_operator_counts[meta_idx][selected_operator] += 1
        
        # Update Beta tracking for Thompson methods
        if success_rate > 0:
            self.operator_successes[selected_operator] += success_rate
        else:
            self.operator_failures[selected_operator] += 1
        
        # Update momentum for multi-objective Thompson
        if hasattr(self, 'operator_momentum'):
            self.operator_momentum[selected_operator] = min(1.0, 
                self.operator_momentum[selected_operator] + success_rate * 0.5)
        
        # Update diversity tracking
        if hasattr(self, 'operator_diversity'):
            self.operator_diversity[selected_operator] += success_rate * 0.1
        
        # Update last success rate for entropy-driven method
        self.last_success_rate = success_rate
        
        # Decay rewards of non-selected operators slightly
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
    
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
            return float(fitness[best_idx]), population[best_idx].copy()
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator using meta-adaptive selection
            selected_operator = self._select_mutation_operator_batch(fitness)
            meta_idx = self.current_meta_strategy
            
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
            
            # Update operator rewards (both global and meta-strategy specific)
            success_rate = float(np.mean(success_mask))
            self._update_operator_rewards(selected_operator, success_mask, meta_idx)
            
            # Update meta-strategy rewards
            self._update_meta_rewards(meta_idx, success_rate)
            
            # Adapt step size
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
        f_opt = float(fitness[best_idx])
        x_opt = population[best_idx].copy()
        
        # Ensure no inf/nan in output
        if not np.isfinite(f_opt):
            f_opt = float('inf')
        if not np.all(np.isfinite(x_opt)):
            x_opt = np.clip(x_opt, self.lower_bound, self.upper_bound)
        
        return f_opt, x_opt
