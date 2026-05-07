```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    - META-ADAPTIVE: Dynamically selects best operator selection STRATEGY during optimization
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # META-ADAPTIVE: Strategy selection parameters
        # 11 strategies (original + 10 variants from benchmark)
        self.num_selection_strategies = 11
        self.selection_strategy_names = [
            'original', 'variant_01', 'variant_02', 'variant_03', 'variant_04',
            'variant_05', 'variant_06', 'variant_07', 'variant_08', 'variant_09', 'variant_10'
        ]
        
        # Meta-level Thompson Sampling for strategy selection
        self.meta_alpha = np.ones(self.num_selection_strategies)
        self.meta_beta = np.ones(self.num_selection_strategies)
        
        # Track performance of each selection strategy
        self.strategy_rewards = [[] for _ in range(self.num_selection_strategies)]
        self.strategy_counts = np.zeros(self.num_selection_strategies)
        
        # Strategy-specific operator tracking (to avoid state conflicts)
        self.strategy_state = [{} for _ in range(self.num_selection_strategies)]
        
        # Current selected meta-strategy
        self.current_selection_strategy = 0
        
        # Reward history for meta-adaptive learning
        self.meta_reward_window = 15
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            
            # META-ADAPTIVE: Select operator selection strategy dynamically
            self._select_meta_strategy()
            # Select and apply covariance adaptation operator using selected strategy
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            # META-ADAPTIVE: Update meta-level rewards
            self._update_meta_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _select_meta_strategy(self):
        """Select operator selection strategy using meta-level Thompson Sampling."""
        samples = np.random.beta(self.meta_alpha, self.meta_beta)
        self.current_selection_strategy = int(np.argmax(samples))
        self.strategy_counts[self.current_selection_strategy] += 1
        return self.current_selection_strategy
    
    def _update_meta_rewards(self):
        """Update meta-level rewards based on strategy performance."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Log-scaled improvement reward
        if improvement > 1e-15:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            # Small negative reward for no improvement
            reward = -0.1
        
        strategy = self.current_selection_strategy
        self.strategy_rewards[strategy].append(reward)
        
        # Keep window bounded
        if len(self.strategy_rewards[strategy]) > self.meta_reward_window:
            self.strategy_rewards[strategy].pop(0)
        
        # Update Beta distribution parameters based on recent performance
        n = len(self.strategy_rewards[strategy])
        if n >= 3:
            mean_reward = float(np.mean(self.strategy_rewards[strategy]))
            var_reward = float(np.var(self.strategy_rewards[strategy]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.meta_alpha[strategy] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.meta_beta[strategy] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.strategy_rewards[strategy]))
            if sum_reward > 0:
                self.meta_alpha[strategy] = 1.0 + sum_reward
            else:
                self.meta_beta[strategy] = 1.0 - sum_reward
    
    def _select_operator_thompson(self):
        """Select operator using the currently active selection strategy."""
        strategy = self.current_selection_strategy
        
        if strategy == 0:
            return self._select_operator_original()
        elif strategy == 1:
            return self._select_operator_variant_01()
        elif strategy == 2:
            return self._select_operator_variant_02()
        elif strategy == 3:
            return self._select_operator_variant_03()
        elif strategy == 4:
            return self._select_operator_variant_04()
        elif strategy == 5:
            return self._select_operator_variant_05()
        elif strategy == 6:
            return self._select_operator_variant_06()
        elif strategy == 7:
            return self._select_operator_variant_07()
        elif strategy == 8:
            return self._select_operator_variant_08()
        elif strategy == 9:
            return self._select_operator_variant_09()
        elif strategy == 10:
            return self._select_operator_variant_10()
        else:
            return self._select_operator_original()
    
    def _select_operator_original(self):
        """Original operator selection."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_variant_01(self):
        """Adaptive Epsilon-Greedy strategy."""
        state = self.strategy_state[1]
        
        operator_avg = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) > 0:
                operator_avg[i] = float(np.mean(self.operator_rewards[i]))

        total_selections = float(np.sum(self.operator_counts))
        epsilon = 1.0 / (1.0 + total_selections / max(self.dim, 10.0))
        epsilon = float(np.clip(epsilon, 0.01, 0.5))

        if np.random.random() < epsilon:
            selection_probs = 1.0 / (self.operator_counts + 1.0)
            selection_probs /= np.sum(selection_probs)
            selection_probs = np.clip(selection_probs, 1e-10, 1.0 - 1e-10)
            self.current_operator = int(np.random.choice(self.num_operators, p=selection_probs))
        else:
            self.current_operator = int(np.argmax(operator_avg))

        self.operator_counts[self.current_operator] += 1
        self.strategy_state[1] = state
        return self.current_operator
    
    def _select_operator_variant_02(self):
        """Performance-scaled softmax selection."""
        state = self.strategy_state[2]
        
        if 'op_scores' not in state:
            state['op_scores'] = np.zeros(self.num_operators)
            state['op_selections'] = np.zeros(self.num_operators)
            state['op_last_selected'] = np.zeros(self.num_operators, dtype=int)

        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        if improvement > 1e-15:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        decay = 0.9
        state['op_scores'] = decay * state['op_scores'] + (1.0 - decay) * reward

        current_gen = self.generation
        for i in range(self.num_operators):
            generations_since = current_gen - state['op_last_selected'][i]
            if generations_since > 5:
                state['op_scores'][i] += 0.5 * (generations_since - 5) / 10.0

        if self.stagnation_counter > self.max_stagnation // 3:
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
            if diversity < 0.05:
                for i in range(self.num_operators):
                    if self.selection_counts[i] < np.mean(self.selection_counts) * 0.5:
                        state['op_scores'][i] += 1.0

        scores = state['op_scores'] - np.max(state['op_scores']) + 1e-10
        temperature = 0.5
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / np.sum(exp_scores)

        cumsum = np.cumsum(probs)
        r = np.random.random()
        self.current_operator = int(np.searchsorted(cumsum, r))
        self.current_operator = min(self.current_operator, self.num_operators - 1)

        self.operator_counts[self.current_operator] += 1
        state['op_selections'][self.current_operator] += 1
        state['op_last_selected'][self.current_operator] = current_gen

        self.strategy_state[2] = state
        return self.current_operator
    
    def _select_operator_variant_03(self):
        """Difficulty-adaptive with progress tracking."""
        state = self.strategy_state[3]
        
        if 'progress_history' not in state:
            state['progress_history'] = []

        current_improvement = max(0.0, self.f_opt_prev - self.f_opt)
        state['progress_history'].append(current_improvement)

        window_size = 5 * self.dim
        if len(state['progress_history']) > window_size:
            state['progress_history'].pop(0)

        if len(state['progress_history']) >= 3:
            recent = state['progress_history'][-3:]
            avg_progress = np.mean(recent)
        else:
            avg_progress = current_improvement

        is_stagnant = (avg_progress < 1e-12 * max(abs(self.f_opt), 1.0)) and len(state['progress_history']) >= self.dim

        quality_scores = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            if len(self.operator_rewards[op]) > 0:
                quality_scores[op] = np.mean(self.operator_rewards[op])

        if is_stagnant:
            if np.sum(self.operator_counts) > self.dim * 3:
                inv_counts = 1.0 / (self.operator_counts + 1.0)
                self.current_operator = int(np.argmax(inv_counts))
            else:
                self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            if np.max(quality_scores) > 0:
                self.current_operator = int(np.argmax(quality_scores))
            else:
                self.current_operator = int(np.argmin(self.operator_counts))

        self.operator_counts[self.current_operator] += 1
        self.strategy_state[3] = state
        return self.current_operator
    
    def _select_operator_variant_04(self):
        """Entropy-driven Thompson Sampling with forced exploration."""
        state = self.strategy_state[4]
        
        entropy = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            a, b = max(self.alpha[i], 1e-10), max(self.beta[i], 1e-10)
            total = a + b
            e = np.log1p(total) - (a * np.log1p(a) + b * np.log1p(b)) / total
            entropy[i] = max(e, 0.0)

        total_entropy = np.sum(entropy)
        if total_entropy > 1e-10:
            explore_weights = entropy / total_entropy
        else:
            explore_weights = np.ones(self.num_operators) / self.num_operators

        total_counts = np.sum(self.operator_counts) + 1e-10
        count_penalty = 1.0 - (self.operator_counts / total_counts)
        count_penalty = np.clip(count_penalty, 0.0, 1.0)

        samples = np.random.beta(np.maximum(self.alpha, 1e-10), np.maximum(self.beta, 1e-10))
        combined_score = (
            0.4 * samples / (np.max(samples) + 1e-10) +
            0.4 * explore_weights / (np.max(explore_weights) + 1e-10) +
            0.2 * count_penalty / (np.max(count_penalty) + 1e-10)
        )

        if 'last_selected_gen' not in state:
            state['last_selected_gen'] = np.zeros(self.num_operators, dtype=int)
        
        gens_since_selection = self.generation - state['last_selected_gen']
        forced_threshold = max(20, self.dim * 2)
        for i in range(self.num_operators):
            if gens_since_selection[i] > forced_threshold:
                combined_score[i] *= 3.0
                if gens_since_selection[i] > forced_threshold * 2:
                    combined_score[i] *= 2.0

        self.current_operator = int(np.argmax(combined_score))
        self.operator_counts[self.current_operator] += 1
        state['last_selected_gen'][self.current_operator] = self.generation
        self.strategy_state[4] = state
        return self.current_operator
    
    def _select_operator_variant_05(self):
        """