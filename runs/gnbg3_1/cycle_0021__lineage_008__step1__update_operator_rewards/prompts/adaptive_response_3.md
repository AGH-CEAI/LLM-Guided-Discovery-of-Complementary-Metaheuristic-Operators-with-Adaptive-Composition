```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer with meta-adaptive reward strategy selection.
    
    Features:
    - 5 covariance adaptation strategies
    - 9 reward update strategies with meta-Thompson Sampling selection
    - Automatic strategy switching based on performance
    - All edge cases handled (no -inf, no NaN, clipped bounds)
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
        
        # META-ADAPTIVE: Multiple reward update strategies
        self.num_reward_strategies = 9
        self.reward_strategy_names = [
            'basic_log', 'ucb_success', 'rank_multi', 'personal_best',
            'rank_generational', 'diversity_explore', 'rank_multi_component',
            'rank_comparative', 'elite_relative'
        ]
        
        # Meta-Thompson Sampling for reward strategy selection
        self.reward_alpha = np.ones(self.num_reward_strategies)
        self.reward_beta = np.ones(self.num_reward_strategies)
        
        # Track performance of each reward strategy
        self.reward_strategy_rewards = {i: [] for i in range(self.num_reward_strategies)}
        self.reward_strategy_performance = np.zeros(self.num_reward_strategies)
        self.reward_strategy_trials = np.zeros(self.num_reward_strategies)
        self.current_reward_strategy = 0
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
    
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
    
    def _select_reward_strategy_thompson(self):
        """Select reward update strategy using Thompson Sampling."""
        samples = np.random.beta(self.reward_alpha, self.reward_beta)
        self.current_reward_strategy = int(np.argmax(samples))
        self.reward_strategy_trials[self.current_reward_strategy] += 1
        return self.current_reward_strategy
    
    def _update_reward_strategy_rewards(self, reward_value):
        """Update meta-Thompson Sampling based on achieved reward."""
        strategy = self.current_reward_strategy
        self.reward_strategy_rewards[strategy].append(float(reward_value))
        
        if len(self.reward_strategy_rewards[strategy]) > 20:
            self.reward_strategy_rewards[strategy].pop(0)
        
        recent_rewards = self.reward_strategy_rewards[strategy]
        if len(recent_rewards) >= 3:
            mean_reward = float(np.mean(recent_rewards))
            var_reward = float(np.var(recent_rewards))
            n = len(recent_rewards)
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.reward_alpha[strategy] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.reward_beta[strategy] = float(max(1.0, (1.0 - mean_reward) * ((n - 1) * (1.0 - mean_reward) / denom + 1)))
        elif len(recent_rewards) >= 1:
            sum_reward = float(np.sum(recent_rewards))
            if sum_reward > 0:
                self.reward_alpha[strategy] = 1.0 + sum_reward
            else:
                self.reward_beta[strategy] = 1.0 - sum_reward
    
    def _dispatch_reward_update(self):
        """Dispatch to the selected reward update strategy."""
        strategy = self.current_reward_strategy
        
        if strategy == 0:
            return self._update_reward_basic_log()
        elif strategy == 1:
            return self._update_reward_ucb_success()
        elif strategy == 2:
            return self._update_reward_rank_multi()
        elif strategy == 3:
            return self._update_reward_personal_best()
        elif strategy == 4:
            return self._update_reward_rank_generational()
        elif strategy == 5:
            return self._update_reward_diversity_explore()
        elif strategy == 6:
            return self._update_reward_rank_multi_component()
        elif strategy == 7:
            return self._update_reward_rank_comparative()
        else:
            return self._update_reward_elite_relative()
    
    # ============== REWARD UPDATE STRATEGY IMPLEMENTATIONS ==============
    
    def _update_reward_basic_log(self):
        """Basic log-scaled improvement reward (original)."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if not hasattr(self, 'op_cumulative_reward'):
            self.op_cumulative_reward = np.zeros(self.num_operators)
       