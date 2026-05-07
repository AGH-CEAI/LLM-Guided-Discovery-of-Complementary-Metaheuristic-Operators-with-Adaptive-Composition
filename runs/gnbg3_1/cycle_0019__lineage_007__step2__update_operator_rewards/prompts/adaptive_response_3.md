```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    AND reward update strategy during optimization using hierarchical Thompson Sampling.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - 10 reward update strategies (adaptive selection)
    - Hierarchical Thompson Sampling for operator selection
    - Sliding window credit assignment with meta-level adaptation
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
        
        # === COVARIANCE OPERATOR SELECTION ===
        self.num_cov_operators = 5
        self.cov_operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling for covariance operators
        self.alpha_cov = np.ones(self.num_cov_operators)
        self.beta_cov = np.ones(self.num_cov_operators)
        
        # === REWARD STRATEGY SELECTION (META-LEVEL) ===
        self.num_reward_strategies = 10
        self.reward_strategy_names = [
            'original', 'variant_01', 'variant_02', 'variant_04', 'variant_05',
            'variant_06', 'variant_07', 'variant_08', 'variant_09', 'variant_10'
        ]
        
        # Thompson Sampling for reward strategies
        self.alpha_reward = np.ones(self.num_reward_strategies)
        self.beta_reward = np.ones(self.num_reward_strategies)
        
        # Track performance of each reward strategy
        self.reward_strategy_rewards = {i: [] for i in range(self.num_reward_strategies)}
        self.reward_strategy_selection_counts = np.zeros(self.num_reward_strategies)
        self.reward_strategy_improvement_sums = np.zeros(self.num_reward_strategies)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_cov_operators)}
        self.operator_counts = np.zeros(self.num_cov_operators)
        self.selection_counts = np.zeros(self.num_cov_operators)
        
