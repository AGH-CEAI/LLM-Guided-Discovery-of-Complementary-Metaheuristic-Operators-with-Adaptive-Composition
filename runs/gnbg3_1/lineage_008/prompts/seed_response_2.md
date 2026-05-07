```python
import numpy as np


class AdaptiveCovarianceMatrixSearchWithOperatorPool:
    """
    A CMA-ES-inspired optimizer with an adaptive pool of mutation operators.
    Operators compete based on success history, and covariance adaptation
    uses multiple update strategies that can be tuned independently.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self._dim = dim
        self._np = min(150, max(4 * dim, 80))
        self._lower = -100.0
        self._upper = 100.0
        self._lambda = self._np
        
        self._mu_eff = self._np / 10
        self._mu_cov = self._np / 10
        
        self._step_size = 0.5 * (self._upper - self._lower)
        self._p_s_success = 0.5
        self._chi_n = np.sqrt(self._dim) * (1 - 1 / (4 * self._dim) + 1 / (21 * self._dim ** 2))
        
        self._cc = 2 / (dim + 7)
        self._cs = (self._mu_eff + 2) / (dim + self._mu_eff + 3)
        self._c1 = 2 / ((dim + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(1 - self._c1, 2 * self._mu_cov / ((dim + 2) ** 2 + 2 * self._mu_cov))
        self._damps = 1 + 2 * max(0, np.sqrt(self._mu_eff) - 1) / self._cs
        
        self._n_operators = 4
        self._operator_weights = np.ones(self._n_operators) / self._n_operators
        self._operator_rewards = np.zeros(self._n_operators)
        self._operator_counts = np.zeros(self._n_operators)
        
        self._pc = np.zeros(dim)
        self._ps = np.zeros(dim)
        self._generation = 0
        self._best_fitness = np.inf
        self._best_x = None
        self._no_improvement_count = 0
        
        self._stagnation_threshold = max(50, int(5 * dim))
        self._history = {'fitness': [], 'diversity': [], 'step_size': []}
        
        self._active_operator = None
        self._active_z = None
        
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        pop = np.random.uniform(self._lower, self._upper, (self._np, self._dim))
        pop = np.clip(pop, self._lower, self._upper)
        self._population = pop
        self._fitness = np.full(self._np, np.inf)
        self._mean = pop.mean(axis=0)
        self._covariance = np.eye(self._dim)
        self._covariance_sqrt = np.eye(self._dim)
        
    def _sample_trials_batch(self):
        """Sample trial solutions using current covariance and operator weights."""
        z_global = np.random.randn(self._np, self._dim)
        trials_global = self._mean + self._step_size * (self._covariance_sqrt @ z_global.T).T
        trials_global = np.clip(trials_global, self._lower, self._upper)
        
        idx_r1 = np.random.randint(0, self._np, size=self._np)
        idx_r2 = np.random.randint(0, self._np, size=self._np)
        idx_r3 = np.random.randint(0, self._np, size=self._np)
        idx_r4 = np.random.randint(0, self._np, size=self._np)
        diff_idx = idx_r2 != idx_r3
        idx_r3 = idx_r3[diff_idx]
        idx_r4 = idx_r4[diff_idx]
        n_small = np.sum(diff_idx)
        
        trials_de = np.copy(trials_global)
        if n_small > 0:
            diff_mask = np.zeros(self._np, dtype=bool)
            diff_mask[:n_small] = diff_idx
            trials_de[diff_mask] = self._population[idx_r1[diff_mask]] + self._step_size * 0.5 * (
                self._population[idx_r2[diff_mask]] - self._population[idx_r3[diff_mask]]
            )
            trials_de[diff_mask] = np.clip(trials_de[diff_mask], self._lower, self._upper)
        
        z_best = np.random.randn(self._np, self._dim)
        trials_best = self._mean + self._step_size * (
            0.3 * (self._best_x - self._mean) if self._best_x is not None else 0
            + 0.7 * (self._covariance_sqrt @ z_best.T).T
        )
        trials_best = np.clip(trials_best, self._lower, self._upper)
        
        z_iso = np.random.randn(self._np, self._dim)
        trials_iso = self._mean + self._step_size * (self._covariance_sqrt @ z_iso.T).T * 0.8
        trials_iso = np.clip(trials_iso, self._lower, self._upper)
        
        trials_stack = np.stack([trials_global, trials_de, trials_best, trials_iso], axis=1)
        op_indices = np.random.choice(self._n_operators, size=self._np, p=self._operator_weights)
        batch_idx = np.arange(self._np)
        selected_trials = trials_stack[batch_idx, op_indices]
        
        z_selected = z_stack[batch_idx, op_indices]
        self._active_operator = op_indices
        self._active_z = z_selected
        
        return selected_trials
    
    def _select_survivors_batch(self, trials, trial_fitness):
        """Select survivors using comma strategy with operator reward tracking."""
        n_valid = min(len(trial_fitness), len(trials))
        if n_valid < len(trials):
            trials = trials[:n_valid]
            trial_fitness = trial_fitness[:n_valid]
        
        improved = trial_fitness < self._fitness[:n_valid]
        n_improved = np.sum(improved)
        
        op_success_counts = np.zeros(self._n_operators)
        if n_improved > 0:
            improved_idx = np.where(improved)[0]
            success_ops = self._active_operator[improved_idx]
            for op_idx in range(self._n_operators):
                op_success_counts[op_idx] = np.sum(success_ops == op_idx)
            
            self._population[improved_idx] = trials[improved_idx]
            self._fitness[improved_idx] = trial_fitness[improved_idx]
        
        self._operator_counts += op_success_counts
        self._update_operator_rewards(success_counts=op_success_counts)
        
        return self._population.copy(), self._fitness.copy(), n_improved
    
    def _update_operator_rewards(self, success_counts):
        """Update operator weights based on success counts using exponential moving average."""
        total_counts = self._operator_counts + 1e-10
        success_rate = success_counts / total_counts
        
        alpha = 0.1
        self._operator_rewards = (1 - alpha) * self._operator_rewards + alpha * success_rate
        
        max_reward = self._operator_rewards.max()
        exp_weights = np.exp(0.3 * (self._operator_rewards - max_reward))
        self._operator_weights = exp_weights / (exp_weights.sum() + 1e-10)
        
        min_weight = 0.05
        self._operator_weights = np.maximum(self._operator_weights, min_weight)
        self._operator_weights /= self._operator_weights.sum()
    
    def _adapt_step_size(self, n_improved):
        """Adapt step size based on success rate of recent trials."""
        success_rate = n_improved / self._np
        target_rate = 0.2
        
        if success_rate > target_rate:
            self._step_size *= 1 + self._cs * (success_rate - target_rate) / (1 - target_rate)
        else:
            self._step_size *= 1 - self._cs * (target_rate - success_rate) / target_rate
        
        self._step_size = np.clip(self._step_size, 1e-10, self._upper - self._lower)
    
    def _adapt_covariance_variant_01(self, improved_mask):
        """Rank-1 covariance update using evolution path (simplified CMA-ES)."""
        z_mean = self._active_z[improved_mask].mean(axis=0) if np.any(improved_mask) else np.zeros(self._dim)
        
        self._ps = (1 - self._cs) * self._ps + np.sqrt(self._cs * (2 - self._cs) * self._mu_eff) * z_mean
        
        hs = np.linalg.norm(self._ps) / np.sqrt(1 - (1 - self._cs) ** (2 * self._generation)) < 1.5 + 1 / (self._dim + 1)
        self._pc = (1 - self._cc) * self._pc + hs * np.sqrt(self._cc * (2 - self._cc) * self._mu_eff) * z_mean
        
        rank_one = np.outer(self._pc, self._pc)
        self._covariance = (1 - self._c1 - self._c1 * self._cmu) * self._covariance + self._c1 * rank_one
        
        self._eigen_decompose_covariance()
    
    def _adapt_covariance_variant_08(self, improved_mask):
        """Rank-μ covariance update using top-performing individuals."""
        if not np.any(improved_mask):
            return
        
        n_select = min(20, np.sum(improved_mask))
        top_idx = np.argsort(self._fitness)[:n_select]
        z_top = self._active_z[top_idx]
        
        rank_mu = np.zeros((self._dim, self._dim))
        for i in range(n_select):
            rank_mu += np.outer(z_top[i], z_top[i])
        rank_mu /= n_select
        
        self._covariance = (1 - self._cmu) * self._covariance + self._cmu * rank_mu
        
        self._eigen_decompose_covariance()
    
    def _adapt_covariance_variant_09(self, improved_mask):
        """Combined rank-1 and rank-μ update with weighted contribution."""
        z_mean = self._active_z[improved_mask].mean(axis=0) if np.any(improved_mask) else np.zeros(self._dim)
        
        rank_one = np.outer(z_mean, z_mean)
        
        if np.any(improved_mask):
            n_top = min(self._mu_cov, np.sum(improved_mask))
            top_idx = np.argsort(self._fitness)[:n_top]
            z_top = self._active_z[top_idx]
            
            rank_mu = np.zeros((self._dim, self._dim))
            for i in range(n_top):
                rank_mu += np.outer(z_top[i], z_top[i])
            rank_mu /= n_top
        else:
            rank_mu = np.zeros((self._dim, self._dim))
        
        self._covariance = (1 - self._c1 - self._cmu) * self._covariance + self._c1 * rank_one + self._cmu * rank_mu
        
        self._eigen_decompose_covariance()
    
    def _eigen_decompose_covariance(self):
        """Ensure covariance matrix stays positive definite via eigendecomposition."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self._covariance_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        except np.linalg.LinAlgError:
            self._covariance = np.eye(self._dim)
            self._covariance_sqrt = np.eye(self._dim)
    
    def _compute_diversity(self):
        """Compute population diversity as mean pairwise Euclidean distance."""
        if self._np < 2:
            return 0.0
        dist_matrix = np.linalg.norm(self._population[:, None] - self._population[None, :], axis=2)
        diversity = dist_matrix[np.triu_indices(self._np, k=1)].mean()
        return diversity
    
    def _restart_if_stagnant(self, current_best):
        """Restart population if no improvement for threshold generations."""
        if current_best >= self._best_fitness:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        
        if self._no_improvement_count >= self._stagnation_threshold:
            self._no_improvement_count = 0
            self._step_size = 0.5 * (self._upper - self._lower)
            
            n_keep = max(2, self._np // 5)
            keep_indices = np.argsort(self._fitness)[:n_keep]
            kept_pop = self._population[keep_indices].copy()
            kept_fit = self._fitness[keep_indices].copy()
            
            new_pop = np.random.uniform(self._lower, self._upper, (self._np - n_keep, self._dim))
            new_pop = np.clip(new_pop, self._lower, self._upper)
            
            self._population = np.vstack([kept_pop, new_pop])
            self._fitness = np.concatenate([kept_fit, np.full(self._np - n_keep, np.inf)])
            self._mean = self._population.mean(axis=0)
            
            self._covariance = np.eye(self._dim)
            self._covariance_sqrt = np.eye(self._dim)
            self._pc *= 0
            self._ps *= 0
            
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop with batched evaluation."""
        self._initialize_population()
        
        self._fitness = func(self._population)
        if len(self._fitness) < self._np:
            self._fitness = np.pad(self._fitness, (0, self._np - len(self._fitness)), constant_values=np.inf)
        
        self._best_fitness = self._fitness.min()
        self._best_x = self._population[self._fitness.argmin()].copy()
        
        while not stopping_condition():
            self._generation += 1
            
            trials = self._sample_trials_batch()
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if len(trial_fitness) == 0:
                break
            
            if stopping_condition():
                break
            
            improved_mask = trial_fitness < self._fitness[:len(trial_fitness)]
            n_improved = np.sum(improved_mask)
            
            self._population, self._fitness, _ = self._select_survivors_batch(trials, trial_fitness)
            
            self._mean = self._population.mean(axis=0)
            
            self._adapt_step_size(n_improved)
            
            if self._generation % 3 == 0:
                self._adapt_covariance_variant_01(improved_mask)
            if self._generation % 5 == 0:
                self._adapt_covariance_variant_08(improved_mask)
            if self._generation % 7 == 0:
                self._adapt_covariance_variant_09(improved_mask)
            
            current_best = self._fitness.min()
            if current_best < self._best_fitness:
                self._best_fitness = current_best
                self._best_x = self._population[self._fitness.argmin()].copy()
            
            self._restart_if_stagnant(current_best)
            
            self._history['fitness'].append(self._best_fitness)
            self._history['diversity'].append(self._compute_diversity())
            self._history['step_size'].append(self._step_size)
        
        return self._best_fitness, self._best_x
```

```python
import numpy as np


class AdaptiveCovarianceMatrixSearchWithOperatorPool:
    """
    A CMA-ES-inspired optimizer with an adaptive pool of mutation operators.
    Operators compete based on success history, and covariance adaptation
    uses multiple update strategies that can be tuned independently.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self._dim = dim
        self._np = min(150, max(4 * dim, 80))
        self._lower = -100.0
        self._upper = 100.0
        
        self._mu_eff = self._np / 10
        self._mu_cov = self._np / 10
        
        self._step_size = 0.5 * (self._upper - self._lower)
        self._chi_n = np.sqrt(self._dim) * (1 - 1 / (4 * self._dim) + 1 / (21 * self._dim ** 2))
        
        self._cc = 2 / (dim + 7)
        self._cs = (self._mu_eff + 2) / (dim + self._mu_eff + 3)
        self._c1 = 2 / ((dim + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(1 - self._c1, 2 * self._mu_cov / ((dim + 2) ** 2 + 2 * self._mu_cov))
        self._damps = 1 + 2 * max(0, np.sqrt(self._mu_eff) - 1) / self._cs
        
        self._n_operators = 4
        self._operator_weights = np.ones(self._n_operators) / self._n_operators
        self._operator_rewards = np.zeros(self._n_operators)
        self._operator_counts = np.zeros(self._n_operators)
        
        self._pc = np.zeros(dim)
        self._ps = np.zeros(dim)
        self._generation = 0
        self._best_fitness = np.inf
        self._best_x = None
        self._no_improvement_count = 0
        
        self._stagnation_threshold = max(50, int(5 * dim))
        self._history = {'fitness': [], 'diversity': [], 'step_size': []}
        
        self._active_operator = None
        self._active_z = None
        self._z_stack = None
        
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        pop = np.random.uniform(self._lower, self._upper, (self._np, self._dim))
        pop = np.clip(pop, self._lower, self._upper)
        self._population = pop
        self._fitness = np.full(self._np, np.inf)
        self._mean = pop.mean(axis=0)
        self._covariance = np.eye(self._dim)
        self._covariance_sqrt = np.eye(self._dim)
        
    def _sample_trials_batch(self):
        """Sample trial solutions using current covariance and operator weights."""
        self._z_stack = np.random.randn(self._np, self._n_operators, self._dim)
        
        z_global = self._z_stack[:, 0, :]
        trials_global = self._mean + self._step_size * (self._covariance_sqrt @ z_global.T).T
        trials_global = np.clip(trials_global, self._lower, self._upper)
        
        idx_r1 = np.random.randint(0, self._np, size=self._np)
        idx_r2 = np.random.randint(0, self._np, size=self._np)
        idx_r3 = np.random.randint(0, self._np, size=self._np)
        diff_idx = idx_r2 != idx_r3
        n_diff = np.sum(diff_idx)
        
        trials_de = np.copy(trials_global)
        if n_diff > 0:
            diff_mask = np.zeros(self._np, dtype=bool)
            diff_mask[:n_diff] = diff_idx
            trials_de[diff_mask] = self._population[idx_r1[diff_mask]] + self._step_size * 0.5 * (
                self._population[idx_r2[diff_mask]] - self._population[idx_r3[diff_mask]]
            )
            trials_de[diff_mask] = np.clip(trials_de[diff_mask], self._lower, self._upper)
        
        z_best = self._z_stack[:, 2, :]
        best_dir = 0.3 * (self._best_x - self._mean) if self._best_x is not None else np.zeros(self._dim)
        trials_best = self._mean + self._step_size * (best_dir + 0.7 * (self._covariance_sqrt @ z_best.T).T)
        trials_best = np.clip(trials_best, self._lower, self._upper)
        
        z_iso = self._z_stack[:, 3, :]
        trials_iso = self._mean + self._step_size * 0.8 * (self._covariance_sqrt @ z_iso.T).T
        trials_iso = np.clip(trials_iso, self._lower, self._upper)
        
        trials_stack = np.stack([trials_global, trials_de, trials_best, trials_iso], axis=1)
        op_indices = np.random.choice(self._n_operators, size=self._np, p=self._operator_weights)
        batch_idx = np.arange(self._np)
        selected_trials = trials_stack[batch_idx, op_indices]
        
        self._active_operator = op_indices
        self._active_z = self._z_stack[batch_idx, op_indices]
        
        return selected_trials
    
    def _select_survivors_batch(self, trials, trial_fitness):
        """Select survivors using comma strategy with operator reward tracking."""
        n_valid = min(len(trial_fitness), len(trials))
        if n_valid < len(trials):
            trials = trials[:n_valid]
            trial_fitness = trial_fitness[:n_valid]
        
        improved = trial_fitness < self._fitness[:n_valid]
        n_improved = np.sum(improved)
        
        op_success_counts = np.zeros(self._n_operators)
        if n_improved > 0:
            improved_idx = np.where(improved)[0]
            success_ops = self._active_operator[improved_idx]
            for op_idx in range(self._n_operators):
                op_success_counts[op_idx] = np.sum(success_ops == op_idx)
            
            self._population[improved_idx] = trials[improved_idx]
            self._fitness[improved_idx] = trial_fitness[improved_idx]
        
        self._operator_counts += op_success_counts
        self._update_operator_rewards(success_counts=op_success_counts)
        
        return self._population.copy(), self._fitness.copy(), n_improved
    
    def _update_operator_rewards(self, success_counts):
        """Update operator weights based on success counts using exponential moving average."""
        total_counts = self._operator_counts + 1e-10
        success_rate = success_counts / total_counts
        
        alpha = 0.1
        self._operator_rewards = (1 - alpha) * self._operator_rewards + alpha * success_rate
        
        max_reward = self._operator_rewards.max()
        exp_weights = np.exp(0.3 * (self._operator_rewards - max_reward))
        self._operator_weights = exp_weights / (exp_weights.sum() + 1e-10)
        
        min_weight = 0.05
        self._operator_weights = np.maximum(self._operator_weights, min_weight)
        self._operator_weights /= self._operator_weights.sum()
    
    def _adapt_step_size(self, n_improved):
        """Adapt step size based on success rate of recent trials."""
        success_rate = n_improved / self._np
        target_rate = 0.2
        
        if success_rate > target_rate:
            self._step_size *= 1 + self._cs * (success_rate - target_rate) / (1 - target_rate)
        else:
            self._step_size *= 1 - self._cs * (target_rate - success_rate) / target_rate
        
        self._step_size = np.clip(self._step_size, 1e-10, self._upper - self._lower)
    
    def _adapt_covariance_variant_01(self, improved_mask):
        """Rank-1 covariance update using evolution path (simplified CMA-ES)."""
        z_mean = self._active_z[improved_mask].mean(axis=0) if np.any(improved_mask) else np.zeros(self._dim)
        
        self._ps = (1 - self._cs) * self._ps + np.sqrt(self._cs * (2 - self._cs) * self._mu_eff) * z_mean
        
        hs = np.linalg.norm(self._ps) / np.sqrt(1 - (1 - self._cs) ** (2 * self._generation)) < 1.5 + 1 / (self._dim + 1)
        self._pc = (1 - self._cc) * self._pc + hs * np.sqrt(self._cc * (2 - self._cc) * self._mu_eff) * z_mean
        
        rank_one = np.outer(self._pc, self._pc)
        self._covariance = (1 - self._c1 - self._c1 * self._cmu) * self._covariance + self._c1 * rank_one
        
        self._eigen_decompose_covariance()
    
    def _adapt_covariance_variant_08(self, improved_mask):
        """Rank-μ covariance update using top-performing individuals."""
        if not np.any(improved_mask):
            return
        
        n_select = min(20, np.sum(improved_mask))
        top_idx = np.argsort(self._fitness)[:n_select]
        z_top = self._active_z[top_idx]
        
        rank_mu = np.zeros((self._dim, self._dim))
        for i in range(n_select):
            rank_mu += np.outer(z_top[i], z_top[i])
        rank_mu /= n_select
        
        self._covariance = (1 - self._cmu) * self._covariance + self._cmu * rank_mu
        
        self._eigen_decompose_covariance()
    
    def _adapt_covariance_variant_09(self, improved_mask):
        """Combined rank-1 and rank-μ update with weighted contribution."""
        z_mean = self._active_z[improved_mask].mean(axis=0) if np.any(improved_mask) else np.zeros(self._dim)
        
        rank_one = np.outer(z_mean, z_mean)
        
        if np.any(improved_mask):
            n_top = min(self._mu_cov, np.sum(improved_mask))
            top_idx = np.argsort(self._fitness)[:n_top]
            z_top = self._active_z[top_idx]
            
            rank_mu = np.zeros((self._dim, self._dim))
            for i in range(n_top):
                rank_mu += np.outer(z_top[i], z_top[i])
            rank_mu /= n_top
        else:
            rank_mu = np.zeros((self._dim, self._dim))
        
        self._covariance = (1 - self._c1 - self._cmu) * self._covariance + self._c1 * rank_one + self._cmu * rank_mu
        
        self._eigen_decompose_covariance()
    
    def _eigen_decompose_covariance(self):
        """Ensure covariance matrix stays positive definite via eigendecomposition."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            self._covariance_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        except np.linalg.LinAlgError:
            self._covariance = np.eye(self._dim)
            self._covariance_sqrt = np.eye(self._dim)
    
    def _compute_diversity(self):
        """Compute population diversity as mean pairwise Euclidean distance."""
        if self._np < 2:
            return 0.0
        dist_matrix = np.linalg.norm(self._population[:, None] - self._population[None, :], axis=2)
        diversity = dist_matrix[np.triu_indices(self._np, k=1)].mean()
        return diversity
    
    def _restart_if_stagnant(self, current_best):
        """Restart population if no improvement for threshold generations."""
        if current_best >= self._best_fitness:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        
        if self._no_improvement_count >= self._stagnation_threshold:
            self._no_improvement_count = 0
            self._step_size = 0.5 * (self._upper - self._lower)
            
            n_keep = max(2, self._np // 5)
            keep_indices = np.argsort(self._fitness)[:n_keep]
            kept_pop = self._population[keep_indices].copy()
            kept_fit = self._fitness[keep_indices].copy()
            
            new_pop = np.random.uniform(self._lower, self._upper, (self._np - n_keep, self._dim))
            new_pop = np.clip(new_pop, self._lower, self._upper)
            
            self._population = np.vstack([kept_pop, new_pop])
            self._fitness = np.concatenate([kept_fit, np.full(self._np - n_keep, np.inf)])
            self._mean = self._population.mean(axis=0)
            
            self._covariance = np.eye(self._dim)
            self._covariance_sqrt = np.eye(self._dim)
            self._pc *= 0
            self._ps *= 0
            
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop with batched evaluation."""
        self._initialize_population()
        
        self._fitness = func(self._population)
        if len(self._fitness) < self._np:
            self._fitness = np.pad(self._fitness, (0, self._np - len(self._fitness)), constant_values=np.inf)
        
        self._best_fitness = self._fitness.min()
        self._best_x = self._population[self._fitness.argmin()].copy()
        
        while not stopping_condition():
            self._generation += 1
            
            trials = self._sample_trials_batch()
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if len(trial_fitness) == 0:
                break
            
            if stopping_condition():
                break
            
            improved_mask = trial_fitness < self._fitness[:len(trial_fitness)]
            n_improved = np.sum(improved_mask)
            
            self._population, self._fitness, _ = self._select_survivors_batch(trials, trial_fitness)
            
            self._mean = self._population.mean(axis=0)
            
            self._adapt_step_size(n_improved)
            
            if self._generation % 3 == 0:
                self._adapt_covariance_variant_01(improved_mask)
            if self._generation % 5 == 0:
                self._adapt_covariance_variant_08(improved_mask)
            if self._generation % 7 == 0:
                self._adapt_covariance_variant_09(improved_mask)
            
            current_best = self._fitness.min()
            if current_best < self._best_fitness:
                self._best_fitness = current_best
                self._best_x = self._population[self._fitness.argmin()].copy()
            
            self._restart_if_stagnant(current_best)
            
            self._history['fitness'].append(self._best_fitness)
            self._history['diversity'].append(self._compute_diversity())
            self._history['step_size'].append(self._step_size)
        
        return self._best_fitness, self._best_x
```