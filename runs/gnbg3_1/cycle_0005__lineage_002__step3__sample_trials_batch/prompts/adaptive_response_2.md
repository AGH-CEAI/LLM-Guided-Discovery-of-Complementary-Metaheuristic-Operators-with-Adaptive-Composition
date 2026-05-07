```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best sampling strategy
    during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 3 covariance sampling strategies (original, variant_01, variant_03)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    - Robust handling of all edge cases
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
        self.num_operators = 3
        self.operator_names = ['original', 'variant_01', 'variant_03']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Performance tracking for adaptive selection
        self.operator_fitness_history = {i: [] for i in range(self.num_operators)}
        self.generation_fitness = []
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        self.stagnation_counter = 0
        self.f_opt_prev = float('inf')
    
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
            # Select sampling strategy using Thompson Sampling
            self._select_operator_thompson()
            
            # Sample using the selected strategy
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
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
            
            # Periodically boost underperforming operators
            if self.generation > 0 and self.generation % 20 == 0:
                self._boost_underperforming_operators()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        if np.isnan(self.sigma) or self.sigma < 1e-10:
            self.sigma = (self.ub[0] - self.lb[0]) / 10.0
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        self.fitness = np.array(self.fitness, dtype=np.float64).flatten()
        self.fitness = np.clip(self.fitness, -1e100, 1e100)
        
        best_idx = np.argmin(self.fitness)
        f_best = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        if not np.isfinite(f_best):
            f_best = 1e100
        self.f_opt = f_best
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
        self.L = None
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
        if hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using the selected strategy."""
        if self.current_operator == 0:
            self._sample_trials_original()
        elif self.current_operator == 1:
            self._sample_trials_variant_01()
        else:  # current_operator == 2
            self._sample_trials_variant_03()
    
    def _sample_trials_original(self):
        """Sample new trial population using Cholesky decomposition (original strategy)."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        # Sample from standard normal and transform
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_trials_variant_01(self):
        """Sample using eigendecomposition with condition number bounding (variant_01 strategy)."""
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Eigendecomposition: C = V @ D @ V.T
        eigvals, eigvecs = np.linalg.eigh(self.C)

        # Robustness: ensure positive eigenvalues and bound condition number
        eigvals = np.maximum(eigvals, 1e-10)
        max_eig = np.max(eigvals)
        if max_eig > 1e6 * np.min(eigvals):
            eigvals = np.clip(eigvals, np.min(eigvals), 1e6 * np.min(eigvals))

        # Sample from standard normal and transform: mean + sigma * V @ sqrt(D) @ z
        z = np.random.randn(self.dim, self.NP)
        D_sqrt = np.sqrt(eigvals)
        self.trials = self.mean[:, np.newaxis] + self.sigma * (eigvecs * D_sqrt) @ z
        self.trials = self.trials.T

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_trials_variant_03(self):
        """Sample with eigenvalue-weighted diversity and stagnation-triggered mixing (variant_03 strategy)."""
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Eigendecomposition for eigenvalue-weighted sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)

        # Detect stagnation: high condition number means elongated covariance
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1000:
            # Mix with identity to isotropically expand exploration
            mix = min(0.5, (cond - 1000) / 10000)
            C_sampled = (1 - mix) * self.C + mix * np.mean(eigvals) * np.eye(self.dim)
            eigvals_s, eigvecs = np.linalg.eigh(C_sampled)
            eigvals_s = np.maximum(eigvals_s, 1e-10)
        else:
            eigvals_s, eigvecs = eigvals, eigvecs

        # Eigenvalue-weighted sampling: sample more in smaller eigenvalue directions
        # Use power-law weights: smaller eigvals get higher probability
        weights = 1.0 / (eigvals_s ** 0.25 + 1e-10)
        weights /= np.sum(weights)

        # Sample eigenvalue indices for each individual
        n_sample = max(1, int(0.1 * self.NP))
        sampled_eigen_indices = np.random.choice(self.dim, size=n_sample, p=weights, replace=True)

        # Standard normal samples
        z = np.random.randn(self.NP, self.dim)

        # Apply eigenvalue scaling: scale each direction by sqrt(eigenvalue)
        sqrt_eigvals = np.sqrt(eigvals_s)
        scales = np.ones(self.dim)
        for idx in sampled_eigen_indices:
            scales[idx] *= 1.5
        scales = np.clip(scales, 0.5, 2.0)

        # Transform samples: z * sqrt(eigvals) * scales
        transformed = z * (sqrt_eigvals * scales)

        # Apply to mean
        self.trials = self.mean + self.sigma * (transformed @ eigvecs.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)

        # Stagnation detection for restarts
        if self.stagnation_counter > self.max_stagnation // 2:
            # Increase sigma to escape local optima
            self.sigma = min(self.sigma * 2.0, 10.0)
            # Reinitialize poorly-performing individuals
            n_bad = min(self.NP // 2, len(self.population))
            if n_bad > 0:
                bad_indices = np.argsort(self.fitness)[-n_bad:]
                for idx in bad_indices:
                    if idx > 0:  # Keep best individual
                        self.population[idx] = np.random.uniform(self.lb, self.ub)
                        self.fitness[idx] = float('inf')
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
        self.trial_fitness = np.array(self.trial_fitness, dtype=np.float64).flatten()
        self.trial_fitness = np.clip(self.trial_fitness, -1e100, 1e100)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivors_batch(self):
        """Select survivors via elitist (mu, lambda)-selection."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
                diff = self.fitness[i] - self.trial_fitness[i]
                total_improvement += max(diff, 0.0)
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Compute reward based on improvement and diversity
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Track fitness history for this operator
        self.operator_fitness_history[op].append(self.f_opt)
        if len(self.operator_fitness_history[op]) > self.reward_window_size * 2:
            self.operator_fitness_history[op].pop(0)
        
        # Update Beta distribution parameters
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
        
        self.f_opt_prev = self.f_opt
    
    def _boost_underperforming_operators(self):
        """Boost underperforming operators to maintain exploration."""
        total_selections = np.sum(self.selection_counts)
        if total_selections < 10:
            return
        
        # Check if any operator has very low selection rate
        for op in range(self.num_operators):
            selection_rate = self.selection_counts[op] / total_selections
            if selection_rate < 0.1 and self.selection_counts[op] > 5:
                # Boost this operator's alpha to increase its selection probability
                self.alpha[op] = min(self.alpha[op] * 1.5, 100.0)
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        else:  # current_operator == 2
            self._adapt_covariance_variant_03()
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_01(self):
        """Cholesky factor update - maintains C = LL^T directly."""
        if self.L is None:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        z_1 = y_mean / np.maximum(np.linalg.norm(y_mean), 1e-10)
        
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += w
        
        if total_weight > 0:
            rank_mu /= total_weight
        
        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        
        C_new = (1.0 - ccov_1 - ccov_mu) * self.C
        C_new += ccov_1 * np.outer(z_1, z_1)
        C_new += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])
        
        self.C = self._ensure_positive_definite(C_new)
        
        try:
            v = np.linalg.solve(self.L.T, z_1)
            alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
            self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
    
    def _adapt_covariance_variant_03(self):
        """Variant_03 covariance adaptation with eigenvalue weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Adaptive learning rate based on condition number
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.min(eigvals + 1e-10)
        
        if cond > 1000:
            # Reduce learning rate when ill-conditioned
            cc_adaptive = self.cc * 0.5
            ccov_adaptive = self.ccov * 0.5
        else:
            cc_adaptive = self.cc
            ccov_adaptive = self.ccov
        
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        # Eigenvalue-weighted rank-mu update
        sqrt_eigvals = np.sqrt(eigvals)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            # Weight by inverse condition contribution
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
        
        # Additional diversity injection if needed
        if cond > 1e6:
            self.C += 0.01 * np.mean(eigvals) * np.eye(self.dim)
    
    def _compute_diversity(self):
        """Compute population diversity as mean per-dimension std."""
        return np.mean(np.std(self.population, axis=0))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            
            f_elite = float(np.asarray(elite_fit).flatten()[0])
            if np.isfinite(f_elite):
                self.f_opt = f_elite
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```