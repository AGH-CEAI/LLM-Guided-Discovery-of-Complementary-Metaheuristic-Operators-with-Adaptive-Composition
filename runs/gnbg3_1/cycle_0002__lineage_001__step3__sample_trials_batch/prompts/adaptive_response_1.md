```python
import numpy as np


class AdaptiveSamplingCovarianceEvolution:
    """
    Adaptive optimizer that automatically selects the best sampling strategy
    during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 7 sampling strategies (original + 6 variants from benchmark)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
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
        self.num_operators = 7
        self.operator_names = [
            'original', 'variant_01', 'variant_03', 'variant_04',
            'variant_05', 'variant_06', 'variant_09'
        ]
        
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
        self.trials = None
        self.trial_fitness = None
        self.population = None
        self.fitness = None
    
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
            # Select sampling operator using Thompson Sampling
            self._select_operator_thompson()
            
            # Sample trials using the selected operator
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
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
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
        if hasattr(self, 'in_explosion'):
            self.in_explosion = False
        if hasattr(self, 'explosion_gen'):
            self.explosion_gen = 0
        if hasattr(self, 'sigma_backup'):
            self.sigma_backup = None
        if hasattr(self, '_best_at_last_improvement'):
            self._best_at_last_improvement = self.f_opt
            self._gen_at_last_improvement = self.generation
        if hasattr(self, 'success_directions'):
            self.success_directions = []
    
    def _sample_trials_batch(self):
        """Dispatch to the selected sampling strategy."""
        if self.current_operator == 0:
            self._sample_original()
        elif self.current_operator == 1:
            self._sample_variant_01()
        elif self.current_operator == 2:
            self._sample_variant_03()
        elif self.current_operator == 3:
            self._sample_variant_04()
        elif self.current_operator == 4:
            self._sample_variant_05()
        elif self.current_operator == 5:
            self._sample_variant_06()
        else:
            self._sample_variant_09()
    
    def _sample_original(self):
        """Standard CMA-ES sampling using eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_variant_01(self):
        """Sample new trial population using Cholesky decomposition."""
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_variant_03(self):
        """Sample trials with restart-triggered hybrid global+local exploration."""
        is_stagnating = self.stagnation_counter > self.max_stagnation // 3

        if is_stagnating:
            n_global = int(0.4 * self.NP)
            n_local = self.NP - n_global

            global_samples = np.random.uniform(self.lb, self.ub, (n_global, self.dim))

            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(n_local, self.dim)
            local_samples = self.mean + 2.5 * self.sigma * (z @ L.T)

            self.trials = np.vstack([global_samples, local_samples])
        else:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * (z @ L.T)

        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_variant_04(self):
        """Sample with stagnation-triggered heavy-tailed exploration."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        if not hasattr(self, '_best_at_last_improvement'):
            self._best_at_last_improvement = self.f_opt
            self._gen_at_last_improvement = self.generation

        gens_since_best = self.generation - self._gen_at_last_improvement

        if self.f_opt < self._best_at_last_improvement - 1e-14:
            self._best_at_last_improvement = self.f_opt
            self._gen_at_last_improvement = self.generation
            gens_since_best = 0

        stagnation_threshold = max(8, self.dim // 2)

        if gens_since_best >= stagnation_threshold:
            df = max(2.0, self.dim / 5.0)
            z_raw = np.random.standard_t(df, size=(self.NP, self.dim))

            row_norms = np.linalg.norm(z_raw, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-8)
            z_normalized = z_raw / row_norms
            jump_factors = np.random.uniform(1.5, 4.0, size=(self.NP, 1))
            z_explore = z_normalized * jump_factors

            sigma_explore = self.sigma * 3.0
            self.trials = self.mean + sigma_explore * (z_explore @ L.T)
        else:
            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * (z @ L.T)

        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_variant_05(self):
        """Sample with pulsed explosive exploration on stagnation."""
        is_stagnant = (self.stagnation_counter > self.max_stagnation // 2)

        if not hasattr(self, 'in_explosion'):
            self.in_explosion = False
            self.explosion_gen = 0
            self.sigma_backup = None

        if is_stagnant and not self.in_explosion:
            self.in_explosion = True
            self.explosion_gen = 0
            self.sigma_backup = self.sigma
            self.sigma = min(self.sigma * 15.0, (self.ub[0] - self.lb[0]) * 0.4)
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
            self.C = 0.5 * np.eye(self.dim) + 0.5 * np.diag(np.diag(self.C))

        if self.in_explosion:
            self.explosion_gen += 1
            explosion_decay = max(0.05, 1.0 - self.explosion_gen / 20.0)

            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * explosion_decay * (z @ L.T)
            self.trials = self._clip_to_bounds(self.trials)

            if self.explosion_gen >= 20 and self.sigma_backup is not None:
                self.in_explosion = False
                self.sigma = self.sigma_backup
                self.explosion_gen = 0
                self.sigma_backup = None
        else:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            z = np.random.randn(self.NP, self.dim)
            self.trials = self.mean + self.sigma * (z @ L.T)
            self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_variant_06(self):
        """Sample with covariance restart on degeneracy detection."""
        eigvals, eigvecs = np.linalg.eigh(self.C)

        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        cond_num = max_eig / max(min_eig, 1e-30)

        if cond_num > 1000.0 * self.dim:
            self.C = self.sigma * np.eye(self.dim)
            eigvals, eigvecs = np.linalg.eigh(self.C)

        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_variant_09(self):
        """Adaptive multi-component hybrid sampling."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        z_cov = np.random.randn(self.NP, self.dim)
        samples_cov = z_cov @ L_cov.T

        z_iso = np.random.randn(self.NP, self.dim)
        samples_iso = z_iso

        if hasattr(self, 'success_directions') and len(self.success_directions) > 0:
            stacked = np.array(self.success_directions)
            stacked = stacked - np.mean(stacked, axis=0)
            Q, _ = np.linalg.qr(stacked.T)
            z_hist = np.random.randn(self.NP, min(self.dim, len(self.success_directions)))
            samples_hist = z_hist @ Q[:len(self.success_directions), :].T
        else:
            samples_hist = np.zeros((self.NP, self.dim))

        pop_var = np.mean(np.var(self.population, axis=0)) if hasattr(self, 'population') and len(self.population) > 1 else 1.0
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_var / (expected_var + 1e-10), 0.01, 1.0)

        iso_ratio = 0.3 * (1.0 - diversity) + 0.1
        iso_ratio = np.clip(iso_ratio, 0.05, 0.5)

        hist_ratio = 0.2 * diversity + 0.05
        hist_ratio = np.clip(hist_ratio, 0.05, 0.3)

        cov_ratio = max(0.0, 1.0 - iso_ratio - hist_ratio)

        combined = (cov_ratio * samples_cov + 
                    iso_ratio * samples_iso + 
                    hist_ratio * samples_hist)

        self.trials = self.mean + self.sigma * combined
        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
            
            # Track successful directions for variant_09
            if not hasattr(self, 'success_directions'):
                self.success_directions = []
            best_trial = self.trials[trial_best_idx]
            direction = best_trial - self.mean
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                self.success_directions.append(direction / dir_norm)
                if len(self.success_directions) > 50:
                    self.success_directions.pop(0)
    
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
        """Adapt step-size using short-term success history."""
        recent_improved = 0
        recent_total = 0

        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation = np.clip(adaptation, -0.5, 0.5)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim))
        damping_adaptive = np.clip(damping_adaptive, 0.1, 100.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
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
    
    def _adapt_covariance(self):
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
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```