```python
import numpy as np


class AdaptiveSurvivorSelectionOptimizer:
    """
    Adaptive optimizer that automatically selects the best _select_survivors_batch
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 4 survivor selection strategies (original, variant_05, variant_07, variant_09)
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
        
        # Adaptive survivor selection parameters
        self.num_survivor_operators = 4
        self.operator_names = ['original', 'variant_05', 'variant_07', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_survivor_operators)
        self.beta = np.ones(self.num_survivor_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_survivor_operators)}
        self.operator_counts = np.zeros(self.num_survivor_operators)
        self.selection_counts = np.zeros(self.num_survivor_operators)
        
        # Per-operator fitness tracking for credit assignment
        self.operator_fitness_history = {i: [] for i in range(self.num_survivor_operators)}
        
        # Runtime state
        self.generation = 0
        self.current_survivor_operator = 0
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
            
            # Select and apply survivor selection operator
            self._select_survivor_operator_thompson()
            self._select_survivors_batch()
            
            # Update operator rewards based on improvement
            self._update_survivor_operator_rewards()
            
            self._adapt_step_size()
            self._adapt_covariance()
            
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
        self.fitness = np.asarray(self.fitness).flatten().astype(float)
        
        best_idx = np.argmin(self.fitness)
        f_opt_val = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        if np.isfinite(f_opt_val):
            self.f_opt = f_opt_val
        else:
            self.f_opt = 1e100
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_survivor_operator = 0
        
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
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
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
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
        self.trial_fitness = np.asarray(self.trial_fitness).flatten().astype(float)
        # Replace any inf or nan with large value
        self.trial_fitness = np.where(np.isfinite(self.trial_fitness), self.trial_fitness, 1e100)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt and np.isfinite(trial_best_fit):
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivor_operator_thompson(self):
        """Select survivor operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_survivor_operator = int(np.argmax(samples))
        self.operator_counts[self.current_survivor_operator] += 1
        return self.current_survivor_operator
    
    def _update_survivor_operator_rewards(self):
        """Update survivor operator rewards using sliding window credit assignment."""
        # Calculate improvement from this generation
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improvement = float(np.clip(improvement, 0.0, 1e20))
        
        # Calculate diversity of current population
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        # Calculate fitness spread (normalized)
        fit_min = float(np.min(self.fitness))
        fit_max = float(np.max(self.fitness))
        fit_spread = float(np.clip((fit_max - fit_min) / (abs(fit_min) + 1e-10), 0.0, 10.0))
        
        # Combined reward: improvement + diversity bonus + spread bonus
        reward = float(np.log1p(improvement * 1e10) / 10.0)
        reward += 0.15 * diversity
        reward += 0.05 * min(fit_spread, 1.0)
        reward = float(np.clip(reward, -5.0, 10.0))
        
        op = self.current_survivor_operator
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Update Beta distribution parameters using empirical Bayes
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            # Clamp mean to avoid numerical issues
            mean_reward = float(np.clip(mean_reward, 0.01, 0.99))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = float(max(1.0, 1.0 + sum_reward))
            else:
                self.beta[op] = float(max(1.0, 1.0 - sum_reward))
        
        # Track fitness history for this operator
        self.operator_fitness_history[op].append(float(self.f_opt))
        if len(self.operator_fitness_history[op]) > self.reward_window_size:
            self.operator_fitness_history[op].pop(0)
    
    def _select_survivors_batch(self):
        """Dispatch to the selected survivor selection strategy."""
        if self.current_survivor_operator == 0:
            self._select_survivors_original()
        elif self.current_survivor_operator == 1:
            self._select_survivors_variant_05()
        elif self.current_survivor_operator == 2:
            self._select_survivors_variant_07()
        else:
            self._select_survivors_variant_09()
    
    def _select_survivors_original(self):
        """Select survivors via elitist (mu, lambda)-selection (original baseline)."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        # Ensure all fitness values are valid
        valid_mask = np.isfinite(combined_fit)
        if not np.all(valid_mask):
            combined_fit = np.where(valid_mask, combined_fit, 1e100)
        
        sorted_indices = np.argsort(combined_fit)
        selected_indices = sorted_indices[:self.NP]
        
        self.population = combined_pop[selected_indices]
        self.fitness = combined_fit[selected_indices]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _select_survivors_variant_05(self):
        """Select survivors via diversity-penalized fitness ranking."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        n_combined = len(combined_fit)
        
        # Ensure valid fitness values
        valid_mask = np.isfinite(combined_fit)
        if not np.all(valid_mask):
            combined_fit = np.where(valid_mask, combined_fit, 1e100)

        if n_combined <= self.NP:
            self.population = combined_pop[:self.NP]
            self.fitness = combined_fit[:self.NP]
            self.old_mean = self.mean.copy()
            self.mean = np.mean(self.population, axis=0)
            self.generation += 1
            return

        # Exponential ranking of raw fitness
        sorted_fit_indices = np.argsort(combined_fit)
        ranks = np.zeros(n_combined)
        ranks[sorted_fit_indices] = np.arange(1, n_combined + 1)
        exp_fitness = np.exp(-0.01 * ranks)

        # Diversity factor: inverse distance to nearest neighbor in population
        diversity = np.ones(n_combined)
        pop_range = float(np.linalg.norm(self.ub - self.lb))
        
        for i in range(n_combined):
            dists = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
            dists[i] = np.inf
            if np.isfinite(dists).any():
                nn_dist = float(np.min(dists))
                diversity[i] = 1.0 + 5.0 * (nn_dist / (pop_range + 1e-10))

        diversity = np.clip(diversity, 1.0, 10.0)

        # Composite score: balance fitness (exponential rank) with diversity
        composite = exp_fitness * diversity

        # Stochastic selection weighted by composite score
        total_composite = float(np.sum(composite))
        if total_composite <= 0 or not np.isfinite(total_composite):
            # Fallback to deterministic selection
            sorted_indices = np.argsort(combined_fit)
            self.population = combined_pop[sorted_indices[:self.NP]]
            self.fitness = combined_fit[sorted_indices[:self.NP]]
        else:
            probs = composite / total_composite
            probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
            probs = probs / np.sum(probs)

            selected_indices = np.random.choice(n_combined, size=self.NP, replace=False, p=probs)

            self.population = combined_pop[selected_indices]
            self.fitness = combined_fit[selected_indices]

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _select_survivors_variant_07(self):
        """Select survivors via fitness-distance crowding to preserve diversity."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        n_parents = len(self.population)
        n_trials = len(self.trials)

        # Ensure valid fitness values
        valid_mask = np.isfinite(combined_fit)
        if not np.all(valid_mask):
            combined_fit = np.where(valid_mask, combined_fit, 1e100)

        # Compute pairwise Euclidean distances between trials and parents
        diff = self.trials[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))

        # For each trial, find the nearest parent (crowding partner)
        nearest_parent = np.argmin(dists, axis=1)
        nearest_dist = np.min(dists, axis=1)

        # Initialize replacement array: -1 means not replaced yet
        replacements = np.full(n_trials, -1, dtype=int)

        # Track which parents are claimed
        parent_claimed = np.zeros(n_parents, dtype=bool)

        # Sort trials by fitness (best first) to give better solutions priority
        trial_order = np.argsort(self.trial_fitness)

        for ti in trial_order:
            pi = nearest_parent[ti]
            if not parent_claimed[pi]:
                # Trial beats or ties parent in fitness - claim this parent slot
                if self.trial_fitness[ti] <= self.fitness[pi] + 1e-15:
                    replacements[ti] = pi
                    parent_claimed[pi] = True

        # Build new population
        new_pop = self.population.copy()
        new_fit = self.fitness.copy()

        for ti in range(n_trials):
            if replacements[ti] >= 0:
                pi = replacements[ti]
                new_pop[pi] = self.trials[ti]
                new_fit[pi] = self.trial_fitness[ti]

        # Handle unclaimed parents: keep best unclaimed individuals from combined pool
        unclaimed_mask = ~parent_claimed
        unclaimed_indices = np.where(unclaimed_mask)[0]

        if len(unclaimed_indices) > 0:
            # Find best individuals from combined pool excluding already-selected
            selected_mask = np.zeros(len(combined_fit), dtype=bool)
            for pi in np.where(parent_claimed)[0]:
                selected_mask[pi] = True
            for ti in np.where(replacements >= 0)[0]:
                selected_mask[n_parents + ti] = True

            remaining_pool = np.where(~selected_mask)[0]
            if len(remaining_pool) > 0:
                remaining_fits = combined_fit[remaining_pool]
                best_remaining = np.argsort(remaining_fits)
                fill_count = min(len(unclaimed_indices), len(best_remaining))
                for k in range(fill_count):
                    idx = remaining_pool[best_remaining[k]]
                    if idx < n_parents:
                        new_pop[unclaimed_indices[k]] = combined_pop[idx]
                        new_fit[unclaimed_indices[k]] = combined_fit[idx]

        self.population = new_pop
        self.fitness = new_fit
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _select_survivors_variant_09(self):
        """Select survivors using fitness-sharing niching to preserve diversity."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        n_combined = len(combined_fit)

        # Ensure valid fitness values
        valid_mask = np.isfinite(combined_fit)
        if not np.all(valid_mask):
            combined_fit = np.where(valid_mask, combined_fit, 1e100)

        sigma_share = 0.1 * (self.ub[0] - self.lb[0]) / self.NP

        niche_counts = np.ones(n_combined)
        for i in range(n_combined):
            distances = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
            niche_counts[i] = float(np.sum(distances < sigma_share)) + 1e-10

        shared_fitness = combined_fit * niche_counts

        sorted_indices = np.argsort(shared_fitness)
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
                total_improvement += max(float(diff), 0.0)
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(recent_improved / recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        avg_improvement = total_improvement / max(float(recent_improved), 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = float(0.7 * self.improvement_ema + 0.3 * improvement_magnitude)

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * float(np.tanh(self.improvement_ema - 1.0))
        adaptation = float(np.clip(adaptation, -0.8, 0.8))

        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = float(np.clip(diversity_boost, 0.5, 3.0))

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 200.0))

        self.sigma *= float(np.exp(adaptation * self.cs / damping_adaptive))
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _adapt_covariance(self):
        """Original covariance adaptation."""
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
        return float(np.mean(np.std(self.population, axis=0)))
    
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
            elite_fit = float(self.fitness[best_idx])
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            
            f_opt_val = float(elite_fit)
            if np.isfinite(f_opt_val):
                self.f_opt = f_opt_val
            else:
                self.f_opt = 1e100
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```