Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_survivors_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.314496e-02            -inf                    5.534968e-02            5.409962e+00            -inf                    5.523219e-02            5.178898e-02            2.531228e+00            5.936088e-02            2.803355e+00            
1      4.447001e-02            -inf                    4.386648e-02            9.384448e+00            -inf                    3.549066e-02            3.784888e-02            3.076981e+01            4.711332e-02            5.794472e+00            
2      2.877613e-01            -inf                    3.899730e+00            5.114669e+01            -inf                    8.219267e-01            4.479662e+00            2.960315e+00            3.486829e+00            2.779969e+01            
3      1.347502e-01            -inf                    3.200944e+00            4.448625e+01            -inf                    2.356202e+00            2.810669e+00            8.759404e+01            3.061982e+00            3.854093e+01            
4      5.231861e-01            -inf                    5.222825e-01            2.328552e+00            -inf                    5.064727e-01            5.430275e-01            2.202903e+00            1.448495e+00            1.996552e+00            
5      9.226954e-05            -inf                    1.202859e-03            3.981588e+05            -inf                    2.051348e-04            1.248673e-03            7.436739e+07            1.858500e-04            1.978858e+03            
6      4.082087e-02            -inf                    2.097410e+01            1.594290e+02            -inf                    2.872266e+01            1.059919e+01            1.141110e+03            1.240698e+00            6.232499e+02            
7      8.003612e-02            -inf                    1.921747e-01            3.573131e+01            -inf                    1.938579e-01            2.068757e-01            8.014808e+01            2.554442e+00            1.875460e+01            
8      2.510861e-01            -inf                    9.631313e-01            1.121241e+01            -inf                    3.190720e-01            1.800661e+00            1.112656e+01            1.775192e+00            6.831711e+00            
9      7.031523e-02            -inf                    4.643282e-01            1.383704e+01            -inf                    3.277103e+00            5.188382e+00            7.646853e+01            7.215359e+00            3.948523e+01            
10     1.880487e+01            -inf                    6.966444e+01            6.549148e+01            -inf                    1.553683e+01            1.432131e+02            2.303515e+02            2.698494e+01            8.761618e+01            
11     1.598485e+01            -inf                    1.809556e+02            2.372755e+02            -inf                    1.344471e+01            4.082421e+02            7.482243e+02            1.164549e+01            3.732246e+02            
12     1.843518e+01            -inf                    8.193479e+01            8.031993e+01            -inf                    1.897768e+01            1.212461e+02            2.196216e+02            2.703429e+01            9.236604e+01            
13     2.988347e+00            -inf                    1.640389e+01            1.612508e+01            -inf                    3.705025e+00            2.352980e+01            3.159782e+01            3.371173e+00            2.058643e+01            
14     2.851132e+00            -inf                    3.190380e+00            6.411220e+00            -inf                    2.897085e+00            3.968498e+00            9.044603e+00            2.896932e+00            3.915702e+00            
15     2.814486e+00            -inf                    3.323138e+00            3.495974e+00            -inf                    3.076315e+00            3.856929e+00            4.051652e+00            2.804955e+00            3.678958e+00            
16     1.072903e+02            -inf                    3.765674e+02            4.496696e+02            -inf                    2.547032e+02            3.049139e+02            4.854054e+03            9.906170e+01            8.476774e+02            
17     4.180814e+01            -inf                    1.509921e+04            2.400578e+04            -inf                    3.085410e+01            4.037907e+04            1.666356e+05            1.149129e+03            4.164594e+04            
18     1.701105e+01            -inf                    3.194105e+01            3.967895e+01            -inf                    2.191903e+01            4.409027e+01            6.830788e+01            2.006857e+01            4.322713e+01            
19     1.652616e+01            -inf                    5.290778e+01            8.947964e+01            -inf                    2.459037e+01            1.122774e+02            2.322496e+02            2.633599e+01            7.351585e+01            
20     2.055197e+01            -inf                    2.972368e+01            2.904546e+01            -inf                    2.270477e+01            3.198047e+01            3.813457e+01            2.279031e+01            2.589069e+01            
21     4.509719e+00            -inf                    4.651486e+00            5.249551e+00            -inf                    4.531756e+00            5.068911e+00            5.640673e+00            4.599885e+00            4.852992e+00            
22     8.070405e+00            -inf                    1.240335e+01            1.285019e+01            -inf                    1.195471e+01            1.341725e+01            1.728753e+01            7.997768e+00            1.096977e+01            
23     1.977124e+01            -inf                    2.601758e+01            2.815638e+01            -inf                    1.745881e+01            4.163003e+01            5.765529e+01            2.120690e+01            4.809423e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_07_idea_0.py  (error=5.178898e-02)
Task  1: variant_05_idea_0.py  (error=3.549066e-02)
Task  2: original.py  (error=2.877613e-01)
Task  3: original.py  (error=1.347502e-01)
Task  4: variant_05_idea_0.py  (error=5.064727e-01)
Task  5: original.py  (error=9.226954e-05)
Task  6: original.py  (error=4.082087e-02)
Task  7: original.py  (error=8.003612e-02)
Task  8: original.py  (error=2.510861e-01)
Task  9: original.py  (error=7.031523e-02)
Task 10: variant_05_idea_0.py  (error=1.553683e+01)
Task 11: variant_09_idea_0.py  (error=1.164549e+01)
Task 12: original.py  (error=1.843518e+01)
Task 13: original.py  (error=2.988347e+00)
Task 14: original.py  (error=2.851132e+00)
Task 15: variant_09_idea_0.py  (error=2.804955e+00)
Task 16: variant_09_idea_0.py  (error=9.906170e+01)
Task 17: variant_05_idea_0.py  (error=3.085410e+01)
Task 18: original.py  (error=1.701105e+01)
Task 19: original.py  (error=1.652616e+01)
Task 20: original.py  (error=2.055197e+01)
Task 21: original.py  (error=4.509719e+00)
Task 22: variant_09_idea_0.py  (error=7.997768e+00)
Task 23: variant_05_idea_0.py  (error=1.745881e+01)

WIN COUNTS:
  original.py: 14 wins
  variant_05_idea_0.py: 5 wins
  variant_09_idea_0.py: 4 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_05_idea_0.py (5 wins) ---
```python
def _select_survivors_batch(self):
        """Select survivors via diversity-penalized fitness ranking."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        n_combined = len(combined_fit)
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
        for i in range(n_combined):
            dists = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
            dists[i] = np.inf
            if np.isfinite(dists).any():
                nn_dist = np.min(dists)
                pop_range = np.linalg.norm(self.ub - self.lb)
                diversity[i] = 1.0 + 5.0 * (nn_dist / (pop_range + 1e-10))

        diversity = np.clip(diversity, 1.0, 10.0)

        # Composite score: balance fitness (exponential rank) with diversity
        composite = exp_fitness * diversity

        # Stochastic selection weighted by composite score
        probs = composite / np.sum(composite)
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        probs = probs / np.sum(probs)

        selected_indices = np.random.choice(n_combined, size=self.NP, replace=False, p=probs)

        self.population = combined_pop[selected_indices]
        self.fitness = combined_fit[selected_indices]

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self):
        """Select survivors via fitness-distance crowding to preserve diversity."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        n_parents = self.NP
        n_trials = len(self.trials)

        # Compute pairwise Euclidean distances between trials and parents
        # Shape: (n_trials, n_parents)
        diff = self.trials[:, np.newaxis, :] - self.population[np.newaxis, :, :]  # Broadcasting
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # Euclidean distances

        # For each trial, find the nearest parent (crowding partner)
        nearest_parent = np.argmin(dists, axis=1)  # Shape: (n_trials,)
        nearest_dist = np.min(dists, axis=1)  # Shape: (n_trials,)

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
                    # else: trial already placed, skip

        self.population = new_pop
        self.fitness = new_fit
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
```

# --- From variant_09_idea_0.py (4 wins) ---
```python
def _select_survivors_batch(self):
        """Select survivors using fitness-sharing niching to preserve diversity."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        n_combined = len(combined_fit)
        sigma_share = 0.1 * (self.ub[0] - self.lb[0]) / self.NP

        niche_counts = np.zeros(n_combined)
        for i in range(n_combined):
            distances = np.linalg.norm(combined_pop - combined_pop[i], axis=1)
            niche_counts[i] = np.sum(distances < sigma_share) + 1e-10

        shared_fitness = combined_fit * niche_counts

        sorted_indices = np.argsort(shared_fitness)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
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
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
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
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        # More efficient than eigendecomposition for sampling
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

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
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
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_08()
        else:
            self._adapt_covariance_variant_09()
    
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
        """Condition-number triggered restart with adaptive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute condition number as diversity proxy
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

        # Adaptive learning rate based on condition number
        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
            cc_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.5
            cc_scale = 0.5
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        # Fitness variance for additional adaptation signal
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if rel_var < 1e-4:
            ccov_scale *= 0.5
            cc_scale *= 0.5

        # Trigger covariance restart on extreme condition number
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

        # Adaptive learning rates
        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path with adaptive rate
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_06(self):
        """Diversity-sensitive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
        
        if is_converging:
            scale = 10.0
        else:
            scale = 1.0 + 5.0 * min(rel_var, 0.1)
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Dual active evolution paths with exponential history weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        alpha = 0.05
        if not hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        self.pc_weighted = (1.0 - alpha) * self.pc_weighted + alpha * y_mean
        
        beta = 0.02
        if not hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            cross_contribution = np.sqrt(beta) * (y_mean - self.prev_y_mean)
            self.p_cross = (1.0 - beta) * self.p_cross + cross_contribution
        self.prev_y_mean = y_mean.copy()
        
        rank_one_primary = np.outer(self.pc, self.pc)
        rank_one_weighted = np.outer(self.pc_weighted, self.pc_weighted)
        rank_one_cross = np.outer(self.p_cross, self.p_cross)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_primary = self.ccov
        ccov_weighted = 0.3 * self.ccov
        ccov_cross = 0.1 * self.ccov
        
        self.C = ((1.0 - ccov_primary) * self.C + 
                  ccov_primary * rank_one_primary +
                  ccov_weighted * rank_one_weighted +
                  ccov_cross * rank_one_cross +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
        
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1e7:
            self.C *= 0.5
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking for escaping local optima."""
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt - self.f_opt
        self.prev_f_opt = self.f_opt

        fitness_gradient = max(abs(delta_f_opt), 1e-15)
        temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
        self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * self.exploration_temp
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

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