Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_operator_rewards` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.665524e-02            6.055232e-02            5.746104e-02            5.927445e-02            5.762745e-02            5.899845e-02            5.658902e-02            6.199895e-02            5.651761e-02            5.829245e-02            5.559027e-02            
1      4.402381e-02            5.337484e-02            4.612638e-02            4.469924e-02            4.494657e-02            4.273601e-02            4.889819e-02            4.740257e-02            4.565365e-02            4.990889e-02            4.323839e-02            
2      1.599018e-01            2.789983e+00            1.484026e-01            1.695608e-01            1.472359e-01            1.706168e-01            1.854344e-01            3.988876e+00            2.116076e-01            2.869251e+00            1.525748e-01            
3      1.330377e-01            2.491105e+00            1.297790e-01            1.385922e-01            1.334633e-01            1.380842e-01            1.734788e-01            2.238456e+00            1.387498e-01            1.948525e+00            4.671270e-01            
4      5.132094e-01            5.553317e-01            5.205970e-01            5.213147e-01            5.172480e-01            5.141570e-01            5.221241e-01            5.594020e-01            5.155339e-01            5.202890e-01            5.289107e-01            
5      8.994948e-05            9.525728e-05            8.967386e-05            8.575523e-05            1.075705e-04            8.859500e-05            9.301769e-05            1.113002e-04            1.013273e-04            1.096595e-04            8.330718e-05            
6      1.544828e-02            3.436862e+01            3.297120e-02            1.544679e-02            1.590122e-02            1.583771e-02            1.507164e-02            2.209995e+01            1.498916e-02            1.515052e+01            1.572349e-02            
7      7.777278e-02            2.656920e-01            7.504067e-02            7.964489e-02            7.723688e-02            7.836027e-02            8.136647e-02            9.777699e-01            7.731178e-02            7.471650e-02            7.766073e-02            
8      2.628168e-01            1.895644e+00            2.556501e-01            3.203117e-01            2.591857e-01            2.635268e-01            2.637513e-01            1.639532e+00            2.555880e-01            2.651285e-01            2.541925e-01            
9      1.293048e-01            7.584839e+00            2.105896e-01            6.961319e-02            1.395991e-01            6.832398e-02            7.095271e-02            7.103648e+00            1.174783e-01            2.942225e+00            6.967710e-02            
10     1.259035e+01            8.955059e+00            8.162930e+00            1.365727e+01            9.354643e+00            8.490591e+00            8.571336e+00            8.803694e+00            1.176038e+01            9.165906e+00            8.763994e+00            
11     1.776846e+01            8.566886e+00            8.108542e+00            1.404062e+01            9.037246e+00            7.714532e+00            1.168980e+01            1.432027e+01            1.722330e+01            1.333090e+01            1.457952e+01            
12     1.375762e+01            8.098369e+00            8.080322e+00            1.647377e+01            1.045521e+01            8.692513e+00            8.578792e+00            8.980625e+00            1.523195e+01            1.142522e+01            1.143540e+01            
13     2.749330e+00            1.955387e+00            1.955729e+00            2.744173e+00            2.185460e+00            1.851094e+00            2.442382e+00            2.404637e+00            2.592212e+00            2.742798e+00            2.186993e+00            
14     2.719233e+00            2.785712e+00            2.626254e+00            2.803118e+00            2.818642e+00            2.748222e+00            2.776400e+00            2.712880e+00            2.654008e+00            2.781202e+00            2.694954e+00            
15     2.666066e+00            2.380935e+00            2.643676e+00            2.456679e+00            2.626594e+00            2.584713e+00            2.508712e+00            2.199197e+00            2.745326e+00            2.727144e+00            2.414504e+00            
16     8.012821e+01            1.526997e+02            5.739839e+01            7.182745e+01            7.052587e+01            5.178944e+01            6.022206e+01            1.274100e+02            8.242489e+01            7.500529e+01            7.721633e+01            
17     3.286199e+01            5.017827e+01            3.284113e+01            4.395414e+01            3.153675e+01            3.570907e+01            3.026128e+01            4.694237e+01            3.295398e+01            8.626709e+01            3.474454e+01            
18     1.627988e+01            9.654368e+00            1.168989e+01            1.573655e+01            1.302548e+01            1.113497e+01            1.269101e+01            1.179778e+01            2.016562e+01            1.599028e+01            1.190403e+01            
19     1.556031e+01            1.359738e+01            9.269010e+00            1.240815e+01            1.338908e+01            1.299883e+01            1.646008e+01            1.207945e+01            1.298989e+01            1.742033e+01            1.182043e+01            
20     2.207555e+01            1.712695e+01            1.955043e+01            1.932041e+01            1.978709e+01            1.844703e+01            1.986290e+01            1.698451e+01            2.014811e+01            1.860556e+01            1.703880e+01            
21     4.372075e+00            4.477630e+00            4.483022e+00            4.533298e+00            4.386372e+00            4.339523e+00            4.344477e+00            4.374755e+00            4.445053e+00            4.419507e+00            4.379261e+00            
22     8.629916e+00            5.293243e+00            5.523163e+00            8.391963e+00            6.187924e+00            9.802064e+00            8.532690e+00            6.286850e+00            9.590900e+00            9.297770e+00            3.840704e+00            
23     2.340256e+01            2.217554e+01            1.624313e+01            1.621048e+01            1.513967e+01            2.078576e+01            2.168445e+01            2.123628e+01            2.244448e+01            2.229230e+01            1.561599e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_10_idea_0.py  (error=5.559027e-02)
Task  1: variant_05_idea_0.py  (error=4.273601e-02)
Task  2: variant_04_idea_0.py  (error=1.472359e-01)
Task  3: variant_02_idea_0.py  (error=1.297790e-01)
Task  4: original.py  (error=5.132094e-01)
Task  5: variant_10_idea_0.py  (error=8.330718e-05)
Task  6: variant_08_idea_0.py  (error=1.498916e-02)
Task  7: variant_09_idea_0.py  (error=7.471650e-02)
Task  8: variant_10_idea_0.py  (error=2.541925e-01)
Task  9: variant_05_idea_0.py  (error=6.832398e-02)
Task 10: variant_02_idea_0.py  (error=8.162930e+00)
Task 11: variant_05_idea_0.py  (error=7.714532e+00)
Task 12: variant_02_idea_0.py  (error=8.080322e+00)
Task 13: variant_05_idea_0.py  (error=1.851094e+00)
Task 14: variant_02_idea_0.py  (error=2.626254e+00)
Task 15: variant_07_idea_0.py  (error=2.199197e+00)
Task 16: variant_05_idea_0.py  (error=5.178944e+01)
Task 17: variant_06_idea_0.py  (error=3.026128e+01)
Task 18: variant_01_idea_0.py  (error=9.654368e+00)
Task 19: variant_02_idea_0.py  (error=9.269010e+00)
Task 20: variant_07_idea_0.py  (error=1.698451e+01)
Task 21: variant_05_idea_0.py  (error=4.339523e+00)
Task 22: variant_10_idea_0.py  (error=3.840704e+00)
Task 23: variant_04_idea_0.py  (error=1.513967e+01)

WIN COUNTS:
  variant_05_idea_0.py: 6 wins
  variant_02_idea_0.py: 5 wins
  variant_10_idea_0.py: 4 wins
  variant_04_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  original.py: 1 wins
  variant_08_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins
  variant_01_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (1 wins) ---
```python
def _update_operator_rewards(self):
        """Update operator rewards using immediate sigmoid reward with EMA tracking."""
        # Compute signed improvement (negative = worsening)
        improvement = self.f_opt_prev - self.f_opt

        # Scale-invariant relative improvement
        scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
        relative_improvement = improvement / scale

        # Bounded sigmoid reward in [-1, 1] - scale-invariant and numerically stable
        reward = 2.0 / (1.0 + np.exp(-5.0 * np.clip(relative_improvement, -3.0, 3.0))) - 1.0
        reward = float(np.clip(reward, -1.0, 1.0))

        op = self.current_operator

        # Initialize EMA state for each operator if needed
        if not hasattr(self, 'reward_ema'):
            self.reward_ema = np.zeros(self.num_operators)
            self.reward_count = np.zeros(self.num_operators)

        # Exponential moving average update (EMA alpha = 0.3)
        if self.reward_count[op] == 0:
            self.reward_ema[op] = reward
        else:
            self.ema_alpha = 0.3
            self.reward_ema[op] = (1.0 - self.ema_alpha) * self.reward_ema[op] + self.ema_alpha * reward

        self.reward_count[op] += 1

        # Standard Beta distribution update with bounded reward
        # Maps reward from [-1, 1] to [0, 1] for Beta parameters
        reward_mapped = (reward + 1.0) / 2.0
        reward_mapped = float(np.clip(reward_mapped, 1e-10, 1.0 - 1e-10))

        # Update Beta distribution parameters (Thompson Sampling)
        self.alpha[op] += reward_mapped
        self.beta[op] += (1.0 - reward_mapped)

        # Keep parameters bounded to prevent numerical issues
        self.alpha[op] = float(np.clip(self.alpha[op], 1e-10, 1e6))
        self.beta[op] = float(np.clip(self.beta[op], 1e-10, 1e6))
```

# --- From variant_02_idea_0.py (5 wins) ---
```python
def _update_operator_rewards(self):
        """Rank-based credit assignment with stagnation escape detection."""
        # Compute rank-based improvement: how much did f_opt improve relative to population spread?
        pop_range = max(np.max(self.fitness) - np.min(self.fitness), 1e-10)
        rank_improvement = max(0.0, self.f_opt_prev - self.f_opt) / pop_range

        # Compute diversity signal (normalized)
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Stagnation escape detection: reward operators when stagnation breaks
        is_escaping = (self.stagnation_counter == 0 and 
                       hasattr(self, 'prev_stagnation') and 
                       self.prev_stagnation >= self.max_stagnation // 2)
        escape_bonus = 5.0 if is_escaping else 0.0
        self.prev_stagnation = self.stagnation_counter

        # Compute rank-based reward: operator's contribution to population improvement
        op = self.current_operator

        # Track per-operator best fitness history for rank computation
        if not hasattr(self, 'op_best_history'):
            self.op_best_history = {i: [] for i in range(self.num_operators)}

        # Compute how operator's sampled trials rank in the trial population
        if len(self.trial_fitness) >= self.NP // 4:
            sorted_trial_fitness = np.sort(self.trial_fitness)
            best_trial_rank = np.searchsorted(sorted_trial_fitness, self.f_opt, side='left') / len(sorted_trial_fitness)
            rank_reward = float(1.0 - best_trial_rank)  # Higher = better (closer to best trial)
        else:
            rank_reward = 0.5

        # Composite reward: rank contribution + diversity boost + escape bonus
        reward = float(2.0 * rank_reward + 0.5 * diversity + escape_bonus)
        reward = float(np.clip(reward, -10.0, 20.0))  # Allow higher positive rewards for escape

        # Sliding window with longer memory for rank-based signals
        window_size = max(self.reward_window_size, 15)
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            # Use exponential moving average for more responsive tracking
            if hasattr(self, 'ema_reward'):
                self.ema_reward[op] = 0.7 * self.ema_reward[op] + 0.3 * reward
            else:
                self.ema_reward = {i: 0.0 for i in range(self.num_operators)}
                self.ema_reward[op] = reward

            # Adaptive Beta parameters based on EMA and window statistics
            ema = self.ema_reward[op]
            var_reward = float(np.var(self.operator_rewards[op]))

            # Reward scaling: amplify differences between operators
            scaled_mean = np.clip(ema * 2.0, 0.0, 1.0)
            scaled_var = np.clip(var_reward * 4.0, 0.01, 1.0)

            self.alpha[op] = float(max(1.0, scaled_mean * (scaled_mean * (n - 1) / (scaled_var + 1e-10) + 1)))
            self.beta[op] = float(max(1.0, (1 - scaled_mean) * ((n - 1) * (1 - scaled_mean) / (scaled_var + 1e-10) + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _update_operator_rewards(self):
        """Update operator rewards using trend-aware multi-signal credit assignment with landscape analysis."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        if not hasattr(self, 'ema_improvement'):
            self.ema_improvement = 0.0
            self.ema_reward = 0.0
            self.op_trend_count = {i: 0 for i in range(self.num_operators)}
            self.op_last_reward = {i: 0.0 for i in range(self.num_operators)}

        # Trend-based improvement signal: smooth improvement rate
        self.ema_improvement = 0.7 * self.ema_improvement + 0.3 * improvement
        trend_signal = np.log1p(max(self.ema_improvement * 1e10, 1e-15))

        # Fitness landscape analysis
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Rank entropy for ruggedness detection
        ranks = np.argsort(np.argsort(self.fitness))
        rank_probs = (ranks + 1) / (self.NP + 1)
        rank_probs = np.clip(rank_probs, 1e-10, 1.0)
        rank_entropy = -np.mean(rank_probs * np.log(rank_probs))
        rank_entropy /= np.log(self.NP + 1) + 1e-10

        # Consistency signal: reward operators that improve steadily
        op = self.current_operator
        if improvement > 1e-15:
            if self.op_last_reward[op] > 0.0 or self.op_trend_count[op] > 0:
                self.op_trend_count[op] += 1
            else:
                self.op_trend_count[op] = 1
        else:
            self.op_trend_count[op] = max(0, self.op_trend_count[op] - 1)
        self.op_last_reward[op] = improvement

        consistency = np.tanh(self.op_trend_count[op] / 5.0)

        # Multi-signal reward combination
        base_reward = trend_signal / 10.0
        diversity_reward = 0.15 * diversity
        ruggedness_penalty = 0.05 * (1.0 - rank_entropy)
        consistency_reward = 0.3 * consistency

        reward = base_reward + diversity_reward + ruggedness_penalty + consistency_reward
        reward = float(np.clip(reward, -10.0, 10.0))

        # Exponential moving average of reward for trend tracking
        self.ema_reward = 0.9 * self.ema_reward + 0.1 * reward

        self.operator_rewards[op].append(reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            ema_reward = float(self.ema_reward)
            combined = 0.6 * mean_reward + 0.4 * ema_reward

            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)

            self.alpha[op] = float(max(1.0, combined * (combined * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - combined) * ((n - 1) * (1 - combined) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
```

# --- From variant_05_idea_0.py (6 wins) ---
```python
def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Initialize cumulative tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay to accumulated reward (recent rewards weighted more)
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Log-scaled improvement reward (more sensitive to small improvements)
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Explicit stagnation detection
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Exploration bonus when stagnant and diversity is low
        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        # Update cumulative weighted reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize by decay sum to get comparable reward values
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

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
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _update_operator_rewards(self):
        """Update operator rewards using max-reward tracking with EMA smoothing."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        current_reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        current_reward = float(np.clip(current_reward, -10.0, 10.0))

        op = self.current_operator

        # Initialize max-reward EMA tracking per operator
        if not hasattr(self, 'op_max_reward_ema'):
            self.op_max_reward_ema = np.ones(self.num_operators) * 0.5
        if not hasattr(self, 'op_max_reward_count'):
            self.op_max_reward_count = np.zeros(self.num_operators)

        # Update max-reward EMA: tracks the best reward level this operator achieves
        prev_max_ema = self.op_max_reward_ema[op]
        if current_reward > prev_max_ema:
            # New breakthrough: fast update toward it
            self.op_max_reward_ema[op] = 0.7 * prev_max_ema + 0.3 * current_reward
        else:
            # No breakthrough: slow decay toward current reward
            self.op_max_reward_ema[op] = 0.98 * prev_max_ema + 0.02 * current_reward

        # Track how many times this operator set a new max
        if current_reward >= prev_max_ema and self.op_max_reward_count[op] > 0:
            self.op_max_reward_count[op] += 1
        elif self.op_max_reward_count[op] == 0 and current_reward > 0:
            self.op_max_reward_count[op] = 1

        # Combined reward: current + weighted max-reward component
        # This rewards operators that achieve HIGH peak performance
        max_component = 0.5 * max(self.op_max_reward_ema[op], 0.0)
        reward = 0.6 * current_reward + 0.4 * max_component
        reward = float(np.clip(reward, -10.0, 10.0))

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
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _update_operator_rewards(self):
        """Update operator rewards using per-operator rank-normalized credit assignment."""
        # Compute global metrics
        f_vals = np.concatenate([self.fitness, self.trial_fitness])
        f_prev = self.f_opt_prev

        # Per-operator success tracking
        if not hasattr(self, 'op_success_history'):
            self.op_success_history = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'op_improvement_history'):
            self.op_improvement_history = {i: [] for i in range(self.num_operators)}

        # Compute operator-specific improvement (based on selection counts)
        op = self.current_operator
        self.selection_counts[op] += 1

        # Individual improvement contribution
        if len(self.trial_fitness) >= self.NP:
            individual_improvement = max(0.0, self.f_opt_prev - np.min(self.trial_fitness))
        else:
            individual_improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Store in per-operator history
        self.op_improvement_history[op].append(individual_improvement)
        if len(self.op_improvement_history[op]) > self.reward_window_size:
            self.op_improvement_history[op].pop(0)

        # Per-operator success rate
        recent_improved = 0
        recent_total = min(self.NP, len(self.trial_fitness))
        if recent_total > 0:
            for i in range(recent_total):
                if i < len(self.trial_fitness) and i < len(self.fitness):
                    if self.trial_fitness[i] < self.fitness[i]:
                        recent_improved += 1
        op_success_rate = recent_improved / max(recent_total, 1)

        self.op_success_history[op].append(op_success_rate)
        if len(self.op_success_history[op]) > self.reward_window_size:
            self.op_success_history[op].pop(0)

        # Population diversity
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Fitness landscape roughness (variance of recent improvements)
        all_improvements = []
        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                imp = max(0.0, self.fitness[i] - self.trial_fitness[i])
                all_improvements.append(imp)

        roughness = 0.0
        if len(all_improvements) > 1:
            roughness = np.var(all_improvements) / (np.mean(all_improvements) ** 2 + 1e-10)
        roughness = np.clip(roughness, 0.0, 10.0)

        # Rank-normalized improvement: compare this operator's recent performance
        # to what it achieved in previous windows
        op_improvement_mean = np.mean(self.op_improvement_history[op]) if self.op_improvement_history[op] else 0.0
        op_success_mean = np.mean(self.op_success_history[op]) if self.op_success_history[op] else 0.0

        # Normalize by operator's own baseline (adaptive comparison)
        improvement_ratio = 1.0
        if op_improvement_mean > 1e-15:
            improvement_ratio = np.clip(individual_improvement / (op_improvement_mean + 1e-15), 0.0, 5.0)

        success_ratio = 1.0
        if op_success_mean > 1e-15:
            success_ratio = np.clip(op_success_rate / (op_success_mean + 1e-15), 0.0, 5.0)

        # Combined reward with multiple signals
        reward = 0.0

        # 1. Log-scaled improvement (core signal)
        reward += np.log1p(individual_improvement * 1e10) / 10.0

        # 2. Per-operator rank normalization bonus
        reward += 0.5 * (improvement_ratio - 1.0)
        reward += 0.3 * (success_ratio - 1.0)

        # 3. Diversity contribution (diversity = good, low diversity = penalize)
        reward += 0.2 * diversity

        # 4. Landscape roughness adaptation
        # High roughness = complex landscape = reward exploration
        reward += 0.1 * np.tanh(roughness - 1.0)

        reward = float(np.clip(reward, -10.0, 10.0))

        # Update sliding window
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        # Update Beta distribution parameters
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))

            # Adaptive learning rate based on variance
            cv_sq = var_reward / (mean_reward ** 2 + 1e-10)  # coefficient of variation squared
            lr_scale = 1.0 / (1.0 + np.sqrt(max(cv_sq, 0.01)))
            lr_scale = np.clip(lr_scale, 0.3, 3.0)

            denom = max(var_reward * n + 1e-10, 1e-10)
            alpha_update = mean_reward * (mean_reward * (n - 1) / denom + 1) * lr_scale
            beta_update = (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1) * lr_scale

            self.alpha[op] = float(max(1.0, alpha_update))
            self.beta[op] = float(max(1.0, beta_update))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = max(1.0, 1.0 + sum_reward)
            else:
                self.beta[op] = max(1.0, 1.0 - sum_reward)
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _update_operator_rewards(self):
            """Update operator rewards using inverse-error task-difficulty weighting.

            Amplifies rewards for improvements on tasks with large errors, so operators
            that escape severe local optima receive substantially higher credit.
            """
            improvement = max(0.0, self.f_opt_prev - self.f_opt)

            # Task difficulty weighting: amplify credit when stuck at high error
            # Scale factor grows with absolute fitness magnitude
            error_magnitude = max(abs(self.f_opt), 1.0)
            difficulty_weight = np.log1p(error_magnitude) + 1.0
            difficulty_weight = float(np.clip(difficulty_weight, 1.0, 100.0))

            # Diversity component
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

            # Base reward scaled by task difficulty
            base_reward = float(np.log1p(improvement * 1e10) / 10.0)
            reward = base_reward * difficulty_weight + 0.1 * diversity
            reward = float(np.clip(reward, -20.0, 20.0))

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
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _update_operator_rewards(self):
        """Update operator rewards using error-scaled exploration-weighted credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Error scale factor: larger for harder (higher error) tasks
        # This gives more credit to operators that make progress on hard problems
        error_scale = np.log1p(max(self.f_opt, 1e-15))
        error_scale = float(np.clip(error_scale / 10.0, 0.1, 5.0))

        # Population diversity and exploration metrics
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Exploration bonus: operators that generate solutions far from current mean
        # are valuable for escaping local optima on hard tasks
        mean_dist = np.mean(np.linalg.norm(self.population - self.mean, axis=1))
        expected_dist = (self.ub[0] - self.lb[0]) / np.sqrt(12.0)
        exploration_ratio = np.clip(mean_dist / (expected_dist + 1e-10), 0.0, 2.0)

        # Best-so-far progress rate (per generation over recent window)
        if not hasattr(self, 'progress_history'):
            self.progress_history = []
        self.progress_history.append(self.f_opt)
        if len(self.progress_history) > 20:
            self.progress_history.pop(0)

        # Calculate progress rate: negative = improvement
        progress_rate = 0.0
        if len(self.progress_history) >= 5:
            recent = np.array(self.progress_history[-5:])
            older = np.array(self.progress_history[:len(recent)])
            if len(older) >= len(recent):
                progress_rate = float(np.mean(recent - older[:len(recent)]))

        # Long-horizon exponential credit: weight recent rewards more but keep history
        if not hasattr(self, 'operator_ema'):
            self.operator_ema = {i: 0.0 for i in range(self.num_operators)}

        # Base reward from improvement, scaled by error magnitude
        base_reward = float(np.log1p(improvement * 1e10 + 1e-15) / 10.0)

        # Exploration bonus: significant reward for maintaining diversity
        # This encourages operators to keep exploring on hard tasks
        exploration_bonus = 0.5 * diversity * exploration_ratio

        # Progress rate reward: reward operators contributing to downward trend
        progress_reward = -0.3 * np.tanh(progress_rate)

        # Combine: error_scale amplifies all rewards for harder tasks
        reward = error_scale * (base_reward + exploration_bonus + progress_reward)
        reward += 0.1 * diversity  # Small direct diversity bonus
        reward = float(np.clip(reward, -10.0, 10.0))

        op = self.current_operator

        # Exponential moving average for long-horizon credit
        # This gives credit to operators even after the fact if they set up future success
        decay = 0.85
        self.operator_ema[op] = decay * self.operator_ema[op] + (1.0 - decay) * reward

        # Store in sliding window for Beta distribution updates
        self.operator_rewards[op].append(self.operator_ema[op])

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))

            # Ensure numerical stability
            denom = max(var_reward * n + 1e-10, 1e-10)

            # Beta distribution parameters with exploration incentive
            # Higher mean + higher variance = more exploratory = better for hard tasks
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1) + 0.5 * diversity * n))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
```

# --- From variant_10_idea_0.py (4 wins) ---
```python
def _update_operator_rewards(self):
            """Update operator rewards using multi-signal trajectory tracking."""
            improvement = max(0.0, self.f_opt_prev - self.f_opt)

            # Signal 1: Immediate improvement (log-scaled, clamped)
            reward_immediate = float(np.log1p(improvement * 1e10) / 10.0)
            reward_immediate = np.clip(reward_immediate, -10.0, 10.0)

            # Signal 2: Exponential moving average of improvement rate
            if not hasattr(self, 'reward_ema'):
                self.reward_ema = 0.0
            self.reward_ema = 0.7 * self.reward_ema + 0.3 * reward_immediate

            # Signal 3: Cumulative improvement over sliding window
            if not hasattr(self, 'reward_cumsum'):
                self.reward_cumsum = 0.0
            self.reward_cumsum += improvement
            if not hasattr(self, 'cumsum_buffer'):
                self.cumsum_buffer = 0.0
            self.cumsum_buffer += 1
            if self.cumsum_buffer > 20:
                self.reward_cumsum *= 0.95
                self.cumsum_buffer *= 0.95

            # Signal 4: Convergence status bonus/penalty
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

            # Stagnation detection
            if not hasattr(self, 'stagnation_for_reward'):
                self.stagnation_for_reward = 0
            if improvement > 1e-15:
                self.stagnation_for_reward = 0
            else:
                self.stagnation_for_reward += 1

            # Multi-signal combination: immediate (30%) + EMA (40%) + diversity (15%) + cumsum (15%)
            reward = (0.30 * reward_immediate + 
                      0.40 * self.reward_ema + 
                      0.15 * diversity + 
                      0.15 * np.clip(np.log1p(self.reward_cumsum * 1e5) / 10.0, -5.0, 5.0))

            # Stagnation penalty: reduce reward when stuck
            if self.stagnation_for_reward > 10:
                stagnation_factor = max(0.0, 1.0 - (self.stagnation_for_reward - 10) / 50.0)
                reward *= stagnation_factor

            # Exploitation bonus when converging well
            if self.stagnation_for_reward == 0 and self.reward_ema > 1.0:
                reward += 0.5

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
        """Archive-guided covariance perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Initialize archive for tracking distinct best solutions
        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        # Add current best to archive if distinct enough
        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for arch_pos in self.archive_positions:
            dist = np.linalg.norm(self.x_opt - arch_pos)
            if dist < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.archive_positions) < 10:
            self.archive_positions.append(self.x_opt.copy())
            self.archive_fitness.append(self.f_opt)
            self.archive_generations.append(self.generation)
        elif is_distinct and len(self.archive_positions) >= 10:
            # Replace worst entry
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        # Compute diversity and condition metrics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        # Compute archive-based escape direction
        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            # Direction from mean toward archive centroid, weighted by diversity
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                # Also consider directions to individual archive members
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        # Perturb along diverse directions
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            # Add global exploration along principal axes
            escape_perturb += 0.1 * np.eye(self.dim)

        # Adaptive learning rates with exploration boost when trapped
        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with archive-based perturbation
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Inject escape perturbation when trapped
        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        # Full restart on extreme ill-conditioning
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
    
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
        """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Evolution path update (unchanged)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with positive/negative decomposition
        rank_mu_pos = np.zeros((self.dim, self.dim))
        rank_mu_neg = np.zeros((self.dim, self.dim))

        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            weight = self.weights[i]
            outer = np.outer(diff, diff)

            if weight > 0:
                rank_mu_pos += weight * outer
            else:
                rank_mu_neg += abs(weight) * outer

        # Normalize by sum of positive and negative weights separately
        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        # Combine positive and negative updates (active CMA-ES core)
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
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