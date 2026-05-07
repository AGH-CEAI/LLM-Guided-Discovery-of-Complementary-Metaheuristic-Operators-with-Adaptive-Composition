Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_operator_rewards` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.957150e-02            5.732685e-02            5.987112e-02            6.364076e-02            5.797416e-02            5.861481e-02            5.896978e-02            6.235864e-02            7.027216e-02            6.069334e-02            5.706800e-02            
1      4.521667e-02            4.682527e-02            4.907657e-02            5.244478e-02            4.371125e-02            4.420284e-02            4.515112e-02            5.003911e-02            8.889055e-02            5.228660e-02            4.595898e-02            
2      1.563511e-01            3.286231e+00            4.015518e+00            3.080394e+00            1.508036e-01            1.846612e-01            1.859452e-01            1.488248e-01            8.278320e-01            1.292064e+00            1.505261e-01            
3      1.308743e-01            1.823059e+00            2.537856e+00            2.554195e+00            1.642763e-01            1.330639e-01            5.230169e-01            1.311981e-01            2.896256e+00            1.885419e+00            1.397300e-01            
4      5.290559e-01            5.167318e-01            5.339720e-01            6.163511e-01            5.113010e-01            5.129309e-01            5.166033e-01            5.080802e-01            6.559312e-01            5.413893e-01            5.152491e-01            
5      8.374466e-05            1.093610e-04            8.225278e-05            1.110900e-04            1.038192e-04            8.800092e-05            1.174399e-04            8.780993e-05            1.279432e-04            9.674271e-05            1.059537e-04            
6      1.468631e-02            4.526459e-01            2.740552e+01            2.909704e+01            1.492615e-02            1.492257e-02            2.657981e-02            1.584171e-02            1.375349e+02            1.540483e+01            1.568217e-02            
7      8.010645e-02            1.097442e-01            2.570306e+00            2.080658e+00            8.207013e-02            8.313185e-02            7.816825e-02            8.157060e-02            4.414304e-01            1.234492e-01            7.798654e-02            
8      2.498045e-01            3.297001e-01            1.986868e+00            1.775470e+00            2.693935e-01            2.521520e-01            2.873048e-01            2.597658e-01            7.581014e-01            1.901730e+00            2.596239e-01            
9      6.947953e-02            1.454908e-01            7.915875e+00            7.820826e+00            6.794622e-02            9.320406e-02            1.370635e-01            7.038459e-02            7.930906e+00            7.418278e+00            2.799460e-01            
10     1.074929e+01            6.783819e+00            7.346800e+00            8.986923e+00            8.526014e+00            9.595392e+00            7.992696e+00            7.312882e+00            9.138276e+00            7.463492e+00            8.203423e+00            
11     5.671458e+00            7.799662e+00            7.675671e+00            9.141950e+00            8.323834e+00            1.014889e+01            1.019534e+01            1.100011e+01            5.133810e+00            1.021597e+01            8.323218e+00            
12     6.976409e+00            6.951824e+00            7.937099e+00            1.021233e+01            8.185424e+00            9.022802e+00            1.082270e+01            8.614528e+00            7.906259e+00            1.167401e+01            7.453896e+00            
13     1.788624e+00            1.982079e+00            1.906227e+00            1.830828e+00            2.301881e+00            1.757159e+00            2.030597e+00            1.934920e+00            1.868320e+00            1.899388e+00            2.541388e+00            
14     2.702223e+00            2.721199e+00            2.800911e+00            2.817845e+00            2.653601e+00            2.651954e+00            2.672778e+00            2.636557e+00            2.609878e+00            2.652696e+00            2.792118e+00            
15     2.531735e+00            2.550994e+00            2.053814e+00            1.937442e+00            2.418852e+00            2.503044e+00            2.593793e+00            2.592220e+00            2.663019e+00            2.371629e+00            2.425405e+00            
16     7.043052e+01            1.184580e+02            1.434221e+02            1.417065e+02            5.989797e+01            5.918168e+01            1.128026e+02            5.941856e+01            2.030693e+02            8.930634e+01            8.608727e+01            
17     4.016573e+01            3.081971e+01            6.519654e+01            3.221203e+01            3.041124e+01            7.193677e+01            4.226572e+01            5.946502e+01            4.685380e+01            3.069143e+01            2.852541e+01            
18     1.037510e+01            1.144035e+01            7.996750e+00            1.107075e+01            1.152669e+01            1.554146e+01            1.695497e+01            1.538194e+01            1.709324e+01            9.862057e+00            1.220905e+01            
19     1.306437e+01            1.312798e+01            1.645299e+01            1.201569e+01            1.260969e+01            1.548252e+01            1.603379e+01            1.480049e+01            9.934939e+00            1.276364e+01            1.434815e+01            
20     1.830514e+01            1.991013e+01            1.687108e+01            1.763832e+01            1.818560e+01            1.868340e+01            2.086741e+01            1.828549e+01            2.428654e+01            1.892977e+01            1.892251e+01            
21     4.374806e+00            4.470445e+00            4.298341e+00            4.445695e+00            4.322218e+00            4.329452e+00            4.315351e+00            4.353576e+00            4.578597e+00            4.334484e+00            4.436595e+00            
22     4.685324e+00            5.210573e+00            4.486609e+00            6.859021e+00            4.794897e+00            9.097292e+00            9.003471e+00            8.675637e+00            1.078874e+01            7.898004e+00            5.645003e+00            
23     1.979292e+01            2.140333e+01            1.969655e+01            2.127321e+01            1.684415e+01            2.071040e+01            1.806200e+01            8.873557e+00            2.639010e+01            1.964918e+01            2.306755e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_10_idea_0.py  (error=5.706800e-02)
Task  1: variant_04_idea_0.py  (error=4.371125e-02)
Task  2: variant_07_idea_0.py  (error=1.488248e-01)
Task  3: original.py  (error=1.308743e-01)
Task  4: variant_07_idea_0.py  (error=5.080802e-01)
Task  5: variant_02_idea_0.py  (error=8.225278e-05)
Task  6: original.py  (error=1.468631e-02)
Task  7: variant_10_idea_0.py  (error=7.798654e-02)
Task  8: original.py  (error=2.498045e-01)
Task  9: variant_04_idea_0.py  (error=6.794622e-02)
Task 10: variant_01_idea_0.py  (error=6.783819e+00)
Task 11: variant_08_idea_0.py  (error=5.133810e+00)
Task 12: variant_01_idea_0.py  (error=6.951824e+00)
Task 13: variant_05_idea_0.py  (error=1.757159e+00)
Task 14: variant_08_idea_0.py  (error=2.609878e+00)
Task 15: variant_03_idea_0.py  (error=1.937442e+00)
Task 16: variant_05_idea_0.py  (error=5.918168e+01)
Task 17: variant_10_idea_0.py  (error=2.852541e+01)
Task 18: variant_02_idea_0.py  (error=7.996750e+00)
Task 19: variant_08_idea_0.py  (error=9.934939e+00)
Task 20: variant_02_idea_0.py  (error=1.687108e+01)
Task 21: variant_02_idea_0.py  (error=4.298341e+00)
Task 22: variant_02_idea_0.py  (error=4.486609e+00)
Task 23: variant_07_idea_0.py  (error=8.873557e+00)

WIN COUNTS:
  variant_02_idea_0.py: 5 wins
  variant_10_idea_0.py: 3 wins
  variant_07_idea_0.py: 3 wins
  original.py: 3 wins
  variant_08_idea_0.py: 3 wins
  variant_04_idea_0.py: 2 wins
  variant_01_idea_0.py: 2 wins
  variant_05_idea_0.py: 2 wins
  variant_03_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _update_operator_rewards(self):
        """UCB-inspired success rate tracking with immediate credit assignment."""
        # Immediate improvement signal
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        success = 1.0 if improvement > 0 else 0.0

        # Initialize tracking if needed
        if not hasattr(self, 'operator_successes'):
            self.operator_successes = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_trials'):
            self.operator_trials = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_ema_reward'):
            self.operator_ema_reward = np.zeros(self.num_operators)

        op = self.current_operator
        self.operator_trials[op] += 1.0

        # Update EMA reward for current operator
        self.operator_ema_reward[op] = 0.9 * self.operator_ema_reward[op] + 0.1 * success

        # Credit assignment: immediate success signal
        if success > 0:
            self.operator_successes[op] += success

        # Compute UCB-style exploration bonus for all operators
        ucb_scores = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            trials_i = max(self.operator_trials[i], 1.0)
            success_rate = self.operator_successes[i] / trials_i
            # UCB bonus: higher for less-explored operators
            exploration_bonus = np.sqrt(2.0 * np.log(max(self.generation + 1, 1)) / trials_i)
            ucb_scores[i] = success_rate + exploration_bonus

        # Scale exploration bonus by current exploration temperature
        exploration_temp = getattr(self, 'exploration_temp', 1.0)
        ucb_bonus_scaled = ucb_scores[op] - (self.operator_successes[op] / max(self.operator_trials[op], 1.0))
        ucb_bonus_scaled *= exploration_temp

        # Compute reward for current operator
        reward = float(np.clip(self.operator_ema_reward[op] + ucb_bonus_scaled, 0.0, 1.0))

        # Update operator rewards sliding window
        self.operator_rewards.setdefault(op, [])
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        # Direct Beta distribution update from empirical success rate
        total_trials = max(self.operator_trials[op], 1.0)
        empirical_success_rate = self.operator_successes[op] / total_trials

        # Combine EMA reward with current success for robust estimate
        combined_rate = 0.7 * empirical_success_rate + 0.3 * self.operator_ema_reward[op]
        combined_rate = float(np.clip(combined_rate, 0.0, 1.0))

        # Update Beta parameters with Laplace smoothing
        alpha_prior = 1.0
        beta_prior = 1.0
        self.alpha[op] = alpha_prior + self.operator_successes[op]
        self.beta[op] = beta_prior + (self.operator_trials[op] - self.operator_successes[op])

        # Apply reward-based adjustment to Beta parameters
        reward_adjustment = reward * 5.0
        self.alpha[op] = max(1.0, self.alpha[op] * (1.0 + reward_adjustment))
        self.beta[op] = max(1.0, self.beta[op] * (1.0 + (1.0 - reward) * 2.0))

        # Ensure alpha + beta stays bounded for numerical stability
        total_ab = self.alpha[op] + self.beta[op]
        if total_ab > 1000.0:
            scale = 1000.0 / total_ab
            self.alpha[op] *= scale
            self.beta[op] *= scale
```

# --- From variant_02_idea_0.py (5 wins) ---
```python
def _update_operator_rewards(self):
            """Rank-based multi-dimensional credit assignment for stable operator selection."""
            # Initialize tracking structures
            if not hasattr(self, 'operator_rank_scores'):
                self.operator_rank_scores = {i: [] for i in range(self.num_operators)}
            if not hasattr(self, 'operator_spread_scores'):
                self.operator_spread_scores = {i: [] for i in range(self.num_operators)}
            if not hasattr(self, 'operator_improvement_scores'):
                self.operator_improvement_scores = {i: [] for i in range(self.num_operators)}

            # === DIMENSION 1: Fitness Improvement ===
            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            if improvement > 0:
                # Log-scaled with temperature scaling based on problem difficulty
                temp = max(abs(self.f_opt), 1.0)
                imp_reward = float(np.log1p(improvement / (temp + 1e-10)) / np.log(11.0))
            else:
                imp_reward = 0.0

            # === DIMENSION 2: Population Spread (diversity contribution) ===
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            spread_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
            spread_reward = float(spread_ratio)

            # === DIMENSION 3: Best-Fitness Rank in Population ===
            # Compare current best to population distribution
            fit_mean = np.mean(self.fitness)
            fit_std = max(np.std(self.fitness), 1e-10)
            if fit_std > 1e-10:
                best_z = (self.f_opt - fit_mean) / fit_std
                rank_reward = float(np.clip(0.5 + best_z / 4.0, 0.0, 1.0))
            else:
                rank_reward = 0.5

            # === Stagnation Detection ===
            is_stagnant = self.stagnation_counter > self.max_stagnation // 2
            is_very_stagnant = self.stagnation_counter > self.max_stagnation

            # === Combine Rewards with Adaptive Weights ===
            if is_very_stagnant:
                # When very stuck, prioritize diversity and rank over raw improvement
                w_improve, w_spread, w_rank = 0.2, 0.5, 0.3
            elif is_stagnant:
                # When stagnating, balance all dimensions
                w_improve, w_spread, w_rank = 0.4, 0.3, 0.3
            else:
                # Normal operation: emphasize improvement
                w_improve, w_spread, w_rank = 0.6, 0.2, 0.2

            combined_reward = (w_improve * imp_reward + 
                              w_spread * spread_reward + 
                              w_rank * rank_reward)
            combined_reward = float(np.clip(combined_reward, 0.0, 1.0))

            # === Exploration Bonus ===
            if is_stagnant and spread_ratio < 0.1:
                combined_reward *= 0.5  # Penalize operators that don't help escape

            # === Update Sliding Windows ===
            op = self.current_operator
            self.operator_improvement_scores[op].append(imp_reward)
            self.operator_spread_scores[op].append(spread_reward)
            self.operator_rank_scores[op].append(rank_reward)

            window = self.reward_window_size
            for scores in [self.operator_improvement_scores, 
                           self.operator_spread_scores, 
                           self.operator_rank_scores]:
                if len(scores[op]) > window:
                    scores[op].pop(0)

            # === Compute Aggregated Reward for Beta Update ===
            n = len(self.operator_improvement_scores[op])
            if n >= 3:
                # Use smoothed reward from window
                mean_improve = np.mean(self.operator_improvement_scores[op])
                mean_spread = np.mean(self.operator_spread_scores[op])
                mean_rank = np.mean(self.operator_rank_scores[op])

                # Weighted combination matching the adaptive weights
                smoothed_reward = (w_improve * mean_improve + 
                                 w_spread * mean_spread + 
                                 w_rank * mean_rank)
            elif n >= 1:
                smoothed_reward = combined_reward
            else:
                smoothed_reward = 0.5

            smoothed_reward = float(np.clip(smoothed_reward, 0.0, 1.0))

            # === Update Beta Distribution Parameters ===
            # More stable update: use running mean directly
            prior_strength = 2.0
            if n >= 1:
                # Bayesian-style update: blend prior with observed
                obs_weight = min(n / (n + 2.0), 0.8)
                effective_reward = (1 - obs_weight) * 0.5 + obs_weight * smoothed_reward
            else:
                effective_reward = smoothed_reward

            effective_reward = float(np.clip(effective_reward, 0.01, 0.99))

            # Update alpha/beta with controlled learning rate
            lr = 0.3
            self.alpha[op] = float(max(1.0, (1 - lr) * self.alpha[op] + lr * (1.0 + effective_reward * prior_strength)))
            self.beta[op] = float(max(1.0, (1 - lr) * self.beta[op] + lr * (1.0 + (1 - effective_reward) * prior_strength)))

            # Ensure reasonable bounds on Beta parameters
            self.alpha[op] = float(np.clip(self.alpha[op], 1.0, 50.0))
            self.beta[op] = float(np.clip(self.beta[op], 1.0, 50.0))
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _update_operator_rewards(self):
        """Personal-best centric credit assignment with exploration awareness."""
        op = self.current_operator

        # Initialize per-operator personal best tracking
        if not hasattr(self, 'operator_best_fitness'):
            self.operator_best_fitness = np.full(self.num_operators, float('inf'))
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Get current best fitness in population
        current_fitness = float(np.min(self.fitness))

        # Track if this operator found a new personal best
        is_new_best = False
        if current_fitness < self.operator_best_fitness[op]:
            improvement = self.operator_best_fitness[op] - current_fitness
            self.operator_best_fitness[op] = current_fitness
            is_new_best = True
        else:
            improvement = 0.0

        # Relative improvement from operator's own best (0 to 1 scale)
        best_op = self.operator_best_fitness[op]
        worst_op = float(np.max(self.fitness))
        fitness_range = max(worst_op - best_op, 1e-10)
        relative_improvement = improvement / fitness_range
        improvement_reward = float(np.clip(relative_improvement * 10.0, 0.0, 5.0))

        # Diversity-based exploration reward
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        diversity_reward = float(2.0 * (1.0 - diversity))

        # Combine rewards: exploration when stagnant, improvement otherwise
        reward = 0.6 * improvement_reward + 0.4 * diversity_reward

        # Strong bonus for new personal best (key for escaping bad local optima)
        if is_new_best:
            new_best_bonus = 5.0 * float(np.log1p(max(improvement, 1e-10) / max(abs(best_op), 1e-10)))
            reward += new_best_bonus

        # Apply exponential decay and update cumulative reward
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay
        self.operator_cumulative_reward[op] += reward
        self.operator_decay_sum[op] += 1.0

        # Normalize for comparability
        norm = max(self.operator_decay_sum[op], 1.0)
        normalized_reward = float(np.clip(self.operator_cumulative_reward[op] / norm, -5.0, 10.0))

        # Update sliding window
        self.operator_rewards[op].append(normalized_reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        # Update Beta distribution for Thompson Sampling
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

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _update_operator_rewards(self):
        """Rank-based credit assignment with generational memory and difficulty scaling."""
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)
        if not hasattr(self, 'gen_rewards_history'):
            self.gen_rewards_history = []

        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        fit_scale = max(abs(self.f_opt), 1.0)
        difficulty = max(1.0, np.log10(fit_scale + 1.0))

        self.gen_rewards_history.append((improvement, self.current_operator))
        if len(self.gen_rewards_history) > self.reward_window_size:
            self.gen_rewards_history.pop(0)

        if len(self.gen_rewards_history) >= 3:
            all_improvements = [g[0] for g in self.gen_rewards_history]
            sorted_imps = sorted(all_improvements, reverse=True)
            rank = sorted_imps.index(improvement) if improvement > 0 else len(sorted_imps) - 1
            max_rank = max(len(sorted_imps) - 1, 1)
            rank_pct = 1.0 - (rank / max_rank)
            reward = float(np.log1p(rank_pct * difficulty * 10.0))
        elif improvement > 0:
            reward = float(np.log1p(improvement * 1e8) / 20.0)
        else:
            reward = 0.0

        decay = 0.9
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

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

# --- From variant_05_idea_0.py (2 wins) ---
```python
def _update_operator_rewards(self):
        """Reward operators for diversity contribution and exploration, not just improvement."""
        # Initialize tracking state
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)
        if not hasattr(self, 'prev_population'):
            self.prev_population = self.population.copy()
        if not hasattr(self, 'prev_mean'):
            self.prev_mean = self.mean.copy()

        # Compute diversity metrics
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        current_diversity = pop_variance / (expected_var + 1e-10)

        prev_variance = np.mean(np.var(self.prev_population, axis=0))
        prev_diversity = prev_variance / (expected_var + 1e-10)

        # Diversity change reward (positive if operator increased spread)
        diversity_delta = current_diversity - prev_diversity
        diversity_reward = np.log1p(max(diversity_delta, 1e-15)) - np.log1p(max(-diversity_delta, 1e-15))
        diversity_reward = float(np.clip(diversity_reward, -3.0, 3.0))

        # Exploration reward: measure of population movement away from current mean
        pop_distances = np.linalg.norm(self.population - self.mean, axis=1)
        mean_movement = np.linalg.norm(self.mean - self.prev_mean)
        exploration_score = float(np.mean(pop_distances) / (mean_movement + 1e-10))
        exploration_reward = float(np.clip(np.log1p(exploration_score) / 3.0, -2.0, 2.0))

        # Improvement reward (keep for tiebreaking when improvement happens)
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improvement_reward = 0.0
        if improvement > 0:
            improvement_reward = float(np.clip(np.log1p(improvement * 1e10) / 15.0, 0.0, 5.0))

        # Stagnation detection and bonus
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        stagnation_bonus = 1.5 if is_stagnant else 1.0

        # Combine rewards: diversity and exploration weighted higher when stagnant
        if is_stagnant:
            total_reward = (0.15 * improvement_reward + 
                           0.45 * diversity_reward + 
                           0.40 * exploration_reward) * stagnation_bonus
        else:
            total_reward = (0.50 * improvement_reward + 
                           0.25 * diversity_reward + 
                           0.25 * exploration_reward)

        total_reward = float(np.clip(total_reward, -5.0, 5.0))

        # Apply exponential decay to accumulated reward
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Update cumulative reward for selected operator
        self.operator_cumulative_reward[self.current_operator] += total_reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize reward
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        # Store in sliding window
        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        # Update Beta distribution parameters for Thompson Sampling
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

        # Update tracking state
        self.prev_population = self.population.copy()
        self.prev_mean = self.mean.copy()
```

# --- From variant_07_idea_0.py (3 wins) ---
```python
def _update_operator_rewards(self):
            """Rank-based multi-component credit assignment for deceptive multi-modal optimization."""
            # === Initialize tracking state ===
            if not hasattr(self, 'operator_cumulative_reward'):
                self.operator_cumulative_reward = np.zeros(self.num_operators)
            if not hasattr(self, 'operator_selection_quality'):
                self.operator_selection_quality = np.zeros(self.num_operators)
            if not hasattr(self, 'operator_diversity_contribution'):
                self.operator_diversity_contribution = np.zeros(self.num_operators)

            # === Component 1: Rank-based selection quality ===
            # Rank each trial candidate by fitness (lower fitness = better rank = higher reward)
            if len(self.trial_fitness) >= 2:
                ranks = np.argsort(np.argsort(self.trial_fitness))  # 0 = best, N-1 = worst
                n_trials = len(self.trial_fitness)
                rank_fraction = ranks / max(n_trials - 1, 1)  # Normalize to [0, 1]
                avg_rank_fraction = float(np.mean(rank_fraction))
                # Invert: lower rank_fraction is better -> higher reward
                selection_quality = 1.0 - avg_rank_fraction
            else:
                selection_quality = 0.5

            # === Component 2: Diversity maintenance contribution ===
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

            # Individual diversity contribution via pairwise distances
            if len(self.population) >= 2:
                pop_center = np.mean(self.population, axis=0)
                dists = np.linalg.norm(self.population - pop_center, axis=1)
                diversity_contribution = float(np.mean(dists) / (expected_var ** 0.5 + 1e-10))
                diversity_contribution = np.clip(diversity_contribution, 0.0, 1.0)
            else:
                diversity_contribution = 0.5

            # === Component 3: Regret-based (comparing to expected random performance) ===
            # Compute expected fitness from random sampling as baseline
            if not hasattr(self, 'random_baseline_fitness'):
                self.random_baseline_fitness = float(np.mean(self.func(np.random.uniform(
                    self.lb, self.ub, (min(1000, 10 * self.dim), self.dim))))
                )
            if not hasattr(self, 'regret_ema'):
                self.regret_ema = max(0.0, self.random_baseline_fitness - self.f_opt)

            # Regret: gap between current best and what random sampling would achieve
            regret = max(0.0, self.random_baseline_fitness - self.f_opt)
            self.regret_ema = 0.9 * self.regret_ema + 0.1 * regret
            # Higher regret = worse = operators not doing well
            regret_score = 1.0 / (1.0 + np.log1p(max(self.regret_ema, 1e-10)))

            # === Component 4: Stagnation-aware exploration bonus ===
            is_stagnant = self.stagnation_counter > self.max_stagnation // 3
            stagnation_bonus = 2.0 if is_stagnant else 1.0

            # === Combine components with adaptive weights ===
            # Weight by stagnation: when stagnant, emphasize diversity
            if is_stagnant:
                reward = (0.2 * selection_quality + 0.6 * diversity_contribution + 0.2 * regret_score)
            else:
                reward = (0.5 * selection_quality + 0.3 * diversity_contribution + 0.2 * regret_score)

            reward *= stagnation_bonus
            reward = float(np.clip(reward, 0.0, 5.0))

            # === Update per-operator tracking ===
            op = self.current_operator
            self.operator_cumulative_reward[op] += reward
            self.operator_selection_quality[op] = 0.9 * self.operator_selection_quality[op] + 0.1 * selection_quality
            self.operator_diversity_contribution[op] = 0.9 * self.operator_diversity_contribution[op] + 0.1 * diversity_contribution

            # === Compute normalized reward for Beta distribution update ===
            total_reward = float(np.sum(self.operator_cumulative_reward))
            if total_reward > 1e-10:
                normalized_reward = self.operator_cumulative_reward[op] / total_reward
            else:
                normalized_reward = 1.0 / self.num_operators

            # Also factor in selection quality and diversity contribution
            composite_score = 0.5 * normalized_reward + 0.3 * self.operator_selection_quality[op] + 0.2 * self.operator_diversity_contribution[op]
            composite_score = float(np.clip(composite_score, 0.0, 1.0))

            # === Update sliding window ===
            self.operator_rewards[op].append(composite_score)
            if len(self.operator_rewards[op]) > self.reward_window_size:
                self.operator_rewards[op].pop(0)

            # === Update Beta distribution parameters for Thompson Sampling ===
            n = len(self.operator_rewards[op])
            if n >= 3:
                mean_reward = float(np.mean(self.operator_rewards[op]))
                var_reward = float(np.var(self.operator_rewards[op]))
                # Bayesian update with uncertainty
                denom = max(var_reward * n + 1e-10, 1e-10)
                self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
                self.beta[op] = float(max(1.0, (1.0 - mean_reward) * ((n - 1) * (1.0 - mean_reward) / denom + 1)))
            elif n >= 1:
                sum_reward = float(np.sum(self.operator_rewards[op]))
                self.alpha[op] = 1.0 + sum_reward
                self.beta[op] = 1.0 + (n - sum_reward)
```

# --- From variant_08_idea_0.py (3 wins) ---
```python
def _update_operator_rewards(self):
        """Comparative rank-based multi-objective credit assignment with difficulty-aware scaling."""
        # Initialize tracking structures
        if not hasattr(self, 'operator_rank_history'):
            self.operator_rank_history = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'operator_streak'):
            self.operator_streak = np.zeros(self.num_operators)
        if not hasattr(self, 'best_known_errors'):
            self.best_known_errors = np.full(24, np.inf)
        if not hasattr(self, 'generation_counter'):
            self.generation_counter = 0

        self.generation_counter += 1

        # Compute improvement and normalize by problem scale
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        problem_scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
        relative_improvement = improvement / problem_scale if problem_scale > 0 else 0.0

        # Difficulty-aware scaling: amplify rewards for harder problems
        # Large errors (multi-modal/deceptive) need stronger signals
        if self.f_opt > 1.0:
            difficulty_mult = min(10.0, np.log10(self.f_opt + 1.0) + 1.0)
        elif self.f_opt > 1e-3:
            difficulty_mult = 2.0
        elif self.f_opt > 1e-6:
            difficulty_mult = 1.5
        else:
            difficulty_mult = 1.0

        # Compute comparative rank of this operator vs others in recent history
        op = self.current_operator
        recent_rewards = []
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) > 0:
                recent_rewards.append((i, np.mean(self.operator_rewards[i][-5:])))
            else:
                recent_rewards.append((i, 0.0))

        # Rank operators by recent performance (1 = best)
        sorted_by_perf = sorted(recent_rewards, key=lambda x: x[1], reverse=True)
        rank_map = {op_idx: rank + 1 for rank, (op_idx, _) in enumerate(sorted_by_perf)}
        current_rank = rank_map[op]

        # Rank-based reward: top rank gets high reward, bottom gets penalty
        rank_score = (self.num_operators - current_rank) / max(self.num_operators - 1, 1)
        rank_score = 2.0 * rank_score - 1.0  # Map to [-1, 1]

        # Improvement-based component (not log-compressed for large improvements)
        if improvement > 0:
            # Use sqrt scaling to preserve discrimination at large errors
            improvement_score = np.sqrt(improvement) / (np.sqrt(problem_scale) + 1.0)
            improvement_score = min(improvement_score, 5.0)
        else:
            improvement_score = 0.0

        # Streak tracking: reward consistent performers
        if improvement > 1e-15:
            self.operator_streak[op] += 1
        else:
            self.operator_streak[op] = max(0, self.operator_streak[op] - 1)

        streak_bonus = np.tanh(self.operator_streak[op] / 10.0) * 0.5

        # Diversity contribution reward
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Check if this operator helped maintain/increase diversity
        diversity_bonus = 0.0
        if hasattr(self, 'prev_diversity'):
            if diversity > self.prev_diversity * 1.1:
                diversity_bonus = 0.3 * (diversity - self.prev_diversity) / (self.prev_diversity + 1e-10)
        self.prev_diversity = diversity

        # Combine all reward components
        combined_reward = (
            0.4 * rank_score +
            0.4 * improvement_score * difficulty_mult +
            0.1 * streak_bonus +
            0.1 * diversity_bonus
        )
        combined_reward = float(np.clip(combined_reward, -5.0, 10.0))

        # Update rank history
        self.operator_rank_history[op].append(current_rank)
        if len(self.operator_rank_history[op]) > 20:
            self.operator_rank_history[op].pop(0)

        # Update reward window
        self.operator_rewards[op].append(combined_reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        # Update Beta distribution parameters for Thompson Sampling
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))

            # Use rank consistency as confidence multiplier
            avg_rank = np.mean(self.operator_rank_history[op])
            rank_confidence = 1.0 / (1.0 + avg_rank / self.num_operators)

            effective_mean = mean_reward * (0.5 + 0.5 * rank_confidence)
            denom = max(var_reward * n + 1e-10, 1e-10)

            self.alpha[op] = float(max(0.5, effective_mean * (effective_mean * (n - 1) / denom + 1)))
            self.beta[op] = float(max(0.5, (1 - effective_mean) * ((n - 1) * (1 - effective_mean) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
                self.beta[op] = 1.0
            else:
                self.alpha[op] = 1.0
                self.beta[op] = 1.0 - sum_reward + 1.0
```

# --- From variant_10_idea_0.py (3 wins) ---
```python
def _update_operator_rewards(self):
        """Credit assignment via elite-relative progress and adaptive scaling for hard tasks."""
        # Initialize tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay
        decay = 0.9
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Elite-relative progress: count how many trials beat current best
        elite = self.f_opt
        trial_array = np.asarray(self.trial_fitness).flatten()
        n_trials = len(trial_array)
        if n_trials > 0:
            beaten = np.sum(trial_array < elite)
            elite_progress = beaten / n_trials
        else:
            elite_progress = 0.0

        # Log-scaled improvement reward (crushes large errors better)
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        if improvement > 0:
            base_reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            base_reward = 0.0

        # Combine: 70% elite-relative progress + 30% absolute improvement
        # This provides dense signals even on hard tasks with rare global improvements
        reward = 0.7 * elite_progress * 5.0 + 0.3 * base_reward

        # Task-difficulty-adaptive scaling: harder tasks get exploration boost
        # The worst tasks have errors ~1e+1 to 5e+1, so scale rewards accordingly
        task_difficulty = max(1.0, np.log10(max(abs(self.f_opt), 1e-3) + 1.0))
        difficulty_scale = 1.0 + 0.5 * min(task_difficulty, 5.0)
        reward *= difficulty_scale

        # Stagnation bonus: when stuck, reward operators that explore diverse regions
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        if is_stagnant:
            reward *= 1.5

        # Breakthrough bonus for new global best
        if improvement > 0:
            reward += 2.0

        # Update cumulative reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

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