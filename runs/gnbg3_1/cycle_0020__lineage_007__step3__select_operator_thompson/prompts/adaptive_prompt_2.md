Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_operator_thompson` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.899845e-02            5.724736e-02            6.521905e-02            5.586675e-02            5.670716e-02            6.931975e-02            5.763501e-02            6.150785e-02            6.567155e-02            6.378041e-02            6.277033e-02            
1      4.273601e-02            4.589026e-02            5.116708e-02            4.407456e-02            4.739832e-02            7.034781e-02            4.859399e-02            4.823721e-02            4.908054e-02            5.343368e-02            5.247993e-02            
2      1.706168e-01            2.046336e-01            4.308879e+00            1.481311e-01            5.591328e-01            3.760405e+00            1.545295e-01            5.991470e-01            3.659959e+00            3.699534e+00            3.983671e+00            
3      1.380842e-01            1.079541e+00            2.635947e+00            3.014626e-01            2.056242e+00            2.887360e+00            1.318742e-01            4.009582e-01            2.558880e+00            2.409887e+00            2.603810e+00            
4      5.141570e-01            5.314396e-01            5.796041e-01            5.099703e-01            5.239865e-01            1.333743e+00            5.100389e-01            5.434526e-01            6.202613e-01            5.638800e-01            6.005336e-01            
5      8.859500e-05            7.793102e-05            8.729735e-05            8.935549e-05            1.097650e-04            4.534402e+01            9.128221e-05            1.314152e-04            1.142061e-04            1.188463e-04            1.314407e-04            
6      1.583771e-02            1.350820e+01            3.133881e+01            2.061058e-02            1.945559e+01            3.686290e+01            1.540441e-02            1.519337e+01            3.060193e+01            3.221068e+01            2.781417e+01            
7      7.836027e-02            7.828894e-02            1.555882e-01            7.865877e-02            7.918196e-02            2.329494e+00            8.002438e-02            7.766288e-02            3.062559e-01            7.881513e-02            2.613247e-01            
8      2.635268e-01            1.360285e+00            1.851714e+00            2.754427e-01            2.913011e-01            1.774744e+00            3.858474e-01            3.632737e-01            1.643312e+00            1.771479e+00            1.691459e+00            
9      6.832398e-02            6.912795e+00            7.511698e+00            4.656948e+00            7.294911e+00            8.248256e+00            1.411377e-01            7.164857e+00            7.388482e+00            7.494435e+00            7.546070e+00            
10     8.490591e+00            8.703590e+00            7.773818e+00            8.316266e+00            7.995711e+00            1.734494e+01            9.440314e+00            8.451648e+00            9.152878e+00            8.284181e+00            7.380021e+00            
11     7.714532e+00            1.143725e+01            9.899662e+00            8.938761e+00            5.382491e+00            2.323719e+01            1.773889e+01            6.521071e+00            8.706200e+00            1.296040e+01            6.655611e+00            
12     8.692513e+00            1.166853e+01            8.504147e+00            8.268150e+00            8.624891e+00            1.767192e+01            1.090039e+01            8.687102e+00            9.640972e+00            8.176783e+00            1.062374e+01            
13     1.851094e+00            2.623742e+00            2.111217e+00            2.184726e+00            2.219706e+00            2.629434e+00            2.647059e+00            2.447166e+00            2.198163e+00            2.178673e+00            1.898590e+00            
14     2.748222e+00            2.736808e+00            2.834221e+00            2.736869e+00            2.759215e+00            2.948969e+00            2.626274e+00            2.741684e+00            2.794524e+00            2.751947e+00            2.858730e+00            
15     2.584713e+00            2.470917e+00            2.240005e+00            2.142408e+00            2.472547e+00            1.772386e+00            2.413956e+00            2.137227e+00            2.134657e+00            2.366708e+00            2.321216e+00            
16     5.178944e+01            1.091534e+02            1.373111e+02            8.533395e+01            9.293175e+01            1.604960e+02            7.235722e+01            1.001656e+02            2.308799e+02            1.512013e+02            1.412773e+02            
17     3.570907e+01            3.341542e+01            2.866914e+02            1.245406e+02            2.999000e+01            7.098276e+02            1.826483e+02            3.205690e+01            2.938745e+01            9.192738e+01            9.997567e+01            
18     1.113497e+01            1.115271e+01            8.835609e+00            1.017675e+01            9.314860e+00            1.729740e+01            9.860965e+00            1.153250e+01            9.333314e+00            9.164142e+00            9.694921e+00            
19     1.299883e+01            1.277108e+01            1.556455e+01            1.858178e+01            1.285104e+01            2.265472e+01            1.445740e+01            1.649960e+01            1.587813e+01            1.509848e+01            1.566925e+01            
20     1.844703e+01            1.935830e+01            1.955733e+01            1.824072e+01            1.714294e+01            2.143805e+01            1.765489e+01            1.901797e+01            1.808253e+01            1.646204e+01            1.864685e+01            
21     4.339523e+00            4.388682e+00            4.427989e+00            4.303434e+00            4.391364e+00            4.372014e+00            4.415960e+00            4.320587e+00            4.326511e+00            4.342826e+00            4.381484e+00            
22     9.802064e+00            7.475339e+00            7.760827e+00            6.567711e+00            6.092728e+00            9.028616e+00            9.282475e+00            4.734623e+00            8.359906e+00            6.976189e+00            9.163307e+00            
23     2.078576e+01            1.845023e+01            2.164187e+01            1.936330e+01            1.672669e+01            2.534262e+01            1.938969e+01            2.095004e+01            2.239208e+01            1.792090e+01            2.277947e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_03_idea_0.py  (error=5.586675e-02)
Task  1: original.py  (error=4.273601e-02)
Task  2: variant_03_idea_0.py  (error=1.481311e-01)
Task  3: variant_06_idea_0.py  (error=1.318742e-01)
Task  4: variant_03_idea_0.py  (error=5.099703e-01)
Task  5: variant_01_idea_0.py  (error=7.793102e-05)
Task  6: variant_06_idea_0.py  (error=1.540441e-02)
Task  7: variant_07_idea_0.py  (error=7.766288e-02)
Task  8: original.py  (error=2.635268e-01)
Task  9: original.py  (error=6.832398e-02)
Task 10: variant_10_idea_0.py  (error=7.380021e+00)
Task 11: variant_04_idea_0.py  (error=5.382491e+00)
Task 12: variant_09_idea_0.py  (error=8.176783e+00)
Task 13: original.py  (error=1.851094e+00)
Task 14: variant_06_idea_0.py  (error=2.626274e+00)
Task 15: variant_05_idea_0.py  (error=1.772386e+00)
Task 16: original.py  (error=5.178944e+01)
Task 17: variant_08_idea_0.py  (error=2.938745e+01)
Task 18: variant_02_idea_0.py  (error=8.835609e+00)
Task 19: variant_01_idea_0.py  (error=1.277108e+01)
Task 20: variant_09_idea_0.py  (error=1.646204e+01)
Task 21: variant_03_idea_0.py  (error=4.303434e+00)
Task 22: variant_07_idea_0.py  (error=4.734623e+00)
Task 23: variant_04_idea_0.py  (error=1.672669e+01)

WIN COUNTS:
  original.py: 5 wins
  variant_03_idea_0.py: 4 wins
  variant_06_idea_0.py: 3 wins
  variant_01_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  variant_09_idea_0.py: 2 wins
  variant_10_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins
  variant_08_idea_0.py: 1 wins
  variant_02_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (2 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using Adaptive Epsilon-Greedy strategy."""
        # Compute empirical mean reward for each operator
        operator_avg = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) > 0:
                operator_avg[i] = float(np.mean(self.operator_rewards[i]))

        # Adaptive epsilon: starts high (exploration), decreases over time (exploitation)
        total_selections = float(np.sum(self.operator_counts))
        epsilon = 1.0 / (1.0 + total_selections / max(self.dim, 10.0))
        epsilon = float(np.clip(epsilon, 0.01, 0.5))

        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            # Explore: random operator with bias toward less-selected ones
            selection_probs = 1.0 / (self.operator_counts + 1.0)
            selection_probs /= np.sum(selection_probs)
            selection_probs = np.clip(selection_probs, 1e-10, 1.0 - 1e-10)
            self.current_operator = int(np.random.choice(self.num_operators, p=selection_probs))
        else:
            # Exploit: select operator with highest empirical reward
            self.current_operator = int(np.argmax(operator_avg))

        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_02_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using adaptive ensemble with performance-scaled softmax selection."""
        if not hasattr(self, 'op_scores'):
            self.op_scores = np.zeros(self.num_operators)
        if not hasattr(self, 'op_selections'):
            self.op_selections = np.zeros(self.num_operators)
        if not hasattr(self, 'op_last_selected'):
            self.op_last_selected = np.zeros(self.num_operators, dtype=int)

        # Compute per-operator reward from recent fitness improvement
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Log-scaled reward with performance scaling
        if improvement > 1e-15:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Apply performance-scaled update
        decay = 0.9
        self.op_scores = decay * self.op_scores + (1.0 - decay) * reward

        # Exploration bonus: reward operators not recently selected
        current_gen = self.generation
        for i in range(self.num_operators):
            generations_since = current_gen - self.op_last_selected[i]
            if generations_since > 5:
                self.op_scores[i] += 0.5 * (generations_since - 5) / 10.0

        # Diversity bonus when stagnant
        if self.stagnation_counter > self.max_stagnation // 3:
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
            if diversity < 0.05:
                for i in range(self.num_operators):
                    if self.selection_counts[i] < np.mean(self.selection_counts) * 0.5:
                        self.op_scores[i] += 1.0

        # Softmax selection with temperature
        scores = self.op_scores - np.max(self.op_scores) + 1e-10
        temperature = 0.5
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Sample from categorical distribution
        cumsum = np.cumsum(probs)
        r = np.random.random()
        self.current_operator = int(np.searchsorted(cumsum, r))
        self.current_operator = min(self.current_operator, self.num_operators - 1)

        self.operator_counts[self.current_operator] += 1
        self.op_selections[self.current_operator] += 1
        self.op_last_selected[self.current_operator] = current_gen

        return self.current_operator
```

# --- From variant_03_idea_0.py (4 wins) ---
```python
def _select_operator_thompson(self):
        """Difficulty-adaptive operator selection with progress tracking."""
        # Track recent best improvements across sliding window
        if not hasattr(self, 'progress_history'):
            self.progress_history = []

        current_improvement = max(0.0, self.f_opt_prev - self.f_opt)
        self.progress_history.append(current_improvement)

        # Keep window bounded
        window_size = 5 * self.dim
        if len(self.progress_history) > window_size:
            self.progress_history.pop(0)

        # Compute smoothed progress rate
        if len(self.progress_history) >= 3:
            recent = self.progress_history[-3:]
            avg_progress = np.mean(recent)
        else:
            avg_progress = current_improvement

        # Detect stagnation: no meaningful improvement in recent generations
        is_stagnant = (avg_progress < 1e-12 * max(abs(self.f_opt), 1.0)) and len(self.progress_history) >= self.dim

        # Compute operator quality scores from cumulative rewards
        quality_scores = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            if len(self.operator_rewards[op]) > 0:
                quality_scores[op] = np.mean(self.operator_rewards[op])

        if is_stagnant:
            # Exploration mode: select operator with lowest count or highest variance in rewards
            if np.sum(self.operator_counts) > self.dim * 3:
                # Favor under-explored operators
                inv_counts = 1.0 / (self.operator_counts + 1.0)
                self.current_operator = int(np.argmax(inv_counts))
            else:
                # Random exploration when very stagnant
                self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            # Exploitation mode: select operator with highest quality score
            if np.max(quality_scores) > 0:
                self.current_operator = int(np.argmax(quality_scores))
            else:
                # Fallback to least-used operator
                self.current_operator = int(np.argmin(self.operator_counts))

        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using entropy-driven Thompson Sampling with forced exploration."""
        # Compute entropy-based uncertainty for each operator
        entropy = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            a, b = max(self.alpha[i], 1e-10), max(self.beta[i], 1e-10)
            total = a + b
            e = np.log1p(total) - (a * np.log1p(a) + b * np.log1p(b)) / total
            entropy[i] = max(e, 0.0)

        # Normalize entropy to get exploration weights
        total_entropy = np.sum(entropy)
        if total_entropy > 1e-10:
            explore_weights = entropy / total_entropy
        else:
            explore_weights = np.ones(self.num_operators) / self.num_operators

        # Compute selection count penalty (favor less-selected operators)
        total_counts = np.sum(self.operator_counts) + 1e-10
        count_penalty = 1.0 - (self.operator_counts / total_counts)
        count_penalty = np.clip(count_penalty, 0.0, 1.0)

        # Combine exploitation (Thompson beta) + exploration (entropy + count)
        samples = np.random.beta(np.maximum(self.alpha, 1e-10), np.maximum(self.beta, 1e-10))
        combined_score = (
            0.4 * samples / (np.max(samples) + 1e-10) +
            0.4 * explore_weights / (np.max(explore_weights) + 1e-10) +
            0.2 * count_penalty / (np.max(count_penalty) + 1e-10)
        )

        # Forced exploration: if operator not selected in N generations, boost its score
        if hasattr(self, 'last_selected_gen'):
            gens_since_selection = self.generation - self.last_selected_gen
            forced_threshold = max(20, self.dim * 2)
            for i in range(self.num_operators):
                if gens_since_selection[i] > forced_threshold:
                    combined_score[i] *= 3.0
                    if gens_since_selection[i] > forced_threshold * 2:
                        combined_score[i] *= 2.0
        else:
            self.last_selected_gen = np.zeros(self.num_operators, dtype=int)

        self.current_operator = int(np.argmax(combined_score))
        self.operator_counts[self.current_operator] += 1
        self.last_selected_gen[self.current_operator] = self.generation
        return self.current_operator
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using diversity-guided regime detection."""
        # Compute regime indicators
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min, eig_max = np.min(eigvals), np.max(eigvals)
        cond = eig_max / (max(eig_min, 1e-10))
        eig_spread = eig_min / (eig_max + 1e-10)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        # Regime 1: Severely stuck (high error tasks need this)
        is_severe_trap = (diversity < 0.05) or (cond > 1e6) or (eig_spread < 1e-6)

        # Regime 2: Moderate convergence pressure
        is_converging = (rel_var < 0.001) or (diversity < 0.1)

        # Regime 3: Normal exploration
        is_normal = not (is_severe_trap or is_converging)

        # Map regimes to preferred operators (0=original, 1=archive, 2=diversity, 3=active, 4=temperature)
        if is_severe_trap:
            # Archive-guided (variant_01) for escaping severe local optima
            preferred = 1
        elif is_converging:
            # Diversity-sensitive (variant_06) or active CMA (variant_08)
            preferred = np.random.choice([2, 3])
        else:
            # Thompson sampling with bias toward successful operators
            preferred = None

        if preferred is not None:
            self.current_operator = int(preferred)
        else:
            # Fallback: Thompson sampling with regime-aware priors
            regime_prior = np.ones(self.num_operators)
            regime_prior[1] = 2.0  # boost archive for next cycle
            combined_alpha = self.alpha * regime_prior
            samples = np.random.beta(combined_alpha, self.beta)
            self.current_operator = int(np.argmax(samples))

        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_06_idea_0.py (3 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using entropy-weighted Thompson Sampling with diversity pressure."""
        # Compute entropy-based exploration control
        total_reward = np.sum(self.alpha) + np.sum(self.beta)
        if total_reward > self.num_operators * 2:
            # Compute entropy of reward distribution
            alpha_sum = np.sum(self.alpha)
            beta_sum = np.sum(self.beta)
            probs = self.alpha / (alpha_sum + 1e-10)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(self.num_operators)
            entropy_ratio = entropy / (max_entropy + 1e-10)

            # High entropy = explore more (widen distributions)
            # Low entropy = exploit (narrow distributions)
            exploration_scale = 0.5 + 2.0 * (1.0 - entropy_ratio)
            exploration_scale = np.clip(exploration_scale, 0.5, 3.0)
        else:
            exploration_scale = 1.5  # Initial exploration bias

        # Sample from scaled Beta distributions
        samples = np.random.beta(self.alpha * exploration_scale, self.beta * exploration_scale)

        # Apply diversity pressure: boost probability of under-explored operators
        if np.sum(self.selection_counts) > self.num_operators * 3:
            min_selections = np.min(self.selection_counts)
            selection_deficit = min_selections / (self.selection_counts + 1e-10)
            diversity_boost = 0.1 * selection_deficit
            samples = samples + diversity_boost * np.random.random(self.num_operators)

        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using adaptive softmax with stagnation-driven temperature."""
        # Compute mean reward for each operator
        qualities = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            if len(self.operator_rewards[op]) > 0:
                qualities[op] = float(np.mean(self.operator_rewards[op]))
            else:
                qualities[op] = -5.0  # Unexplored operators start pessimistic

        # Adaptive temperature: high when stagnant (explore), low when improving (exploit)
        stagnation_ratio = float(self.stagnation_counter) / max(float(self.max_stagnation), 1.0)
        stagnation_ratio = np.clip(stagnation_ratio, 0.0, 1.0)
        temp = 0.01 + 4.99 * stagnation_ratio
        temp = float(np.clip(temp, 0.01, 5.0))

        # Softmax with numerical stability
        max_q = float(np.max(qualities))
        exp_q = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            exp_q[op] = float(np.exp(np.clip(qualities[op] - max_q, -700.0, 700.0)))

        sum_exp = float(np.sum(exp_q))
        if sum_exp <= 0.0 or np.isnan(sum_exp) or np.isinf(sum_exp):
            probs = np.ones(self.num_operators) / self.num_operators
        else:
            probs = exp_q / sum_exp

        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        probs /= np.sum(probs)

        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using task-difficulty-aware Thompson Sampling."""
        # Track per-operator recent performance
        if not hasattr(self, 'op_reward_buffer'):
            self.op_reward_buffer = [[] for _ in range(self.num_operators)]

        # Compute current performance metrics
        current_error = abs(self.f_opt)
        log_error = np.log10(max(current_error, 1e-15))
        target_log = -8.0
        error_gap = max(0.0, log_error - target_log)

        # Stagnation detection
        is_stagnant = self.stagnation_counter > self.max_stagnation // 3

        # Compute task difficulty proxy (0=hard, 1=easy)
        if error_gap > 8.0:
            difficulty = 0.0  # Very hard
        elif error_gap > 4.0:
            difficulty = 0.25
        elif error_gap > 1.0:
            difficulty = 0.5
        else:
            difficulty = 0.75

        # Success rate for current operator
        recent_wins = 0
        recent_total = 0
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_wins += 1
            recent_total += 1
        success_rate = recent_wins / max(recent_total, 1)

        # Shape reward based on improvement and difficulty
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        if improvement > 0:
            base_reward = np.log1p(improvement * 1e10) / 10.0
        else:
            base_reward = -0.5 if is_stagnant else 0.0

        # Difficulty-adjusted reward: boost exploration on hard tasks
        if difficulty < 0.5:
            exploration_bonus = 2.0 * (0.5 - difficulty)
            base_reward += exploration_bonus

        # Normalize by operator selection count to prevent bias toward frequent operators
        op_count = max(self.operator_counts[self.current_operator], 1)
        normalized_reward = base_reward / np.sqrt(op_count)
        normalized_reward = np.clip(normalized_reward, -5.0, 5.0)

        # Update reward buffer
        self.op_reward_buffer[self.current_operator].append(normalized_reward)
        buffer_size = 15
        if len(self.op_reward_buffer[self.current_operator]) > buffer_size:
            self.op_reward_buffer[self.current_operator].pop(0)

        # Compute per-operator statistics
        op_means = np.zeros(self.num_operators)
        op_vars = np.zeros(self.num_operators)
        op_counts_snapshot = self.operator_counts.copy()

        for op in range(self.num_operators):
            if len(self.op_reward_buffer[op]) >= 3:
                op_means[op] = np.mean(self.op_reward_buffer[op])
                op_vars[op] = np.var(self.op_reward_buffer[op]) + 1e-10
            else:
                op_means[op] = 0.0
                op_vars[op] = 1.0

        # Thompson Sampling with performance-shaped Beta parameters
        # Higher mean reward -> higher alpha (more likely to be selected)
        # Lower variance -> more confident estimate
        alpha_samples = np.zeros(self.num_operators)
        beta_samples = np.zeros(self.num_operators)

        for op in range(self.num_operators):
            count = max(op_counts_snapshot[op], 1)
            confidence = 1.0 / np.sqrt(count)

            # Beta parameters shaped by performance
            alpha_base = max(1.0, 1.0 + op_means[op] * 5.0)
            beta_base = max(1.0, 1.0 - op_means[op] * 5.0)

            # Adjust by confidence (higher confidence = tighter distribution)
            alpha_param = alpha_base * (1.0 + confidence)
            beta_param = beta_base * (1.0 + confidence)

            alpha_samples[op] = alpha_param
            beta_samples[op] = beta_param

        # Sample from Beta distributions
        samples = np.array([np.random.beta(alpha_samples[op], beta_samples[op]) 
                            for op in range(self.num_operators)])

        # Exploration boost for hard tasks: occasionally sample uniformly
        if difficulty < 0.5 or is_stagnant:
            explore_prob = 0.3 if difficulty < 0.25 else 0.15
            if np.random.random() < explore_prob:
                samples = np.random.uniform(size=self.num_operators)

        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_09_idea_0.py (2 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using Thompson Sampling with diversity bonus for stuck populations."""
        # Compute effective diversity per operator (how often it produces distinct solutions)
        if not hasattr(self, 'operator_diversity_scores'):
            self.operator_diversity_scores = np.ones(self.num_operators)

        # Compute per-operator success rate vs expected random improvement
        expected_random_improvement = max(1e-10, 0.1 * (self.ub[0] - self.lb[0]) / self.sigma)
        success_rates = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if self.operator_counts[i] > 0:
                reward_sum = self.operator_cumulative_reward[i] / max(self.operator_decay_sum[i], 1.0)
                success_rates[i] = reward_sum / expected_random_improvement

        # Compute population diversity to detect stagnation
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Diversity bonus: boost operators not recently used when stuck
        time_since_used = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if hasattr(self, 'operator_last_used'):
                time_since_used[i] = self.generation - self.operator_last_used.get(i, 0)
            else:
                time_since_used[i] = self.generation

        # Apply diversity bonus when diversity is low (population stuck)
        is_stuck = (diversity < 0.05) or (self.stagnation_counter > self.max_stagnation // 4)
        if is_stuck:
            diversity_bonus = time_since_used / max(np.max(time_since_used), 1.0)
            diversity_bonus = np.clip(diversity_bonus, 0.0, 3.0)
        else:
            diversity_bonus = np.zeros(self.num_operators)

        # Thompson Sampling with success-rate transformed parameters
        alpha_posterior = np.ones(self.num_operators)
        beta_posterior = np.ones(self.num_operators)

        for i in range(self.num_operators):
            # Transform success rate to Beta parameters
            # Higher success rate -> higher alpha, lower beta
            sr = np.clip(success_rates[i], 0.0, 1.0)
            base_alpha = 1.0 + 10.0 * sr
            base_beta = 1.0 + 10.0 * (1.0 - sr)

            # Apply diversity bonus to encourage exploration of unused operators
            alpha_posterior[i] = max(1.0, base_alpha + diversity_bonus[i] * 5.0)
            beta_posterior[i] = max(1.0, base_beta - diversity_bonus[i] * 2.0)

        # Sample from Beta distributions
        samples = np.random.beta(alpha_posterior, beta_posterior)

        # Track last use
        if not hasattr(self, 'operator_last_used'):
            self.operator_last_used = {}
        self.operator_last_used[self.current_operator] = self.generation

        # Select operator with highest sample
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1

        # Update diversity scores based on operator effectiveness
        self.operator_diversity_scores *= 0.95
        self.operator_diversity_scores[self.current_operator] += 0.05 * diversity

        return self.current_operator
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using variance-adjusted Thompson Sampling with uncertainty scaling."""
        # Compute running statistics for each operator
        n_ops = self.num_operators
        mean_rewards = np.zeros(n_ops)
        var_rewards = np.zeros(n_ops)
        valid_counts = np.zeros(n_ops)

        for i in range(n_ops):
            rewards = self.operator_rewards[i]
            if len(rewards) >= 3:
                mean_rewards[i] = np.mean(rewards)
                var_rewards[i] = np.var(rewards)
                valid_counts[i] = len(rewards)
            elif len(rewards) >= 1:
                mean_rewards[i] = np.mean(rewards)
                var_rewards[i] = 1.0
                valid_counts[i] = len(rewards)
            else:
                mean_rewards[i] = 0.0
                var_rewards[i] = 1.0
                valid_counts[i] = 0.0

        # Wilson score confidence interval for more robust reward estimation
        # This handles small sample sizes better than raw mean
        wilson_scores = np.zeros(n_ops)
        z = 1.96  # 95% confidence
        for i in range(n_ops):
            if valid_counts[i] >= 3:
                p = np.clip(mean_rewards[i], 0.0, 1.0)
                n = valid_counts[i]
                denominator = 1.0 + z**2 / n
                center = (p + z**2 / (2*n)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
                wilson_scores[i] = center + margin  # Upper bound of CI
            elif valid_counts[i] >= 1:
                # For small samples, use log-transformed reward with uncertainty
                log_reward = np.log1p(max(mean_rewards[i], 1e-10))
                uncertainty = 2.0 / max(valid_counts[i], 1.0)
                wilson_scores[i] = log_reward + uncertainty
            else:
                # Unexplored operators get optimistic prior
                wilson_scores[i] = 2.0 + np.random.uniform(0, 0.5)

        # Compute effective exploration rate based on total samples
        total_samples = np.sum(self.operator_counts) + 1.0
        base_exploration = min(2.0, 20.0 / np.sqrt(total_samples))

        # Add variance-based exploration bonus
        # High variance operators may be under- or over-estimated
        max_var = max(np.max(var_rewards), 1e-10)
        variance_bonus = 1.5 * (var_rewards / (max_var + 1e-10))

        # Scale exploration by inverse sample count (less explored = more exploration)
        inv_counts = 1.0 / (valid_counts + 1.0)
        inv_counts_normalized = inv_counts / (np.sum(inv_counts) + 1e-10)

        # Combined exploration bonus
        exploration_bonus = base_exploration * inv_counts_normalized + 0.5 * variance_bonus

        # Compute final scores combining exploitation and exploration
        # Use log-transformed rewards for numerical stability
        log_rewards = np.log1p(np.clip(mean_rewards, 0.0, 10.0))
        scores = log_rewards + exploration_bonus

        # Thompson Sampling: sample from distribution proportional to scores
        # Use softmax with temperature for probabilistic selection
        temperature = max(0.1, 1.0 / (1.0 + np.mean(var_rewards) + 1e-10))
        exp_scores = np.exp((scores - np.max(scores)) / temperature)
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        # Sample from categorical distribution
        self.current_operator = int(np.random.choice(n_ops, p=probs))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
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