Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_operator_thompson` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.581447e-02            5.876238e-02            6.323210e-02            5.745642e-02            5.756991e-02            6.005383e-02            6.214008e-02            6.015624e-02            6.172896e-02            -inf                    7.314142e-02            
1      4.225025e-02            4.542068e-02            4.969092e-02            4.808402e-02            4.657710e-02            4.899237e-02            5.179020e-02            5.538659e-02            5.682540e-02            -inf                    1.015104e-01            
2      1.550907e-01            5.203090e-01            1.961681e+00            1.642345e-01            5.872776e-01            3.180154e+00            3.391477e+00            3.375803e+00            3.580949e+00            -inf                    7.680603e-01            
3      1.731658e-01            1.333618e-01            2.488819e+00            1.391312e-01            2.093085e+00            2.475451e+00            2.228102e+00            2.655189e+00            2.649112e+00            -inf                    2.995461e+00            
4      5.133402e-01            5.178418e-01            5.576520e-01            5.112692e-01            8.680149e-01            5.709746e-01            5.617651e-01            5.581510e-01            5.672358e-01            -inf                    6.734298e-01            
5      1.328620e-04            1.000615e-04            9.948460e-05            8.873252e-05            8.500448e-05            1.021193e-04            1.017025e-04            1.149041e-04            1.072950e-04            -inf                    1.681374e-02            
6      1.577348e-02            8.950167e+00            3.334721e+01            1.530164e-02            1.079437e-01            2.267033e+01            2.361222e+01            3.240079e+01            2.822876e+01            -inf                    3.507376e+01            
7      8.061778e-02            7.609630e-02            7.970480e-02            8.178058e-02            1.278047e-01            2.363198e+00            8.257190e-02            1.331521e+00            8.130702e-02            -inf                    3.793619e-01            
8      2.568841e-01            4.574179e-01            4.998633e-01            2.538654e-01            1.943365e+00            2.106273e+00            1.690536e+00            1.923851e+00            2.104757e+00            -inf                    8.572061e-01            
9      9.359551e-02            4.163806e+00            7.535148e+00            8.506383e-02            7.319709e+00            7.660810e+00            7.418389e+00            7.889568e+00            7.819214e+00            -inf                    7.409912e+00            
10     1.004256e+01            8.940326e+00            9.684684e+00            1.047646e+01            8.653625e+00            7.200252e+00            7.934284e+00            7.910243e+00            7.105611e+00            -inf                    8.117524e+00            
11     1.611530e+01            1.281704e+01            1.085281e+01            1.451933e+01            1.400915e+01            1.076571e+01            1.092344e+01            8.221225e+00            1.050841e+01            -inf                    5.383755e+00            
12     1.431520e+01            1.297381e+01            1.222702e+01            1.762433e+01            8.939910e+00            8.526223e+00            1.095614e+01            6.641359e+00            1.269179e+01            -inf                    5.956158e+00            
13     3.185759e+00            2.789122e+00            1.919930e+00            3.026298e+00            2.211029e+00            2.182727e+00            2.084760e+00            1.923603e+00            1.976348e+00            -inf                    1.847775e+00            
14     2.765349e+00            2.834012e+00            2.813872e+00            2.718750e+00            2.733988e+00            2.828214e+00            2.696957e+00            2.724819e+00            2.949190e+00            -inf                    2.626068e+00            
15     2.708644e+00            2.277586e+00            2.638398e+00            2.679435e+00            2.325596e+00            2.423051e+00            2.278545e+00            2.419303e+00            2.404506e+00            -inf                    2.654429e+00            
16     7.913191e+01            2.043336e+02            1.314105e+02            8.043844e+01            9.801027e+01            1.372759e+02            1.527390e+02            1.884686e+02            1.227783e+02            -inf                    2.902539e+02            
17     4.045684e+01            3.132338e+01            3.731292e+01            4.528543e+01            4.626884e+01            1.774100e+02            9.188163e+01            3.966753e+01            3.813391e+01            -inf                    3.013088e+01            
18     1.283604e+01            1.245526e+01            1.410596e+01            1.556858e+01            1.042984e+01            7.979137e+00            9.352764e+00            7.461971e+00            9.593734e+00            -inf                    1.776004e+01            
19     1.339200e+01            1.525323e+01            1.663146e+01            1.348480e+01            1.299295e+01            1.573697e+01            1.452660e+01            1.532984e+01            1.369206e+01            -inf                    1.187693e+01            
20     1.828716e+01            1.968863e+01            2.179752e+01            2.047384e+01            1.951894e+01            1.831992e+01            1.881575e+01            1.839348e+01            1.777304e+01            -inf                    2.302885e+01            
21     4.361961e+00            4.377284e+00            4.439227e+00            4.297752e+00            4.418597e+00            4.288791e+00            4.311801e+00            4.332856e+00            4.345082e+00            -inf                    4.530841e+00            
22     8.522054e+00            5.327929e+00            9.042386e+00            9.341992e+00            9.336476e+00            4.698478e+00            7.087396e+00            3.889920e+00            5.762754e+00            -inf                    1.060903e+01            
23     1.942306e+01            2.234369e+01            2.044900e+01            2.196253e+01            1.850371e+01            2.512690e+01            1.743482e+01            2.045511e+01            1.953288e+01            -inf                    2.401811e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=5.581447e-02)
Task  1: original.py  (error=4.225025e-02)
Task  2: original.py  (error=1.550907e-01)
Task  3: variant_01_idea_0.py  (error=1.333618e-01)
Task  4: variant_03_idea_0.py  (error=5.112692e-01)
Task  5: variant_04_idea_0.py  (error=8.500448e-05)
Task  6: variant_03_idea_0.py  (error=1.530164e-02)
Task  7: variant_01_idea_0.py  (error=7.609630e-02)
Task  8: variant_03_idea_0.py  (error=2.538654e-01)
Task  9: variant_03_idea_0.py  (error=8.506383e-02)
Task 10: variant_08_idea_0.py  (error=7.105611e+00)
Task 11: variant_10_idea_0.py  (error=5.383755e+00)
Task 12: variant_10_idea_0.py  (error=5.956158e+00)
Task 13: variant_10_idea_0.py  (error=1.847775e+00)
Task 14: variant_10_idea_0.py  (error=2.626068e+00)
Task 15: variant_01_idea_0.py  (error=2.277586e+00)
Task 16: original.py  (error=7.913191e+01)
Task 17: variant_10_idea_0.py  (error=3.013088e+01)
Task 18: variant_07_idea_0.py  (error=7.461971e+00)
Task 19: variant_10_idea_0.py  (error=1.187693e+01)
Task 20: variant_08_idea_0.py  (error=1.777304e+01)
Task 21: variant_05_idea_0.py  (error=4.288791e+00)
Task 22: variant_07_idea_0.py  (error=3.889920e+00)
Task 23: variant_06_idea_0.py  (error=1.743482e+01)

WIN COUNTS:
  variant_10_idea_0.py: 6 wins
  original.py: 4 wins
  variant_03_idea_0.py: 4 wins
  variant_01_idea_0.py: 3 wins
  variant_08_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_04_idea_0.py: 1 wins
  variant_05_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using Upper Confidence Bound (UCB) with optimistic initialization."""
        total_selections = np.sum(self.selection_counts)

        # Initialize with small epsilon reward if no data
        if total_selections == 0:
            self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            # Compute mean rewards for each operator
            mean_rewards = np.zeros(self.num_operators)
            valid_ops = np.zeros(self.num_operators, dtype=bool)

            for i in range(self.num_operators):
                if len(self.operator_rewards[i]) > 0:
                    mean_rewards[i] = np.mean(self.operator_rewards[i])
                    valid_ops[i] = True

            # UCB exploration bonus: sqrt(2 * ln(total) / count)
            exploration = np.zeros(self.num_operators)
            nonzero_mask = self.selection_counts > 0
            exploration[nonzero_mask] = np.sqrt(
                2.0 * np.log(max(total_selections, 1)) / np.maximum(self.selection_counts[nonzero_mask], 1)
            )

            # Optimistic initialization: small boost to operators with no data
            exploration[~nonzero_mask] = np.sqrt(2.0 * np.log(max(total_selections, 1)) + 1)

            # UCB score = mean reward + exploration bonus
            ucb_scores = mean_rewards + exploration

            # If some operators have no data, prioritize them initially
            if np.any(~valid_ops):
                unvisited = np.where(~valid_ops)[0]
                ucb_scores[unvisited] += np.max(ucb_scores[valid_ops]) + 1.0

            # Select operator with highest UCB score
            self.current_operator = int(np.argmax(ucb_scores))

        self.selection_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_03_idea_0.py (4 wins) ---
```python
def _select_operator_thompson(self):
        """Context-aware Thompson Sampling using problem structure features."""
        # Extract meta-features describing current problem state
        fitness_scale = max(abs(self.f_opt), 1.0)
        fitness_scale_log = np.log1p(fitness_scale)

        pc_norm = float(np.linalg.norm(self.pc))

        eigvals = np.linalg.eigvalsh(self.C)
        cond = float(np.max(eigvals) / max(float(np.min(eigvals)), 1e-10))
        cond_log = np.log1p(cond)

        pop_diversity = float(np.mean(np.std(self.population, axis=0)))
        expected_range = float(self.ub[0] - self.lb[0])
        diversity_ratio = float(np.clip(pop_diversity / (expected_range + 1e-10), 0.0, 1.0))

        improvement = float(getattr(self, 'improvement_ema', 0.0))

        gen = float(self.generation)
        phase = float(np.clip(gen / max(1.0, float(self.max_stagnation)), 0.0, 1.0))

        meta_features = np.array([fitness_scale_log, pc_norm, cond_log, diversity_ratio, improvement, phase])

        # Initialize context tracking
        if not hasattr(self, 'context_history'):
            self.context_history = {op: [] for op in range(self.num_operators)}
            self.context_rewards = {op: [] for op in range(self.num_operators)}

        # Compute context-similarity weighted reward for each operator
        operator_context_scores = np.zeros(self.num_operators)

        for op in range(self.num_operators):
            if len(self.context_history[op]) >= 3:
                contexts = np.array(self.context_history[op])
                rewards = np.array(self.context_rewards[op])

                # Gaussian kernel similarity to current context
                diffs = contexts - meta_features
                # Normalize by feature ranges for fair comparison
                feature_ranges = np.array([5.0, 10.0, 10.0, 0.5, 5.0, 1.0])
                similarities = np.exp(-np.sum((diffs / (feature_ranges + 1e-10))**2, axis=1) * 0.5)

                # Weight by recency (exponential decay)
                n = len(rewards)
                recency_weights = np.array([0.5 ** ((n - 1 - i) / 5.0) for i in range(n)])

                weighted_rewards = similarities * recency_weights * np.maximum(rewards, 0.0)
                operator_context_scores[op] = float(np.sum(weighted_rewards))

        # Combine standard Thompson sampling with context-aware scores
        # Context scores provide bias toward historically successful operators for similar problem states
        context_boost = np.zeros(self.num_operators)
        max_score = float(np.max(operator_context_scores))
        if max_score > 1e-10:
            context_boost = operator_context_scores / (max_score + 1e-10) * 2.0

        # Apply boost to Beta distribution parameters
        boosted_alpha = self.alpha + context_boost
        boosted_beta = self.beta.copy()

        # Thompson sampling from boosted distributions
        samples = np.random.beta(boosted_alpha, boosted_beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1

        # Record current context and reward for future decisions
        op = self.current_operator
        reward = float(max(0.0, getattr(self, 'f_opt_prev', self.f_opt) - self.f_opt))

        self.context_history[op].append(meta_features)
        self.context_rewards[op].append(reward)

        max_contexts = 50
        if len(self.context_history[op]) > max_contexts:
            self.context_history[op] = self.context_history[op][-max_contexts:]
            self.context_rewards[op] = self.context_rewards[op][-max_contexts:]

        return self.current_operator
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Thompson Sampling with long-term memory and entropy-promoting exploration."""
        n_ops = self.num_operators

        # Long-term reward tracking: cumulative log-reward sum and count
        if not hasattr(self, 'lt_alpha'):
            self.lt_alpha = np.ones(n_ops)
            self.lt_beta = np.ones(n_ops)
            self.lt_reward_sum = np.zeros(n_ops)
            self.lt_reward_count = np.zeros(n_ops)

        # Compute improvement and diversity reward
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Log-scaled reward (handles both large and small improvements)
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.15 * diversity)
        reward = float(np.clip(reward, -5.0, 5.0))

        op = self.current_operator
        self.lt_reward_sum[op] += reward
        self.lt_reward_count[op] += 1.0

        # Update Beta parameters using empirical Bayes with Laplace smoothing
        epsilon = 1e-6
        for i in range(n_ops):
            if self.lt_reward_count[i] >= 3.0:
                mean_r = self.lt_reward_sum[i] / max(self.lt_reward_count[i], 1.0)
                var_r = 1.0 / max(self.lt_reward_count[i], 1.0)
                mean_r = np.clip(mean_r, epsilon, 1.0 - epsilon)

                denom = max(var_r * self.lt_reward_count[i] + epsilon, epsilon)
                shape1 = max(mean_r * (mean_r * (self.lt_reward_count[i] - 1.0) / denom + 1.0), 1.0)
                shape2 = max((1.0 - mean_r) * ((self.lt_reward_count[i] - 1.0) * (1.0 - mean_r) / denom + 1.0), 1.0)

                self.lt_alpha[i] = float(shape1)
                self.lt_beta[i] = float(shape2)
            elif self.lt_reward_count[i] >= 1.0:
                total = float(np.sum(self.lt_reward_sum))
                if total > 0:
                    self.lt_alpha[i] = 1.0 + max(self.lt_reward_sum[i], 0.0)
                    self.lt_beta[i] = 1.0 + max(-self.lt_reward_sum[i], 0.0)
                else:
                    self.lt_alpha[i] = 1.0
                    self.lt_beta[i] = 1.5

        # Entropy-maximizing exploration bonus
        # Inversely proportional to selection count to promote under-explored operators
        exploration_bonus = np.zeros(n_ops)
        total_selections = float(np.sum(self.operator_counts) + 1.0)
        for i in range(n_ops):
            sel_frac = self.operator_counts[i] / total_selections
            # Bonus peaks for rarely-selected operators, decays for frequently-selected ones
            exploration_bonus[i] = 2.0 * np.exp(-5.0 * sel_frac)

        # Sample from Beta and add exploration bonus
        samples = np.random.beta(self.lt_alpha, self.beta)
        samples += exploration_bonus

        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Thompson Sampling with entropy-regularized exploration bonus."""
        # Compute selection probabilities (softmax of log mean rewards)
        log_rewards = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) >= 3:
                log_rewards[i] = np.log1p(max(np.mean(self.operator_rewards[i]), 1e-15))

        # Entropy bonus inversely proportional to selection count
        total_selections = np.sum(self.operator_counts) + 1.0
        entropy_bonus = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            p_select = (self.operator_counts[i] + 0.1) / (total_selections + 0.5 * self.num_operators)
            p_select = np.clip(p_select, 1e-6, 1.0)
            entropy_bonus[i] = -np.log(p_select + 1e-10) / max(np.log(total_selections + 1), 1.0)

        # Combined scores with entropy regularization
        scores = log_rewards + 0.5 * entropy_bonus

        # Softmax to get selection probabilities
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= np.sum(probs)

        # Sample from categorical distribution
        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using variance-weighted adaptive selection."""
        rewards = np.array([np.mean(r) if len(r) > 0 else 0.0 for r in self.operator_rewards.values()])
        reward_vars = np.array([np.var(r) if len(r) > 1 else 1.0 for r in self.operator_rewards.values()])

        reward_vars = np.clip(reward_vars, 1e-6, None)

        # Temperature inversely proportional to best reward — higher when stuck
        best_reward = np.max(rewards)
        temp = 5.0 if best_reward < 1.0 else max(0.5, 2.0 - best_reward)

        # Variance-weighted exploration bonus
        var_bonus = 0.5 * np.sqrt(reward_vars)

        # Compute adaptive scores
        scores = rewards + var_bonus

        # Softmax with temperature
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / temp)
        probs = exp_scores / np.sum(exp_scores)

        probs = np.clip(probs, 1e-10, 1.0)
        probs /= np.sum(probs)

        # Force exploration if selection is too concentrated
        if np.max(probs) > 0.8:
            probs = 0.7 * probs + 0.3 / self.num_operators

        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using deterministic round-robin with periodic random perturbation."""
        # Deterministic cycle: ensure each operator gets equal opportunity per task
        cycle_position = self.generation % self.num_operators

        # Small probability of random perturbation to maintain some diversity
        if hasattr(self, 'total_selections'):
            self.total_selections += 1
        else:
            self.total_selections = 0

        # 10% chance of random selection to prevent complete determinism
        if np.random.random() < 0.1:
            self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            self.current_operator = int(cycle_position)

        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using empirical performance-weighted sampling."""
        num_ops = self.num_operators

        # Compute empirical scores based on historical performance
        scores = np.zeros(num_ops)
        for op in range(num_ops):
            rewards = self.operator_rewards[op]
            if len(rewards) >= 3:
                # Use mean reward as primary score
                mean_r = np.mean(rewards)
                # Penalize high variance (prefer consistent operators)
                std_r = np.std(rewards) + 1e-10
                consistency = 1.0 / (1.0 + std_r)
                scores[op] = mean_r * consistency
            elif len(rewards) >= 1:
                scores[op] = np.mean(rewards) * 0.5
            else:
                scores[op] = 0.0

        # Add baseline quality from operator_counts (exploitation bonus)
        total_counts = np.sum(self.operator_counts) + 1e-10
        exploitation_bonus = 0.3 * (self.operator_counts / total_counts)
        scores = scores + exploitation_bonus

        # Convert to probability distribution
        scores_shifted = scores - np.min(scores) + 0.1
        probs = scores_shifted / np.sum(scores_shifted)

        # Sample from weighted distribution
        cumsum = np.cumsum(probs)
        r = np.random.random()

        selected = 0
        for op in range(num_ops):
            if r <= cumsum[op]:
                selected = op
                break

        self.current_operator = int(selected)
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
```

# --- From variant_10_idea_0.py (6 wins) ---
```python
def _select_operator_thompson(self):
        """Select operator using softmax over absolute best-fitness per operator."""
        # Track best fitness achieved by each operator
        if not hasattr(self, 'operator_best_fitness'):
            self.operator_best_fitness = np.full(self.num_operators, np.inf)

        # Update best fitness for current operator
        op = self.current_operator
        if self.f_opt < self.operator_best_fitness[op]:
            self.operator_best_fitness[op] = self.f_opt

        # Compute softmax probabilities from negative best fitness
        # Lower (more negative) is better, so use -best_fitness
        neg_best = -self.operator_best_fitness

        # Add small noise to break ties and ensure valid softmax input
        noise = np.random.uniform(0, 1e-10, self.num_operators)
        neg_best = neg_best + noise

        # Adaptive temperature: lower temp when stagnating (explore more)
        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
        temperature = max(0.01, 1.0 - 0.9 * stagnation_ratio)

        # Compute softmax probabilities
        neg_best_shifted = neg_best - np.max(neg_best)  # for numerical stability
        exp_scores = np.exp(neg_best_shifted / temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Sample operator based on probabilities
        cumsum = np.cumsum(probs)
        r = np.random.random()
        self.current_operator = int(np.searchsorted(cumsum, r))

        # Update selection counts
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