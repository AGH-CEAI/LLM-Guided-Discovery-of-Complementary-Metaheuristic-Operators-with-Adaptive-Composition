Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_mutation_operator_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      3.104189e-08            2.837569e-08            -inf                    2.894722e-08            2.719980e-08            -inf                    2.442261e-08            2.332697e-08            2.619455e-08            2.283911e-08            2.382443e-08            
1      1.166227e-03            1.196114e-03            -inf                    1.125118e-03            6.870043e-04            -inf                    1.187397e-03            1.458250e-03            1.270186e-03            1.004732e-03            1.996152e-03            
2      8.932702e-06            8.773895e-06            -inf                    8.198526e-06            8.298525e-06            -inf                    7.509540e-06            6.646339e-06            7.095150e-06            8.129553e-06            8.770656e-06            
3      2.293823e+00            2.523483e+00            -inf                    1.645803e+00            2.098366e+00            -inf                    2.000346e+00            2.400197e+00            2.261571e+00            1.658878e+00            2.191156e+00            
4      5.475169e-02            4.477762e-02            -inf                    4.626085e-02            5.168749e-02            -inf                    4.945362e-02            4.861452e-02            4.687642e-02            5.792167e-02            5.522389e-02            
5      5.927262e+01            1.829277e+01            -inf                    9.476183e+01            4.021858e+01            -inf                    3.881799e+01            1.184708e+02            5.825409e+01            7.152314e+01            2.639311e+01            
6      3.072392e+01            3.500945e+01            -inf                    2.920812e+01            2.859091e+01            -inf                    3.002129e+01            2.775171e+01            3.601110e+01            2.627034e+01            2.846257e+01            
7      2.981565e+00            2.761820e+00            -inf                    3.211026e+00            3.317019e+00            -inf                    2.330039e+00            2.328883e+00            2.823720e+00            2.528211e+00            3.714878e+00            
8      1.285245e-01            1.089066e-01            -inf                    1.451742e-01            1.041542e-01            -inf                    1.239755e-01            1.744754e-01            1.544349e-01            1.097428e-01            1.405737e-01            
9      8.570147e+00            8.475902e+00            -inf                    8.238364e+00            8.729165e+00            -inf                    8.527096e+00            8.718845e+00            8.114868e+00            8.628527e+00            8.774281e+00            
10     1.166087e-04            1.288209e-04            -inf                    1.272951e-04            1.320833e-04            -inf                    1.265889e-04            1.117897e-04            1.351687e-04            1.434278e-04            1.078755e-04            
11     3.303010e+01            2.486287e+01            -inf                    2.550964e+01            2.429688e+01            -inf                    2.906561e+01            2.589426e+01            2.734449e+01            2.554005e+01            4.193750e+01            
12     8.737248e-04            8.464193e-04            -inf                    1.186322e-03            1.256080e-03            -inf                    8.543531e-04            9.161059e-04            1.031221e-03            8.434817e-04            1.233907e-03            
13     2.683066e+00            2.718260e+00            -inf                    2.595041e+00            2.421337e+00            -inf                    2.636412e+00            3.331281e+00            2.996662e+00            2.872214e+00            2.186919e+00            
14     9.805063e+00            1.160240e+01            -inf                    9.643277e+00            7.253171e+00            -inf                    9.407938e+00            7.388118e+00            7.333913e+00            8.491092e+00            7.087242e+00            
15     2.584492e+00            2.572011e+00            -inf                    2.418182e+00            2.582395e+00            -inf                    2.526612e+00            2.335805e+00            2.455956e+00            2.519496e+00            2.582631e+00            
16     1.304408e+03            1.176621e+03            -inf                    1.637725e+03            1.418377e+03            -inf                    2.671304e+03            1.473196e+03            1.493526e+03            2.554097e+03            2.134920e+03            
17     3.462031e+02            1.344710e+03            -inf                    4.353900e+02            6.147003e+02            -inf                    5.850238e+01            3.416293e+02            2.579162e+02            7.361014e+01            1.488005e+03            
18     1.620901e+01            2.371697e+01            -inf                    2.322768e+01            2.111545e+01            -inf                    2.474163e+01            2.634717e+01            1.653627e+01            1.938098e+01            2.126128e+01            
19     3.237361e+02            3.285977e+02            -inf                    3.569954e+02            3.221802e+02            -inf                    3.199270e+02            3.475502e+02            3.159152e+02            3.071030e+02            3.109479e+02            
20     1.692127e+01            1.735888e+01            -inf                    1.731459e+01            1.628473e+01            -inf                    1.636881e+01            1.846383e+01            1.694290e+01            1.933696e+01            1.828187e+01            
21     5.157240e+00            5.030339e+00            -inf                    4.971565e+00            5.108712e+00            -inf                    5.039974e+00            4.970713e+00            5.044917e+00            4.989417e+00            5.076219e+00            
22     1.815550e+01            1.768489e+01            -inf                    1.755814e+01            1.791232e+01            -inf                    1.801774e+01            1.714071e+01            1.696035e+01            1.841744e+01            1.827256e+01            
23     5.514728e+01            5.733054e+01            -inf                    5.233365e+01            4.815428e+01            -inf                    5.977598e+01            4.584563e+01            4.801903e+01            5.579614e+01            5.002094e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_04_idea_0.py  (error=6.870043e-04)
Task  2: variant_07_idea_0.py  (error=6.646339e-06)
Task  3: variant_03_idea_0.py  (error=1.645803e+00)
Task  4: variant_01_idea_0.py  (error=4.477762e-02)
Task  5: variant_01_idea_0.py  (error=1.829277e+01)
Task  6: variant_09_idea_0.py  (error=2.627034e+01)
Task  7: variant_07_idea_0.py  (error=2.328883e+00)
Task  8: variant_04_idea_0.py  (error=1.041542e-01)
Task  9: variant_08_idea_0.py  (error=8.114868e+00)
Task 10: variant_10_idea_0.py  (error=1.078755e-04)
Task 11: variant_04_idea_0.py  (error=2.429688e+01)
Task 12: variant_09_idea_0.py  (error=8.434817e-04)
Task 13: variant_10_idea_0.py  (error=2.186919e+00)
Task 14: variant_10_idea_0.py  (error=7.087242e+00)
Task 15: variant_07_idea_0.py  (error=2.335805e+00)
Task 16: variant_01_idea_0.py  (error=1.176621e+03)
Task 17: variant_06_idea_0.py  (error=5.850238e+01)
Task 18: original.py  (error=1.620901e+01)
Task 19: variant_09_idea_0.py  (error=3.071030e+02)
Task 20: variant_04_idea_0.py  (error=1.628473e+01)
Task 21: variant_07_idea_0.py  (error=4.970713e+00)
Task 22: variant_08_idea_0.py  (error=1.696035e+01)
Task 23: variant_07_idea_0.py  (error=4.584563e+01)

WIN COUNTS:
  variant_07_idea_0.py: 5 wins
  variant_04_idea_0.py: 4 wins
  variant_01_idea_0.py: 3 wins
  variant_09_idea_0.py: 3 wins
  variant_10_idea_0.py: 3 wins
  variant_08_idea_0.py: 2 wins
  variant_03_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins
  original.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _select_mutation_operator_batch(self):
        """Select mutation operator using Thompson Sampling with Beta posterior distributions."""
        # Thompson Sampling: sample from Beta(successes+1, failures+1) for each operator
        # successes = operator_rewards, failures = operator_counts - operator_rewards
        successes = np.clip(self.operator_rewards, 0, self.operator_counts)
        failures = np.maximum(self.operator_counts - successes, 0)

        # Sample from Beta distribution for each operator (add 1 for prior)
        # Use shared random state for batch efficiency
        beta_samples = np.random.beta(successes + 1, failures + 1)

        # Handle numerical edge cases (all zeros -> uniform selection)
        if np.sum(beta_samples) == 0 or np.any(np.isnan(beta_samples)):
            return np.random.randint(0, self.n_strategies)

        # Select operator with highest sampled value
        selected = np.argmax(beta_samples)

        return selected
```

# --- From variant_03_idea_0.py (1 wins) ---
```python
def _select_mutation_operator_batch(self):
        """Select mutation operator using Thompson Sampling with fitness-aware exploration boost."""
        total_counts = np.sum(self.operator_counts) + 1e-10

        # Compute empirical success rates as Bayesian pseudo-rewards
        success_rates = self.operator_rewards / np.maximum(self.operator_counts, 1)

        # Detect exploration deficiency: if all operators have similar low success,
        # we're stuck in a local optimum and need aggressive exploration boost
        rate_variance = np.var(success_rates)
        rate_mean = np.mean(success_rates)

        # Thompson Sampling: sample from Beta distribution for each operator
        # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
        alpha = self.operator_rewards + 1.0
        beta = self.operator_counts - self.operator_rewards + 1.0

        # Clip to avoid numerical issues
        alpha = np.clip(alpha, 1e-10, 1e10)
        beta = np.clip(beta, 1e-10, 1e10)

        # Sample from Beta distribution for each operator
        sampled_values = np.random.beta(alpha, beta)

        # Exploration boost for stuck scenarios: add entropy-based bonus
        # High entropy = we don't know which operator is best = explore more
        if rate_variance < 0.01 and rate_mean < 0.3:
            # Stuck in local optimum: boost exploration by adding random noise to samples
            exploration_noise = np.random.uniform(0, 0.5, size=self.n_strategies)
            sampled_values += exploration_noise

        # Bias toward operators with higher variance in recent performance (novelty signal)
        # This helps escape local optima by favoring operators we haven't fully explored
        novelty_bonus = np.random.uniform(0, 0.2) * (1.0 / np.maximum(self.operator_counts, 1))
        sampled_values += novelty_bonus

        # Select operator with highest sampled value
        selected = int(np.argmax(sampled_values))

        # Ensure valid index
        selected = np.clip(selected, 0, self.n_strategies - 1)

        return selected
```

# --- From variant_04_idea_0.py (4 wins) ---
```python
def _select_mutation_operator_batch(self, fitness=None):
        """Select mutation operator using fitness-rank-aware adaptive selection.

        Key differences from UCB:
        - Computes per-individual operator probabilities based on fitness rank
        - Worst individuals get exploration-focused operators (rand/oppose)
        - Best individuals get exploitation-focused operators (best/elite)
        - Uses historical success rates to weight operator selection
        """
        np_pop = len(self.operator_rewards)

        # Compute historical performance scores for each operator
        # Avoid division by zero with max(1, count)
        perf_scores = self.operator_rewards / np.maximum(self.operator_counts, 1)

        # Normalize to [0, 1] range
        if perf_scores.max() > perf_scores.min():
            perf_norm = (perf_scores - perf_scores.min()) / (perf_scores.max() - perf_scores.min() + 1e-10)
        else:
            perf_norm = np.ones(np_pop) / np_pop

        # Compute exploration bonus based on operator counts (less-tried = explore more)
        total_counts = np.sum(self.operator_counts)
        exploration_bonus = 1.0 / np.log(np.maximum(self.operator_counts, 2))
        exploration_bonus = exploration_bonus / (exploration_bonus.sum() + 1e-10)

        # Blend: 60% performance, 40% exploration
        blended_scores = 0.6 * perf_norm + 0.4 * exploration_bonus

        # Final selection: weighted random with blended scores
        probs = np.clip(blended_scores, 1e-10, None)
        probs = probs / probs.sum()

        # Use weighted random selection (not argmax) for diversity
        selected = np.random.choice(np_pop, p=probs)

        # Clamp to valid range
        selected = int(np.clip(selected, 0, np_pop - 1))

        return selected
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _select_mutation_operator_batch(self):
        """Select mutation operator using simulated annealing schedule.

        Starts with uniform random (max exploration) and gradually shifts
        toward performance-based selection as generations progress.
        This helps escape local optima on difficult multimodal landscapes.
        """
        # Temperature schedule: starts high (random), decreases over time (greedy)
        # Use inverse log schedule for smooth annealing
        total_counts = np.sum(self.operator_counts)

        # Temperature: high early, low late
        temperature = max(0.01, 10.0 / np.log2(total_counts + 2))

        # Softmax-style selection with temperature
        # Transform rewards to positive space using running max + small offset
        reward_offset = np.max(self.operator_rewards) + 1.0
        positive_rewards = self.operator_rewards - reward_offset

        # Compute softmax probabilities
        exp_scores = np.exp(positive_rewards / max(temperature, 1e-10))
        softmax_probs = exp_scores / np.sum(exp_scores)

        # Ensure valid probabilities (handle numerical issues)
        softmax_probs = np.clip(softmax_probs, 1e-10, 1.0)
        softmax_probs = softmax_probs / np.sum(softmax_probs)

        # Select operator using softmax probabilities
        selected = np.random.choice(self.n_strategies, p=softmax_probs)

        return selected
```

# --- From variant_07_idea_0.py (5 wins) ---
```python
def _select_mutation_operator_batch(self):
            """Select mutation operator using Multi-Objective Thompson Sampling with entropy bonus."""
            total_counts = np.sum(self.operator_counts)

            # Compute multi-objective reward signal (3 components)
            avg_rewards = self.operator_rewards / np.maximum(self.operator_counts, 1)

            # Component 1: Success rate (exploitation)
            success_component = avg_rewards.copy()

            # Component 2: Diversity contribution (how much this operator explores different regions)
            diversity_component = np.zeros(self.n_strategies)
            if hasattr(self, 'operator_diversity') and len(self.operator_diversity) == self.n_strategies:
                diversity_component = self.operator_diversity / (np.maximum(np.sum(self.operator_diversity), 1e-10))
            else:
                # Fallback: favor less-used operators for diversity
                diversity_component = 1.0 / np.maximum(self.operator_counts, 1)

            # Component 3: Convergence momentum (recent improvement trend)
            momentum_component = np.zeros(self.n_strategies)
            if hasattr(self, 'operator_momentum') and len(self.operator_momentum) == self.n_strategies:
                momentum_component = np.clip(self.operator_momentum, 0, 1)

            # Combine components with adaptive weights
            combined_reward = (0.5 * success_component + 
                              0.3 * diversity_component + 
                              0.2 * momentum_component)

            # Prevent division by zero / NaN
            combined_reward = np.nan_to_num(combined_reward, nan=0.0, posinf=0.0, neginf=0.0)

            # Thompson Sampling: sample from Gamma distribution parameterized by rewards
            # Shape = reward + 1 (avoid zero), Scale = 1
            shape_param = combined_reward + 1e-6
            sampled_scores = np.random.gamma(shape_param, 1.0)

            # Entropy-maximizing exploration bonus: favor operators with lower selection counts
            exploration_bonus = self.ucb_c * np.sqrt(np.log(total_counts + 1) / np.maximum(self.operator_counts, 1))
            exploration_bonus = np.clip(exploration_bonus, 0, 10.0)  # Prevent overflow

            # Final score with exploration bonus
            final_scores = sampled_scores + exploration_bonus

            # Select operator with highest final score
            selected = int(np.argmax(final_scores))

            # Initialize tracking arrays if needed
            if not hasattr(self, 'operator_diversity'):
                self.operator_diversity = np.ones(self.n_strategies)
            if not hasattr(self, 'operator_momentum'):
                self.operator_momentum = np.zeros(self.n_strategies)

            # Decay momentum (keep only recent history)
            self.operator_momentum *= 0.9

            # Update diversity tracking (will be used next call)
            self.operator_diversity *= 0.95

            return selected
```

# --- From variant_08_idea_0.py (2 wins) ---
```python
def _select_mutation_operator_batch(self):
        """Entropy-driven adaptive operator selection for escaping local optima."""
        n_ops = self.n_strategies

        # Initialize tracking if first call
        if not hasattr(self, 'op_success_history') or self.op_success_history is None:
            self.op_success_history = np.zeros((n_ops, 10))
            self.op_history_idx = 0
            self.last_selected = 0

        # Record last operator's success for history tracking
        if hasattr(self, 'last_success_rate'):
            self.op_success_history[self.last_selected, self.op_history_idx] = self.last_success_rate
            self.op_history_idx = (self.op_history_idx + 1) % 10

        # Compute empirical success rates from history
        success_rates = np.mean(self.op_success_history, axis=1) + 1e-10

        # Compute Shannon entropy of success distribution (measures diversity of recent successes)
        p_normalized = success_rates / (np.sum(success_rates) + 1e-10)
        entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
        max_entropy = np.log(n_ops + 1e-10)
        normalized_entropy = entropy / (max_entropy + 1e-10)

        # Compute success rate variance (high variance = some ops working, some not)
        success_variance = np.var(success_rates)

        # Adaptive exploration factor: low entropy OR high variance → explore more
        # This handles both "stuck in local optima" and "landscape too complex" cases
        explore_factor = max(0.1, 1.0 - normalized_entropy + min(0.4, success_variance * 10))
        exploit_factor = 1.0 - explore_factor

        # Base probabilities from success history (exploitation component)
        exploit_probs = success_rates / np.sum(success_rates)

        # Uniform exploration probabilities
        explore_probs = np.ones(n_ops) / n_ops

        # Mix exploration and exploitation
        mixed_probs = explore_factor * explore_probs + exploit_factor * exploit_probs
        mixed_probs = mixed_probs / np.sum(mixed_probs)

        # Ensure numerical robustness
        mixed_probs = np.clip(mixed_probs, 1e-10, 1.0)
        mixed_probs = mixed_probs / np.sum(mixed_probs)

        # Select operator using computed probabilities
        selected = np.random.choice(n_ops, p=mixed_probs)

        # Force diversity injection every ~20 calls if entropy is low
        if not hasattr(self, 'call_counter'):
            self.call_counter = 0
        self.call_counter += 1

        if self.call_counter % 20 == 0 and normalized_entropy < 0.5:
            # Force selection of least-used operator for diversity
            least_used = np.argmin(self.op_success_history.sum(axis=1))
            if np.random.random() < 0.3:  # 30% chance to force
                selected = least_used

        self.last_selected = selected
        return selected
```

# --- From variant_09_idea_0.py (3 wins) ---
```python
def _select_mutation_operator_batch(self):
        """Select mutation operator using Thompson Sampling with Beta distributions."""
        # Track successes and failures for each operator
        if not hasattr(self, 'operator_successes'):
            self.operator_successes = np.ones(self.n_strategies, dtype=float)
        if not hasattr(self, 'operator_failures'):
            self.operator_failures = np.ones(self.n_strategies, dtype=float)

        # Sample from Beta distributions for each operator
        samples = np.array([np.random.beta(self.operator_successes[i] + 0.1, 
                                            self.operator_failures[i] + 0.1) 
                            for i in range(self.n_strategies)])

        # Add landscape-aware perturbation to bias toward exploration on hard problems
        fitness = getattr(self, 'last_fitness', None)
        if fitness is not None and len(fitness) > 1:
            f_range = np.max(fitness) - np.min(fitness)
            # High variance landscape: boost exploration by adding noise to samples
            if f_range > 1e3:
                samples += np.random.uniform(0, 0.3, size=self.n_strategies)
            elif f_range > 1e1:
                samples += np.random.uniform(0, 0.15, size=self.n_strategies)

        # Select operator with highest sampled value
        selected = np.argmax(samples)
        return selected
```

# --- From variant_10_idea_0.py (3 wins) ---
```python
def _select_mutation_operator_batch(self):
        """Select mutation operator using entropy-adaptive softmax selection."""
        # Track exponential moving average of success rates per operator
        decay = 0.9
        if not hasattr(self, 'operator_success_ema'):
            self.operator_success_ema = np.zeros(self.n_strategies)

        # Compute current success rates
        total_counts = np.sum(self.operator_counts) + 1e-10
        current_rates = self.operator_rewards / np.maximum(self.operator_counts, 1)

        # Update EMA with exponential decay (recent performance weighted higher)
        self.operator_success_ema = decay * self.operator_success_ema + (1 - decay) * current_rates

        # Compute variance and entropy of success rates for adaptive temperature
        rate_variance = np.var(self.operator_success_ema)
        rate_max = np.max(self.operator_success_ema)
        rate_min = np.min(self.operator_success_ema)

        # Entropy-based adaptive temperature:
        # Low variance -> high entropy -> high temperature (explore)
        # High variance -> low entropy -> low temperature (exploit)
        if rate_max > rate_min + 1e-10:
            normalized_rates = (self.operator_success_ema - rate_min) / (rate_max - rate_min + 1e-10)
            normalized_rates = np.clip(normalized_rates, 1e-10, 1.0)
            entropy = -np.sum(normalized_rates * np.log(normalized_rates + 1e-10))
            max_entropy = np.log(self.n_strategies + 1e-10)
            entropy_factor = max(0.1, entropy / (max_entropy + 1e-10))
        else:
            entropy_factor = 1.0

        # Temperature inversely proportional to variance but boosted by entropy
        temperature = max(0.1, 2.0 * entropy_factor / (rate_variance * 10 + 0.1))

        # Softmax probabilities
        scaled = self.operator_success_ema / (temperature + 1e-10)
        scaled = scaled - np.max(scaled)  # Numerical stability
        exp_rates = np.exp(scaled)
        probs = exp_rates / (np.sum(exp_rates) + 1e-10)

        # Ensure valid probabilities
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (np.sum(probs) + 1e-10)

        # Probabilistic selection (fundamentally different from UCB's argmax)
        selected = np.random.choice(self.n_strategies, p=probs)

        return selected
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


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Success-Based Operator Pool.
    
    Key features:
    - Pool of DE mutation strategies selected via success history (UCB-based)
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5*dim (balance between exploration and efficiency)
        self.np = min(max(5 * dim, 20), 200)
        
        # Mutation strategies pool
        self.mutation_strategies = ['rand', 'best', 'current_to_pbest', 'rand_to_best']
        self.n_strategies = len(self.mutation_strategies)
        
        # Success history for operator selection (UCB)
        self.operator_rewards = np.zeros(self.n_strategies)
        self.operator_counts = np.ones(self.n_strategies)
        self.ucb_c = 2.0  # Exploration weight for UCB
        
        # DE parameters
        self.cr = 0.9
        self.f_base = 0.5
        self.f_adapt = 0.1
        
        # Local search parameters
        self.local_search_interval = 5  # Apply LS every N generations
        self.local_search_budget = 0.02  # Fraction of func budget for LS
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE (optional, bounded)
        self.archive = None
        self.archive_max_size = self.np
        
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.np, self.dim))
        
        # LHS for better spread
        grid = np.linspace(0, 1, self.np + 1)[:-1] + np.random.uniform(0, 1/self.np, self.np)
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.lower_bound + (self.upper_bound - self.lower_bound) * grid[perm]
        
        # Evaluate initial population
        fitness = func(pop)
        fitness = self._handle_budget_truncation(fitness, pop)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            # Truncate population to match fitness
            actual_len = len(fitness)
            # Keep first `actual_len` individuals
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Select mutation operator using UCB-based operator selection."""
        # UCB formula
        ucb_values = (self.operator_rewards / np.maximum(self.operator_counts, 1) + 
                      self.ucb_c * np.sqrt(np.log(sum(self.operator_counts)) / 
                      np.maximum(self.operator_counts, 1)))
        
        # Select operator for each individual (with small random perturbation)
        selected = np.argmax(ucb_values) + np.random.randint(-1, 2)
        selected = np.clip(selected, 0, self.n_strategies - 1)
        return selected
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using multi-strategy ensemble with fitness-weighted selection."""
        np_pop = len(population)

        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)

        # Get sorted indices for various selection schemes
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
        worst_idx = sorted_idx[-1]

        # Compute fitness landscape properties for adaptive strategy selection
        f_min, f_max = np.min(fitness), np.max(fitness)
        f_range = max(f_max - f_min, 1e-10)
        f_variance = np.var(fitness)
        best_worst_ratio = (f_min + 1e-10) / (f_max + 1e-10)

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Efficient batch index sampling
        valid_mask = np.ones((np_pop, np_pop), dtype=bool)
        np.fill_diagonal(valid_mask, False)

        for i in range(np_pop):
            valid_mask[i, i] = False

        # Sample r1, r2, r3 for each individual
        r1 = np.array([np.random.choice(np.where(valid_mask[i])[0]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        r3 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i], r2[i]])) 
                       for i in range(np_pop)])

        # Adaptive F with dynamic bounds based on landscape
        base_f = self.f_base * (1.0 + 0.5 * np.log1p(f_variance) / (np_pop + 1))
        f_values = base_f + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.05, 2.5)

        # Strategy 0: Random direction (rand/3/bin - different from rand/1)
        mutant_rand = population[r1] + f_values[:, np.newaxis] * (
            population[r2] - population[r3])

        # Strategy 1: Gradient-following (current-to-pbest/2 with better indices)
        pbest = population[np.random.choice(pbest_idx, np_pop)]
        mutant_gradient = population + f_values[:, np.newaxis] * (
            pbest - population[r1] + population[r2] - population[r3])

        # Strategy 2: Opposition-based (toward better, away from worst)
        best_idx = sorted_idx[0]
        mutant_oppose = population[r1] + f_values[:, np.newaxis] * (
            population[best_idx] - population[r2] + population[r3] - population[r1])

        # Strategy 3: Dimension-wise精英 (elite-guided with dimensional diversity)
        elite = population[sorted_idx[:max(1, int(np_pop * 0.05))]]  # Top 5%
        elite_choice = elite[np.random.randint(0, len(elite), size=np_pop)]
        mutant_elite = elite_choice + f_values[:, np.newaxis] * (
            population[r1] - population[r2])

        # Compute strategy selection probabilities based on fitness landscape
        # High variance = exploration needed, low ratio = stuck in local optima
        explore_weight = min(0.5, f_variance / (f_range + 1e-10))
        exploit_weight = 1.0 - explore_weight

        # Probabilities: [rand, gradient, oppose, elite]
        if best_worst_ratio < 0.1:  # Stuck - favor exploration
            probs = np.array([0.35, 0.25, 0.25, 0.15])
        elif best_worst_ratio > 0.5:  # Good progress - balance
            probs = np.array([0.2, 0.35, 0.15, 0.3])
        else:  # Normal - moderate exploration
            probs = np.array([0.3, 0.3, 0.2, 0.2])

        # Normalize probabilities
        probs = probs / probs.sum()

        # Generate random choices for each individual
        strategy_choices = np.random.choice(4, size=np_pop, p=probs)

        # Assemble final mutants based on strategy choices
        for i in range(np_pop):
            if strategy_choices[i] == 0:
                mutants[i] = mutant_rand[i]
            elif strategy_choices[i] == 1:
                mutants[i] = mutant_gradient[i]
            elif strategy_choices[i] == 2:
                mutants[i] = mutant_oppose[i]
            else:
                mutants[i] = mutant_elite[i]

        # Clip to bounds with numerical safety
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        # Handle any NaN/Inf from numerical issues
        nan_mask = np.any(np.isnan(mutants) | np.isinf(mutants), axis=1)
        if np.any(nan_mask):
            mutants[nan_mask] = population[nan_mask] + np.random.uniform(
                -0.1, 0.1, (np.sum(nan_mask), self.dim))
            mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        np_pop = len(population)
        
        # Random crossover mask
        cr_mask = np.random.rand(np_pop, self.dim) < self.cr
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        cr_mask[np.arange(np_pop), j_rand] = True
        
        # Create trial vectors
        trials = np.where(cr_mask, mutants, population)
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Selection: greedy replacement based on fitness."""
        # Better fitness wins (assuming minimization)
        better_mask = trial_fitness < fitness
        
        # Update population and fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness, better_mask
    
    def _apply_local_refinement(self, population, fitness, func):
        """Apply Nelder-Mead-like local search to top individuals."""
        if len(population) < self.dim + 1:
            return population, fitness
        
        n_elite = min(3, len(population))  # Apply to top 3
        
        for i in range(n_elite):
            # Get current best position
            x_best = population[i].copy()
            f_best = fitness[i]
            
            # Build simplex around current best
            simplex = np.tile(x_best, (self.dim + 1, 1))
            simplex[0] = x_best
            
            # Add small perturbations for other vertices
            step_size = 0.5
            for j in range(1, self.dim + 1):
                simplex[j] = x_best.copy()
                simplex[j, (j-1) % self.dim] += step_size * (self.upper_bound - self.lower_bound)
                simplex[j] = np.clip(simplex[j], self.lower_bound, self.upper_bound)
            
            # Evaluate simplex
            simplex_fitness = func(simplex)
            simplex_fitness = self._handle_budget_truncation(simplex_fitness, simplex)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            # Simple Nelder-Mead iteration
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            # Sort simplex
            sort_idx = np.argsort(simplex_fitness)
            simplex = simplex[sort_idx]
            simplex_fitness = simplex_fitness[sort_idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
            f_r = func(x_r.reshape(1, -1))[0]
            
            if f_r < simplex_fitness[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                f_e = func(x_e.reshape(1, -1))[0]
                
                if f_e < f_r:
                    new_point, new_f = x_e, f_e
                else:
                    new_point, new_f = x_r, f_r
            elif f_r < simplex_fitness[-2]:
                new_point, new_f = x_r, f_r
            else:
                # Shrink
                new_point = simplex[0] + rho * (simplex[-1] - simplex[0])
                new_point = np.clip(new_point, self.lower_bound, self.upper_bound)
                new_f = func(new_point.reshape(1, -1))[0]
            
            # Update if improvement found
            if new_f < f_best:
                population[i] = new_point
                fitness[i] = new_f
        
        return population, fitness
    
    def _update_operator_rewards(self, selected_operator, success_mask):
        """Update operator rewards based on success rate."""
        success_rate = np.mean(success_mask)
        
        # Reward successful operators
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        # Decay rewards of non-selected operators slightly
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
    
    def _adapt_step_size(self, success_rate):
        """Adapt F and CR based on recent success."""
        if success_rate > 0.2:
            # Good success: maintain or slightly increase exploration
            self.f_adapt = min(0.5, self.f_adapt * 1.1)
        elif success_rate < 0.05:
            # Poor success: reduce step size for exploitation
            self.f_adapt = max(0.01, self.f_adapt * 0.9)
        
        # Adapt CR: increase if stuck, decrease if unstable
        if success_rate > 0.3:
            self.cr = min(0.95, self.cr * 1.01)
        elif success_rate < 0.1:
            self.cr = max(0.5, self.cr * 0.99)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Average of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        avg_distance = np.mean(distances[mask])
        
        return avg_distance
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
        self.best_history.append(current_best)
        
        if len(self.best_history) > self.stagnation_limit:
            self.best_history.pop(0)
        
        if len(self.best_history) >= self.stagnation_limit:
            improvement = self.best_history[-1] - self.best_history[0]
            if abs(improvement) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            return self.stagnation_counter >= 2
        
        return False
    
    def _restart_if_needed(self, population, fitness, func, force=False):
        """Restart population if stagnation or diversity loss detected."""
        diversity = self._compute_diversity(population)
        
        if force or diversity < self.diversity_threshold or self.stagnation_counter >= 2:
            # Reinitialize portion of population
            n_reinit = int(0.7 * self.np)
            reinit_idx = np.argsort(fitness)[-n_reinit:]  # Replace worst
            
            # Generate new individuals using LHS
            for d in range(self.dim):
                grid = np.linspace(0, 1, n_reinit + 1)[:-1] + np.random.uniform(0, 1/n_reinit, n_reinit)
                perm = np.random.permutation(n_reinit)
                population[reinit_idx, d] = (self.lower_bound + 
                    (self.upper_bound - self.lower_bound) * grid[perm])
            
            # Evaluate reinitialized portion
            new_fitness = func(population[reinit_idx])
            if len(new_fitness) < len(reinit_idx):
                new_fitness = np.pad(new_fitness, (0, len(reinit_idx) - len(new_fitness)), 
                                     constant_values=np.inf)
            fitness[reinit_idx] = new_fitness
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            self.best_history = []
            
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Ensure all individuals are within bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        population, fitness = self._initialize_population(func)
        
        # Check budget after initialization
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return fitness[best_idx], population[best_idx]
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator
            selected_operator = self._select_mutation_operator_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip trials to bounds
            trials = self._clip_to_bounds(trials)
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget truncation
            if len(trial_fitness) < len(trials):
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                population = population[:valid_trials]
                fitness = fitness[:valid_trials]
                self.np = valid_trials
                if self.np < 10:
                    break
            
            # Check stopping after partial evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update operator rewards
            self._update_operator_rewards(selected_operator, success_mask)
            
            # Adapt step size
            success_rate = np.mean(success_mask)
            self._adapt_step_size(success_rate)
            
            # Periodic local search
            local_search_counter += 1
            if local_search_counter >= self.local_search_interval:
                population, fitness = self._apply_local_refinement(population, fitness, func)
                local_search_counter = 0
                
                if stopping_condition():
                    break
            
            # Check stagnation
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            # Periodic restart check based on diversity
            if generation % 10 == 0:
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            generation += 1
        
        # Return best solution
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]

```