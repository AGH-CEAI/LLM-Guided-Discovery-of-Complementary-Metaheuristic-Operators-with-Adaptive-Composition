Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_strategy_probabilities` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
1      3.317715e+00            2.489846e+00            2.109694e+00            1.618663e+00            1.437770e+00            2.097136e+00            3.098795e+00            1.866599e+00            1.243509e+00            3.196425e+00            5.802104e-01            
2      1.000000e-08            1.995747e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            
3      1.829475e+00            2.094494e+00            1.937319e+00            2.488433e+00            1.785378e+00            2.122039e+00            1.182651e+00            2.343666e+00            1.929015e+00            2.027052e+00            2.174856e+00            
4      8.728016e-02            4.651125e-02            1.618159e-01            3.671615e-02            4.626268e-02            4.374372e-01            1.760552e-01            1.216833e-01            9.610941e-02            8.113583e-02            2.014724e-01            
5      3.740402e+01            8.205490e+01            6.337954e+00            6.947856e+01            2.182857e+01            4.381352e+01            8.446085e+00            4.453270e+01            5.515893e+01            1.053556e+02            1.129296e+02            
6      1.655228e+02            1.549076e+02            1.623302e+02            7.431076e+01            1.468269e+02            3.430454e+02            2.739773e+02            1.972558e+02            1.799500e+02            2.158081e+02            2.161151e+02            
7      3.674261e+00            2.865401e+00            4.114451e+00            3.502552e+00            4.034603e+00            3.183045e+00            3.519141e+00            3.058729e+00            2.871828e+00            5.072810e+00            2.915469e+00            
8      2.048679e+00            1.621083e+00            2.164545e+00            2.278284e+00            1.541854e+00            1.906188e+00            1.875422e+00            1.946554e+00            1.732962e+00            1.622090e+00            1.334519e+00            
9      4.062595e+01            3.044608e+01            4.821276e+01            3.029732e+01            5.620261e+01            7.786857e+01            4.912956e+01            4.837383e+01            3.979485e+01            4.912953e+01            3.805847e+01            
10     3.917279e-04            1.088181e+00            1.420445e-08            1.936182e-07            1.198088e-08            2.085884e-04            1.000000e-08            9.083613e-08            2.513994e-08            3.325941e-05            7.988428e-08            
11     1.030042e+02            1.005859e+02            3.451391e+01            5.794113e+01            5.575740e+01            2.863065e+01            2.239964e+01            2.051305e+01            3.696755e+01            4.368674e+01            3.282261e+01            
12     7.319175e-05            3.237154e+00            3.298543e-04            1.137280e-04            4.043202e-03            1.466677e-01            8.918608e-04            5.011295e-05            8.808779e-05            4.283206e-04            1.500146e-03            
13     8.829929e+00            2.079703e+01            7.717986e+00            1.650697e+01            3.213980e+00            2.618399e+00            2.852412e+00            1.025941e+01            7.070356e+00            1.778391e+01            3.114594e+00            
14     8.033541e+00            6.289835e+00            9.368232e+00            6.920154e+00            5.602335e+00            1.250385e+01            9.204267e+00            9.004168e+00            8.652530e+00            8.140892e+00            9.434181e+00            
15     2.549627e+00            2.799368e+00            2.073949e+00            2.591828e+00            2.722765e+00            2.729414e+00            2.166828e+00            2.294642e+00            2.308431e+00            2.547554e+00            2.162201e+00            
16     8.035273e+02            1.058355e+03            8.518908e+02            5.930114e+02            1.293954e+03            1.728587e+03            7.737291e+02            8.109583e+02            8.769455e+02            9.254470e+02            9.454569e+02            
17     1.258049e+04            1.876192e+04            5.887835e+02            4.096815e+03            5.491262e+02            8.081252e+01            1.305134e+02            1.002662e+03            3.399786e+03            3.680195e+03            3.713915e+02            
18     2.798975e+01            3.597475e+01            2.135203e+01            3.093942e+01            2.554488e+01            2.003370e+01            2.073973e+01            2.739186e+01            2.291453e+01            2.031719e+01            2.055628e+01            
19     1.214525e+02            5.893403e+01            1.324053e+02            7.686048e+01            1.077665e+02            2.248859e+02            1.407519e+02            1.246835e+02            1.280243e+02            1.006638e+02            1.521079e+02            
20     1.621606e+01            1.895225e+01            1.611830e+01            1.459376e+01            1.907216e+01            2.326755e+01            1.610765e+01            1.739799e+01            1.641695e+01            1.791163e+01            1.718427e+01            
21     4.962963e+00            4.737829e+00            5.152090e+00            4.840788e+00            5.209016e+00            5.411576e+00            5.111867e+00            5.072682e+00            5.066332e+00            4.974880e+00            5.165057e+00            
22     9.457390e+00            7.362297e+00            1.121752e+01            7.250628e+00            1.160188e+01            1.311545e+01            1.050792e+01            1.118279e+01            9.642564e+00            1.061813e+01            1.061783e+01            
23     5.676140e+01            4.951089e+01            6.279919e+01            5.203669e+01            4.864329e+01            8.349620e+01            5.927808e+01            5.458656e+01            6.059206e+01            5.206509e+01            6.072772e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: variant_10_idea_0.py  (error=5.802104e-01)
Task  2: SKIPPED (trivial)
Task  3: variant_06_idea_0.py  (error=1.182651e+00)
Task  4: variant_03_idea_0.py  (error=3.671615e-02)
Task  5: variant_02_idea_0.py  (error=6.337954e+00)
Task  6: variant_03_idea_0.py  (error=7.431076e+01)
Task  7: variant_01_idea_0.py  (error=2.865401e+00)
Task  8: variant_10_idea_0.py  (error=1.334519e+00)
Task  9: variant_03_idea_0.py  (error=3.029732e+01)
Task 10: SKIPPED (trivial)
Task 11: variant_07_idea_0.py  (error=2.051305e+01)
Task 12: variant_07_idea_0.py  (error=5.011295e-05)
Task 13: variant_05_idea_0.py  (error=2.618399e+00)
Task 14: variant_04_idea_0.py  (error=5.602335e+00)
Task 15: variant_02_idea_0.py  (error=2.073949e+00)
Task 16: variant_03_idea_0.py  (error=5.930114e+02)
Task 17: variant_05_idea_0.py  (error=8.081252e+01)
Task 18: variant_05_idea_0.py  (error=2.003370e+01)
Task 19: variant_01_idea_0.py  (error=5.893403e+01)
Task 20: variant_03_idea_0.py  (error=1.459376e+01)
Task 21: variant_01_idea_0.py  (error=4.737829e+00)
Task 22: variant_03_idea_0.py  (error=7.250628e+00)
Task 23: variant_04_idea_0.py  (error=4.864329e+01)

WIN COUNTS:
  variant_03_idea_0.py: 6 wins
  variant_01_idea_0.py: 3 wins
  variant_05_idea_0.py: 3 wins
  variant_10_idea_0.py: 2 wins
  variant_02_idea_0.py: 2 wins
  variant_07_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (3 wins) ---
```python
def _adapt_strategy_probabilities(self):
        # Thompson sampling for adaptive operator selection
        # For each strategy, model success probability with Beta distribution
        # using observed successes and failures.
        for i in range(self.NP):
            samples = np.empty(self.n_strategies)
            for s in range(self.n_strategies):
                successes = self.strategy_success_count[s]
                attempts = self.strategy_attempt_count[s]
                # Posterior parameters: alpha = successes + 1 (prior 1)
                # beta = failures + 1 (prior 1)
                failures = max(attempts - successes, 0.0)
                alpha = successes + 1.0
                beta = failures + 1.0
                # Sample from Beta distribution
                samples[s] = self.rng.beta(alpha, beta)
            # Choose strategy with highest sampled value
            self.strategy_idx[i] = np.argmax(samples)
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _adapt_strategy_probabilities(self):
        # Compute magnitude-weighted scores (not just counts)
        # Weight recent improvements by their absolute contribution
        total_attempts = np.sum(self.strategy_attempt_count) + 1e-10
        success_rate = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)

        # Base score: combination of success rate and recency-weighted fitness contribution
        # Use log-scaled fitness contribution to prevent dominance by outliers
        log_fitness = np.log1p(np.maximum(self.strategy_success_fitness, 0))
        base_score = 0.7 * success_rate + 0.3 * (log_fitness / (np.max(log_fitness) + 1e-10))

        # Entropy bonus: boost less-used strategies when best strategy is underperforming
        # This prevents premature convergence to a single strategy
        uniform_prob = 1.0 / self.n_strategies
        current_probs = base_score / (np.sum(base_score) + 1e-10)
        entropy_bonus = uniform_prob - current_probs
        entropy_bonus = np.clip(entropy_bonus, 0, 0.3)

        # Compute convergence indicator: if best strategy dominates, reduce exploration pressure
        max_score = np.max(current_probs)
        convergence_factor = 1.0 - (max_score - uniform_prob) / (1.0 - uniform_prob + 1e-10)
        convergence_factor = np.clip(convergence_factor, 0.1, 1.0)

        # Final probabilities with entropy balancing
        probs = current_probs + entropy_bonus * convergence_factor
        probs = probs / (np.sum(probs) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        probs = probs / (np.sum(probs) + 1e-10)

        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```

# --- From variant_03_idea_0.py (6 wins) ---
```python
def _adapt_strategy_probabilities(self):
            # Compute success rate per strategy
            rates = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)

            # Blend rate (reliability) with count (impact), weighted by generation
            # Early: favor reliability, Late: favor impact
            gen = len(self.diversity_history) + 1
            alpha = min(0.9, 0.3 + 0.005 * gen)  # 0.30 -> 0.90 over generations

            # Exponential weighting to amplify differences; log1p for numerical stability
            weighted_scores = alpha * self.strategy_success_count + (1 - alpha) * np.log1p(self.strategy_success_count)

            # Blend with success rate for balanced selection
            scores = alpha * rates + (1 - alpha) * weighted_scores

            # Ensure non-negative and numerically robust
            scores = np.maximum(scores, 1e-15)

            # Normalize to probabilities
            total = np.sum(scores)
            probs = scores / (total + 1e-10)

            for i in range(self.NP):
                self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _adapt_strategy_probabilities(self):
            # Compute performance score combining success count and improvement magnitude
            scores = np.zeros(self.n_strategies)
            for s in range(self.n_strategies):
                count = self.strategy_success_count[s]
                fitness_improvement = self.strategy_success_fitness[s]
                scores[s] = np.log1p(count) * (1.0 + np.log1p(fitness_improvement) + 1e-10)

            # Rank-based weighting: higher rank = higher weight
            ranks = np.argsort(np.argsort(scores)) + 1
            weighted_scores = scores * ranks

            # Adaptive temperature: lower T when variance is high (force exploitation)
            score_variance = np.var(weighted_scores) + 1e-10
            T = 0.5 / (1.0 + np.log1p(score_variance))

            # Softmax with temperature
            scores_shifted = weighted_scores - np.max(weighted_scores)
            exp_scores = np.exp(scores_shifted / (T + 1e-10))
            probs = exp_scores / (np.sum(exp_scores) + 1e-10)

            # Fallback for edge cases
            if np.any(np.isnan(probs)) or np.sum(probs) < 1e-10:
                probs = np.ones(self.n_strategies) / self.n_strategies
            probs = np.clip(probs, 1e-6, 1.0)
            probs = probs / np.sum(probs)

            for i in range(self.NP):
                self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```

# --- From variant_05_idea_0.py (3 wins) ---
```python
def _adapt_strategy_probabilities(self):
            # Compute rank-based probabilities (robust to outliers, helps near-optimum tasks)
            if np.any(self.strategy_success_count > 0):
                success_rate = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
                ranks = np.argsort(np.argsort(-success_rate)) + 1
                rank_probs = ranks / (np.sum(ranks) + 1e-10)
            else:
                rank_probs = np.ones(self.n_strategies) / self.n_strategies

            # Entropy-based exploration temperature (adapts to stagnation)
            stagnation_factor = min(1.0, self.generation_without_improvement / 20.0)
            temperature = 0.5 + 1.5 * stagnation_factor
            entropy_weight = 0.2 + 0.4 * stagnation_factor

            # Diversity-sensitive perturbation (higher diversity = more exploration)
            if len(self.diversity_history) >= 10:
                recent_div = np.mean(self.diversity_history[-10:])
                div_range = max(np.max(self.diversity_history[-50:]) - np.min(self.diversity_history[-50:]), 1e-10)
                diversity_signal = np.clip((recent_div - np.min(self.diversity_history[-50:])) / div_range, 0.0, 1.0)
            else:
                diversity_signal = 0.5

            entropy_weight = np.clip(entropy_weight * (1.0 - 0.5 * diversity_signal), 0.1, 0.6)

            # Softmax with temperature for controlled randomness
            logits = np.log(rank_probs + 1e-10) / temperature
            logits = logits - np.max(logits)
            exp_logits = np.exp(logits)
            exploration_probs = exp_logits / (np.sum(exp_logits) + 1e-10)

            # Combine exploitation (rank-based) with exploration (entropy-driven)
            final_probs = (1.0 - entropy_weight) * rank_probs + entropy_weight * exploration_probs

            # Numerical safety
            final_probs = np.clip(final_probs, 1e-10, 1.0)
            final_probs = final_probs / (np.sum(final_probs) + 1e-10)

            # Assign strategies with minimal disruption near optimum
            for i in range(self.NP):
                self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=final_probs)
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _adapt_strategy_probabilities(self):
        # Compute success rates per strategy (avoid division by zero)
        success_rates = np.zeros(self.n_strategies)
        valid_strategies = self.strategy_attempt_count > 0
        success_rates[valid_strategies] = (
            self.strategy_success_count[valid_strategies] / 
            self.strategy_attempt_count[valid_strategies]
        )

        # Get current best fitness
        best_fitness = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.min(self.archive) if self.archive else 0.0

        # Threshold for "already near-optimal" - conservative mode
        # Use a relative threshold: 1.1x the best fitness seen so far
        ref_fitness = getattr(self, 'reference_fitness', 1e-3)
        threshold = ref_fitness * 1.5 if ref_fitness > 0 else 1e-4

        # Dual-mode: conservative if already good, adaptive otherwise
        if best_fitness < threshold:
            # Conservative mode: bias toward original uniform distribution
            # This prevents harmful drift on already-converged tasks like #12
            orig_prob = 1.0 / self.n_strategies
            adaptive_prob = success_rates / (np.sum(success_rates) + 1e-10)
            alpha = 0.85  # Strong weight on original distribution
            probs = alpha * orig_prob + (1 - alpha) * adaptive_prob
        else:
            # Adaptive mode: use softmax with temperature on success rates
            # Low temperature = more exploitative, high = more exploratory
            temperature = 0.3
            exp_rates = np.exp(success_rates / (temperature + 1e-10))
            probs = exp_rates / (np.sum(exp_rates) + 1e-10)

        # Numerical safety
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (np.sum(probs) + 1e-10)

        # Assign strategies
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _adapt_strategy_probabilities(self):
        # Compute success rates with minimum floor
        attempts = self.strategy_attempt_count + 1
        success_rate = self.strategy_success_fitness / (attempts + 1e-10)

        # Normalize success scores
        total = np.sum(self.strategy_success_fitness) + 1e-10
        normalized = self.strategy_success_fitness / total

        # Entropy-based temperature: high when uniform (explore), low when peaked (exploit)
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        max_entropy = np.log(self.n_strategies + 1e-10)
        normalized_entropy = entropy / (max_entropy + 1e-10)

        # Temperature: 0.1 (conservative) to 2.0 (exploratory)
        temperature = 0.1 + 1.9 * normalized_entropy
        temperature = np.clip(temperature, 0.05, 2.0)

        # Softmax with temperature
        weights = self.strategy_success_fitness + 1e-10
        logits = np.log(weights) / temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / np.sum(probs)

        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _adapt_strategy_probabilities(self):
        # Compute success rates per strategy
        rates = self.strategy_success_fitness / (self.strategy_attempt_count + 1e-10)
        rates = np.clip(rates, 1e-10, None)

        # Softmax with temperature: high temp = more exploration
        temperature = 0.5 + 0.5 * (1.0 - np.max(rates) / (np.mean(rates) + 1e-10))
        temperature = np.clip(temperature, 0.1, 5.0)
        exp_rates = np.exp(rates / temperature)
        base_probs = exp_rates / np.sum(exp_rates)

        # Entropy floor: if entropy too low, boost rare strategies
        entropy = -np.sum(base_probs * np.log(base_probs + 1e-10))
        max_entropy = np.log(self.n_strategies)
        entropy_ratio = entropy / (max_entropy + 1e-10)

        if entropy_ratio < 0.5:
            # Boost underrepresented strategies by log-inverse probability
            boost = np.log(1.0 / (base_probs + 1e-10))
            boost = boost / np.sum(boost)
            blend = 0.3
            probs = (1.0 - blend) * base_probs + blend * boost
        else:
            probs = base_probs

        # Ensure valid probability distribution
        probs = np.clip(probs, 1e-10, None)
        probs = probs / np.sum(probs)

        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
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


class ReflectiveDEWithStrategyPortfolio:
    """
    Differential Evolution using reflective boundary handling and an adaptive
    portfolio of mutation strategies. Each individual maintains its own
    strategy label and per-strategy success memories that drive strategy
    selection and parameter adaptation.
    """

    def __init__(self, dim, np_factor=6, seed=None):
        self.dim = dim
        self.NP = np_factor * dim
        self.NP = max(20, min(self.NP, 300))
        self.lower = -100.0
        self.upper = 100.0
        self.rng = np.random.default_rng(seed)

        # Strategy definitions (name, n_differences, uses_best)
        self.strategy_names = [
            "rand_1",      # classical rand/1
            "best_1",      # classical best/1
            "current_1",   # current-to-best/1 variant
            "rand_2",      # rand/2 for diversity
            "best_2",      # best/2 for convergence
        ]
        self.n_strategies = len(self.strategy_names)

        # Per-individual state
        self.F = np.empty(self.NP)
        self.CR = np.empty(self.NP)
        self.strategy_idx = np.zeros(self.NP, dtype=int)

        # Per-strategy success memory for adaptation
        self.strategy_success_fitness = np.zeros(self.n_strategies)
        self.strategy_success_count = np.zeros(self.n_strategies)
        self.strategy_attempt_count = np.zeros(self.n_strategies)

        # Global memory for F/CR adaptation (jADE-style)
        self.archive = []
        self.archive_max_size = self.NP

        # Stagnation detection
        self.generation_without_improvement = 0
        self.last_best_fitness = np.inf

        # Diversity tracking
        self.diversity_history = []

    def _initialize_population(self):
        lo = self.lower
        hi = self.upper
        pop = lo + (hi - lo) * self.rng.uniform(size=(self.NP, self.dim))
        self.F[:] = 0.5
        self.CR[:] = 0.5
        self.strategy_idx[:] = self.rng.integers(0, self.n_strategies, size=self.NP)
        self.strategy_success_fitness[:] = 0.0
        self.strategy_success_count[:] = 1
        self.strategy_attempt_count[:] = 1
        self.archive = []
        return pop

    def _evaluate_batch(self, population, func):
        fitness = func(population)
        if len(fitness) < len(population):
            return fitness[:len(fitness)]
        return fitness

    def _clip_reflective(self, population):
        pop = population.copy()
        mask_lo = pop < self.lower
        mask_hi = pop > self.upper
        pop[mask_lo] = 2.0 * self.lower - pop[mask_lo]
        pop[mask_hi] = 2.0 * self.upper - pop[mask_hi]
        pop = np.clip(pop, self.lower, self.upper)
        return pop

    def _select_three_distinct(self, exclude_idx):
        pool = np.arange(self.NP)
        pool = pool[pool != exclude_idx]
        selected = self.rng.choice(pool, size=3, replace=False)
        return selected[0], selected[1], selected[2]

    def _select_five_distinct(self, exclude_idx):
        pool = np.arange(self.NP)
        pool = pool[pool != exclude_idx]
        selected = self.rng.choice(pool, size=5, replace=False)
        return selected[0], selected[1], selected[2], selected[3], selected[4]

    def _mutate_single_reflective(self, idx, population, fitness, trial_pop, trial_fit):
        x_i = population[idx]
        s_idx = self.strategy_idx[idx]
        strategy = self.strategy_names[s_idx]

        a, b, c = self._select_three_distinct(idx)
        x_a, x_b, x_c = population[a], population[b], population[c]
        f_scale = self.F[idx]

        if strategy == "rand_1":
            mutant = x_a + f_scale * (x_b - x_c)
        elif strategy == "best_1":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            mutant = x_best + f_scale * (x_a - x_b)
        elif strategy == "current_1":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            mutant = x_i + f_scale * (x_best - x_i) + f_scale * (x_a - x_b)
        elif strategy == "rand_2":
            d, e = self._select_three_distinct(idx)[:2]
            x_d, x_e = population[d], population[e]
            mutant = x_a + f_scale * (x_b - x_c) + f_scale * (x_d - x_e)
        elif strategy == "best_2":
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            d, e = self._select_three_distinct(idx)[:2]
            x_d, x_e = population[d], population[e]
            mutant = x_best + f_scale * (x_a - x_b) + f_scale * (x_d - x_e)
        else:
            mutant = x_a + f_scale * (x_b - x_c)

        return self._clip_reflective(mutant.reshape(1, -1))[0]

    def _mutate_batch(self, population, fitness):
        n_trials = self.NP
        trial_population = np.empty((n_trials, self.dim))

        for idx in range(n_trials):
            trial_population[idx] = self._mutate_single_reflective(
                idx, population, fitness, trial_population, None
            )
        return trial_population

    def _crossover_batch(self, population, trial_population):
        cr_batch = self.CR.copy()
        n_trials = self.NP
        cross_mask = self.rng.uniform(size=(n_trials, self.dim)) < cr_batch[:, np.newaxis]
        j_rand = self.rng.integers(0, self.dim, size=n_trials)
        for i in range(n_trials):
            cross_mask[i, j_rand[i]] = True
        offspring = np.where(cross_mask, trial_population, population)
        return offspring

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved = trial_fitness < fitness[:len(trial_fitness)]
        new_population = population.copy()
        new_fitness = fitness.copy()
        new_population[:len(trials)][improved] = trials[improved]
        new_fitness[:len(trials)][improved] = trial_fitness[improved]
        return new_population, new_fitness, improved

    def _update_strategy_success(self, improved_mask):
        for s in range(self.n_strategies):
            count_s = np.sum((self.strategy_idx == s) & improved_mask)
            self.strategy_success_count[s] += count_s
            self.strategy_attempt_count[s] += np.sum(self.strategy_idx == s)
            if count_s > 0:
                self.strategy_success_fitness[s] = (
                    0.5 * self.strategy_success_fitness[s] + 0.5 * count_s
                )

    def _adapt_strategy_probabilities(self):
        total_success = np.sum(self.strategy_success_fitness) + 1e-10
        probs = self.strategy_success_fitness / total_success
        probs = probs / (np.sum(probs) + 1e-10)
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)

    def _adapt_F_CR_jade_style(self, population, fitness, trials, trial_fitness, improved_mask):
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        improved_trials = trials[improved_mask]
        improved_fit = trial_fitness[improved_mask]
        improved_parents = population[:len(trials)][improved_mask]

        if len(improved_trials) > 0:
            fi_values = np.abs(improved_trials - improved_parents)
            fi_values = fi_values / (np.linalg.norm(fi_values, axis=1, keepdims=True) + 1e-10)
            fi_means = np.mean(fi_values, axis=1)
            self.archive.extend(improved_trials.tolist())
            if len(self.archive) > self.archive_max_size:
                remove_n = len(self.archive) - self.archive_max_size
                self.archive = self.archive[remove_n:]

        for i in range(self.NP):
            if improved_mask[i]:
                self.F[i] = self.rng.uniform(0.1, 0.9)
            else:
                self.F[i] = 0.9 * self.F[i] + 0.1 * 0.5

            if improved_mask[i]:
                self.CR[i] = self.rng.beta(1.0, 0.5)
            else:
                self.CR[i] = 0.9 * self.CR[i] + 0.1 * 0.5
            self.CR[i] = np.clip(self.CR[i], 0.0, 1.0)

    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)

    def _check_stagnation(self, fitness):
        current_best = np.min(fitness)
        if current_best < self.last_best_fitness - 1e-10:
            self.generation_without_improvement = 0
            self.last_best_fitness = current_best
            return False
        else:
            self.generation_without_improvement += 1
            return self.generation_without_improvement > 50

    def _reinitialize_stagnant_subpopulation(self, population, fitness):
        n_replace = max(5, self.NP // 10)
        replace_idx = self.rng.choice(self.NP, size=n_replace, replace=False)
        lo, hi = self.lower, self.upper
        new_individuals = lo + (hi - lo) * self.rng.uniform(size=(n_replace, self.dim))
        population[replace_idx] = new_individuals
        fitness[replace_idx] = np.inf
        self.generation_without_improvement = 0
        return population, fitness

    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        population = self._clip_reflective(population)
        fitness = self._evaluate_batch(population, func)

        if len(fitness) == 0:
            return np.inf, np.zeros(self.dim)

        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()

        while not stopping_condition():
            trials = self._mutate_batch(population, fitness)
            trials = self._crossover_batch(population, trials)
            trials = self._clip_reflective(trials)
            trial_fitness = self._evaluate_batch(trials, func)

            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]

            if len(trial_fitness) == 0:
                break

            if stopping_condition():
                break

            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()

            self._update_strategy_success(improved_mask)
            self._adapt_strategy_probabilities()
            self._adapt_F_CR_jade_style(population, fitness, trials, trial_fitness, improved_mask)

            div = self._compute_diversity(population)
            self.diversity_history.append(div)

            if self._check_stagnation(fitness):
                population, fitness = self._reinitialize_stagnant_subpopulation(population, fitness)
                self.archive = []
                self.F[:] = 0.5
                self.CR[:] = 0.5

        return f_opt, x_opt

```