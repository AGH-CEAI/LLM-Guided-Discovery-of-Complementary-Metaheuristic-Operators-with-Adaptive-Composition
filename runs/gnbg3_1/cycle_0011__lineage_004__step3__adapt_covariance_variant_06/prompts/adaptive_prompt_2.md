Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance_variant_06` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.608249e-02            5.373612e-02            5.596131e-02            5.638180e-02            -inf                    -inf                    5.556210e-02            -inf                    5.395342e-02            -inf                    
1      4.240407e-02            3.650132e-02            4.258479e-02            4.046055e-02            -inf                    -inf                    4.132323e-02            -inf                    4.228649e-02            -inf                    
2      1.440424e-01            1.465591e-01            1.422670e-01            1.426805e-01            -inf                    -inf                    1.422111e-01            -inf                    1.625373e-01            -inf                    
3      1.307111e-01            1.248477e-01            1.305463e-01            1.277726e-01            -inf                    -inf                    1.297625e-01            -inf                    1.375767e-01            -inf                    
4      5.162655e-01            5.098791e-01            5.033149e-01            5.127900e-01            -inf                    -inf                    5.052129e-01            -inf                    5.113185e-01            -inf                    
5      8.840722e-05            6.587743e-05            8.222860e-05            6.998583e-05            -inf                    -inf                    7.525365e-05            -inf                    8.052193e-05            -inf                    
6      1.564329e-02            1.466274e-02            1.530505e-02            1.544072e-02            -inf                    -inf                    1.506105e-02            -inf                    1.516906e-02            -inf                    
7      7.442408e-02            7.372171e-02            7.397966e-02            7.550048e-02            -inf                    -inf                    7.509927e-02            -inf                    8.118282e-02            -inf                    
8      2.460600e-01            2.486806e-01            2.460845e-01            2.495303e-01            -inf                    -inf                    2.438198e-01            -inf                    2.516238e-01            -inf                    
9      7.226689e-02            7.640464e-02            9.553907e-02            8.658219e-02            -inf                    -inf                    7.390407e-02            -inf                    4.417092e-01            -inf                    
10     1.348303e+01            1.994294e+01            3.332764e+01            1.551298e+01            -inf                    -inf                    3.055403e+01            -inf                    2.645070e+01            -inf                    
11     4.292417e+01            3.422881e+01            7.275331e+01            3.260426e+01            -inf                    -inf                    7.133277e+01            -inf                    7.157613e+01            -inf                    
12     1.745455e+01            2.246692e+01            3.318824e+01            1.984864e+01            -inf                    -inf                    3.029061e+01            -inf                    1.747088e+01            -inf                    
13     1.698113e+00            3.301368e+00            6.373779e+00            2.135646e+00            -inf                    -inf                    4.797185e+00            -inf                    4.738119e+00            -inf                    
14     2.691049e+00            2.798141e+00            2.779764e+00            2.768971e+00            -inf                    -inf                    2.772119e+00            -inf                    2.991474e+00            -inf                    
15     2.841946e+00            2.852101e+00            3.097854e+00            2.818895e+00            -inf                    -inf                    3.008651e+00            -inf                    2.980938e+00            -inf                    
16     6.333563e+01            6.271139e+01            6.715688e+01            6.564791e+01            -inf                    -inf                    6.403032e+01            -inf                    6.482735e+01            -inf                    
17     3.462647e+01            4.852462e+01            6.928606e+02            6.876996e+01            -inf                    -inf                    2.041313e+02            -inf                    3.503019e+02            -inf                    
18     1.927086e+01            2.020154e+01            2.351393e+01            1.514806e+01            -inf                    -inf                    2.802133e+01            -inf                    2.115687e+01            -inf                    
19     1.772518e+01            2.019569e+01            2.320434e+01            1.937095e+01            -inf                    -inf                    1.655870e+01            -inf                    2.049136e+01            -inf                    
20     2.018037e+01            1.846645e+01            1.862853e+01            1.936402e+01            -inf                    -inf                    1.906749e+01            -inf                    1.850582e+01            -inf                    
21     4.555958e+00            4.536509e+00            4.517372e+00            4.613481e+00            -inf                    -inf                    4.539706e+00            -inf                    4.582679e+00            -inf                    
22     1.080803e+01            1.121447e+01            9.265990e+00            9.694276e+00            -inf                    -inf                    9.405621e+00            -inf                    9.852089e+00            -inf                    
23     2.018838e+01            2.142268e+01            1.908567e+01            1.843119e+01            -inf                    -inf                    2.261242e+01            -inf                    2.090202e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_idea_0.py  (error=5.373612e-02)
Task  1: variant_01_idea_0.py  (error=3.650132e-02)
Task  2: variant_07_idea_0.py  (error=1.422111e-01)
Task  3: variant_01_idea_0.py  (error=1.248477e-01)
Task  4: variant_02_idea_0.py  (error=5.033149e-01)
Task  5: variant_01_idea_0.py  (error=6.587743e-05)
Task  6: variant_01_idea_0.py  (error=1.466274e-02)
Task  7: variant_01_idea_0.py  (error=7.372171e-02)
Task  8: variant_07_idea_0.py  (error=2.438198e-01)
Task  9: original.py  (error=7.226689e-02)
Task 10: original.py  (error=1.348303e+01)
Task 11: variant_03_idea_0.py  (error=3.260426e+01)
Task 12: original.py  (error=1.745455e+01)
Task 13: original.py  (error=1.698113e+00)
Task 14: original.py  (error=2.691049e+00)
Task 15: variant_03_idea_0.py  (error=2.818895e+00)
Task 16: variant_01_idea_0.py  (error=6.271139e+01)
Task 17: original.py  (error=3.462647e+01)
Task 18: variant_03_idea_0.py  (error=1.514806e+01)
Task 19: variant_07_idea_0.py  (error=1.655870e+01)
Task 20: variant_01_idea_0.py  (error=1.846645e+01)
Task 21: variant_02_idea_0.py  (error=4.517372e+00)
Task 22: variant_02_idea_0.py  (error=9.265990e+00)
Task 23: variant_03_idea_0.py  (error=1.843119e+01)

WIN COUNTS:
  variant_01_idea_0.py: 8 wins
  original.py: 6 wins
  variant_03_idea_0.py: 4 wins
  variant_07_idea_0.py: 3 wins
  variant_02_idea_0.py: 3 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (8 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Active CMA-ES with negative weights for improved exploration."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Evolution path update
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with positive weights (standard)
        rank_mu_pos = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu_pos += self.weights[i] * np.outer(diff, diff)

        # Active CMA-ES: Rank-mu update with negative weights (worst individuals)
        # Use bottom mu individuals with negative weights
        neg_mu = max(1, self.mu // 2)
        neg_weights = np.zeros(neg_mu)
        sum_neg = 0.0
        for i in range(neg_mu):
            neg_weights[i] = -(np.log((self.NP + i) / 2.0 + 0.5) - np.log(self.NP / 2.0 + 1.0))
            sum_neg += neg_weights[i]

        if sum_neg > 1e-10:
            neg_weights /= sum_neg

        rank_mu_neg = np.zeros((self.dim, self.dim))
        for i in range(neg_mu):
            idx = self.NP - 1 - i
            diff = (self.population[idx] - self.old_mean) / self.sigma
            rank_mu_neg += neg_weights[i] * np.outer(diff, diff)

        # Adaptive learning rates based on condition number
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)

        # Reduce adaptation if already well-conditioned
        if cond < 1e3:
            ccov_scale = 1.0
        elif cond < 1e5:
            ccov_scale = 0.5
        else:
            ccov_scale = 0.25

        ccov_1 = ccov_scale / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        ccov_mu_neg = 0.5 * ccov_scale / (self.dim + 2.0) / (self.dim + 4.0)

        # Combine all updates
        self.C = ((1.0 - ccov_1 - ccov_mu - ccov_mu_neg) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu_pos +
                  ccov_mu_neg * rank_mu_neg)

        # Ensure positive definiteness
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_02_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Multi-direction ensemble covariance: blend diverse search directions by predicted quality."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute candidate updates from top-k diverse population members
        k_directions = min(self.mu, max(3, self.dim // 2))
        candidate_dirs = []
        candidate_scores = []

        for i in range(k_directions):
            diff = (self.population[i] - self.old_mean) / self.sigma
            dir_norm = np.linalg.norm(diff)
            if dir_norm > 1e-10:
                # Score = alignment with fitness improvement potential
                # Higher score = more promising direction
                pop_dist = np.linalg.norm(self.population[i] - self.mean)
                fitness_rank = (self.mu - i) / max(self.mu, 1)
                score = fitness_rank * np.log1p(pop_dist / max(dir_norm, 1e-10))
                candidate_dirs.append(diff)
                candidate_scores.append(score)

        if len(candidate_dirs) >= 2:
            # Normalize scores to weights
            max_score = max(abs(s) for s in candidate_scores)
            if max_score > 1e-10:
                weights_dir = [s / max_score for s in candidate_scores]
            else:
                weights_dir = [1.0 / len(candidate_dirs)] * len(candidate_dirs)

            # Blend weighted ensemble direction for pc
            ensemble_pc = np.zeros(self.dim)
            for d, w in zip(candidate_dirs, weights_dir):
                ensemble_pc += w * d
            ensemble_pc /= sum(weights_dir)

            # Blend rank-one matrices from each candidate direction
            ensemble_rank_one = np.zeros((self.dim, self.dim))
            for d, w in zip(candidate_dirs, weights_dir):
                d_normalized = d / max(np.linalg.norm(d), 1e-10)
                ensemble_rank_one += w * np.outer(d_normalized, d_normalized)
            ensemble_rank_one /= sum(weights_dir)
        else:
            # Fallback to standard pc update
            self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
            ensemble_rank_one = np.outer(self.pc, self.pc)

        # Compute rank-mu from diverse population sampling
        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        # Adaptive learning rates based on condition and diversity
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        # Detect problematic states
        is_ill_cond = cond > 1e5
        is_low_var = rel_var < 1e-3
        is_narrow = eig_spread < 1e-5

        if is_ill_cond or is_low_var or is_narrow:
            # Aggressive exploration mode
            ccov_eff = min(self.ccov * 5.0, 0.5)
            cc_eff = min(self.cc * 2.0, 0.3)
            # Inject diagonal exploration
            rank_mu += 0.2 * np.eye(self.dim)
            ensemble_rank_one += 0.1 * np.eye(self.dim)
        else:
            ccov_eff = self.ccov
            cc_eff = self.cc

        # Final covariance update with ensemble direction
        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * ensemble_rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu)

        # Emergency repair for extreme ill-conditioning
        if cond > 1e8:
            self.C = 0.5 * self.C + 0.5 * np.eye(self.dim) * np.mean(eigvals)
            self.pc *= 0.1

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_03_idea_0.py (4 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Proactive stagnation-guided escaping with directional perturbations."""
        stagnation_threshold = max(40, self.dim * 3)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / (max(eig_min, 1e-10))

        if not hasattr(self, 'best_history'):
            self.best_history = []

        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for hist_pos in self.best_history:
            if np.linalg.norm(self.x_opt - hist_pos) < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.best_history) < 15:
            self.best_history.append(self.x_opt.copy())
        elif is_distinct and len(self.best_history) >= 15:
            worst_idx = np.argmax([np.linalg.norm(self.x_opt - p) for p in self.best_history])
            self.best_history[worst_idx] = self.x_opt.copy()

        if is_stagnant or cond > 1e6:
            escape_dir = np.zeros(self.dim)
            if len(self.best_history) >= 3:
                centroid = np.mean(self.best_history, axis=0)
                toward_centroid = centroid - self.mean
                norm_toward = np.linalg.norm(toward_centroid)
                if norm_toward > 1e-10:
                    toward_centroid /= norm_toward
                    escape_dir = toward_centroid
                else:
                    escape_dir = self.mean - centroid
                    norm_dir = np.linalg.norm(escape_dir)
                    if norm_dir > 1e-10:
                        escape_dir /= norm_dir
            else:
                escape_dir = np.random.randn(self.dim)
                norm_dir = np.linalg.norm(escape_dir)
                if norm_dir > 1e-10:
                    escape_dir /= norm_dir

            escape_strength = 0.5 * (1.0 + self.stagnation_counter / max(stagnation_threshold, 1))
            escape_perturb = escape_strength * np.outer(escape_dir, escape_dir)
            self.C += escape_perturb
            self.C += 0.3 * np.eye(self.dim)
            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.1
            self.stagnation_counter = 0

        y_mean = (self.mean - self.old_mean) / self.sigma

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if is_stagnant:
            cc_scale = 1.5
            ccov_scale = 2.0
        else:
            cc_scale = 1.0
            ccov_scale = min(1.0 + 3.0 * rel_var, 3.0)

        cc_adapt = np.clip(self.cc * cc_scale, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.6)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_07_idea_0.py (3 wins) ---
```python
def _adapt_covariance_variant_06(self):
        """Fitness gradient-directed covariance adaptation with stagnation escape."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute fitness gradient matrix: captures which directions improve fitness
        fit_diff = self.fitness - np.mean(self.fitness)
        fit_diff = fit_diff / (np.std(fit_diff) + 1e-10)

        grad_matrix = np.zeros((self.dim, self.dim))
        for i in range(min(self.mu, len(fit_diff))):
            diff = (self.population[i] - self.old_mean) / self.sigma
            grad_matrix += fit_diff[i] * np.outer(diff, diff)

        # Eigendecomposition of covariance
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)

        # Project gradient onto eigendirections to find most promising axes
        grad_proj = np.zeros(self.dim)
        for j in range(self.dim):
            proj = 0.0
            for i in range(min(self.mu, len(fit_diff))):
                diff = (self.population[i] - self.old_mean) / self.sigma
                proj += fit_diff[i] * np.dot(eigvecs[:, j], diff)
            grad_proj[j] = proj

        # Compute directional weights: positive = improving direction
        dir_weights = np.zeros(self.dim)
        for j in range(self.dim):
            if eigvals[j] > 1e-10:
                dir_weights[j] = np.tanh(grad_proj[j] / (np.sqrt(eigvals[j]) * self.mu + 1e-10))

        # Scale eigendirections by directional weights
        eig_scale = np.ones(self.dim)
        for j in range(self.dim):
            if dir_weights[j] < -0.2:
                eig_scale[j] = 0.5 + 0.5 * dir_weights[j]
            elif dir_weights[j] > 0.2:
                eig_scale[j] = 1.0 + 0.3 * dir_weights[j]

        # Adapt cc based on gradient strength
        grad_strength = np.linalg.norm(grad_proj) / np.sqrt(self.dim)
        cc_adapt = self.cc * np.clip(1.0 + 0.5 * np.tanh(grad_strength - 1.0), 0.01, 0.3)

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Compute rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adapt ccov based on gradient strength
        ccov_adapt = self.ccov * np.clip(1.0 + 0.5 * np.tanh(grad_strength - 0.5), 0.1, 5.0)
        ccov_adapt = np.clip(ccov_adapt, 1e-10, 0.5)

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        # Apply directional scaling to eigendirections
        for j in range(self.dim):
            self.C += (eig_scale[j] - 1.0) * eigvals[j] * np.outer(eigvecs[:, j], eigvecs[:, j])

        # Check for stagnation
        stagnation_threshold = max(20, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            self.stagnation_counter = 0

            # Inject directional perturbation aligned with gradient
            for j in range(self.dim):
                if grad_proj[j] > 0:
                    strength = 1.0 + 0.5 * np.tanh(grad_proj[j])
                else:
                    strength = 1.0 - 0.3 * np.tanh(-grad_proj[j])
                self.C += strength * eigvals[j] * np.outer(eigvecs[:, j], eigvecs[:, j])

            # Add exploration noise
            self.C += 0.1 * np.eye(self.dim)

            self.pc *= 0.1
            self.L = None

        self.C = self._ensure_positive_definite(self.C)

        # Reset on extreme ill-conditioning
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals_check) / np.min(eigvals_check)
        if cond > 1e8:
            self.C = np.eye(self.dim) * np.mean(eigvals_check)
            self.pc = np.zeros(self.dim)
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
        """Explosive eigendirection perturbation on stagnation for multimodal escape."""
        stagnation_threshold = max(30, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            # Explosive exploration: perturb along all principal eigendirections
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)

            # Perturb along top k eigenvectors with decreasing strength
            k = min(self.dim, max(3, self.dim // 4))
            for i in range(k):
                strength = 2.0 * (k - i) / k  # stronger for top eigenvectors
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            # Also add axis-aligned perturbation for uniform coverage
            self.C += 0.5 * np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.1  # reset evolution path
            self.stagnation_counter = 0
        else:
            # Standard covariance update with improvement-weighted scaling
            y_mean = (self.mean - self.old_mean) / self.sigma

            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            improv_scale = np.clip(1.0 + 10.0 * np.log1p(improvement * 1e6), 0.1, 5.0)

            cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals_check) / np.min(eigvals_check)
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