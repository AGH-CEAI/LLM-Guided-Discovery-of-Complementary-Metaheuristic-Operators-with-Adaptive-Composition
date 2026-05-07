Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_generate_parameters_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.000000e-08            1.000000e-08            1.000000e-08            8.570458e-06            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.673925e-04            6.180590e-02            
1      1.000000e-08            1.000000e-08            1.000000e-08            4.809438e-02            4.893855e-08            2.360718e-08            1.000000e-08            1.000000e-08            1.675155e-04            1.473608e+00            2.317361e+01            
2      1.000000e-08            1.000000e-08            1.000000e-08            9.942506e-04            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.292374e-02            2.359114e+00            
3      1.429236e-04            1.373058e-04            1.451578e-04            1.993780e+00            2.036728e-03            1.444761e-04            1.915003e-04            1.495645e-04            7.442172e-01            7.001821e+00            9.582371e+01            
4      1.000000e-08            1.000000e-08            1.000000e-08            4.983961e-01            1.000000e-08            3.999989e-08            1.000000e-08            1.000000e-08            2.365705e-01            1.133763e+00            1.919922e+00            
5      1.000000e-08            1.000000e-08            1.000000e-08            6.428426e-01            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            3.361219e-06            2.286048e+03            1.295539e+06            
6      8.808782e-08            5.577221e-08            1.187928e-07            7.465090e+00            1.024194e-04            9.692006e-08            8.255431e-06            1.283852e-05            4.951712e+00            1.228774e+02            7.010245e+02            
7      4.409885e-06            3.432865e-06            4.624922e-06            7.998286e-01            2.695580e-05            4.571139e-06            4.698994e-06            4.782648e-06            3.781318e-02            7.742081e+00            1.002022e+02            
8      1.000000e-08            1.000000e-08            7.419347e-04            1.030807e+00            9.763930e-04            8.189561e-04            7.875371e-04            7.932558e-04            8.633572e-02            4.072898e+00            1.579262e+01            
9      2.006391e+00            2.439461e-04            7.871227e-01            5.535389e+00            7.741496e-01            6.065413e-01            3.338368e+00            2.938092e+00            5.633617e+00            3.483817e+01            8.490115e+01            
10     1.071177e-02            1.000000e-08            4.688855e-08            2.002933e+00            6.479705e-06            3.037523e-01            6.130082e-08            1.000000e-08            8.204207e-02            3.571158e+00            2.501453e-05            
11     3.176616e+00            3.349277e+00            2.289543e+00            8.618796e+02            2.625289e+00            2.912061e+02            3.478990e+00            2.906529e+00            4.544311e+02            4.380671e+02            1.500951e+02            
12     1.867489e-01            4.344611e-06            2.319107e-04            5.604486e+00            8.587854e-03            1.281699e+00            5.103009e-04            5.357686e-05            9.334672e-01            9.480689e+00            4.279299e-03            
13     1.644850e+00            1.069922e+00            1.313445e+00            2.107792e+01            1.523808e+00            1.258718e+00            1.381030e+00            1.585965e+00            5.191544e+00            1.745848e+01            2.343755e+01            
14     2.467424e+00            2.423631e+00            2.491266e+00            1.567203e+01            2.853822e+00            2.468665e+00            2.524236e+00            2.428994e+00            1.556156e+01            1.145839e+01            3.361396e+00            
15     1.244168e+00            1.157571e+00            1.180627e+00            4.006185e+00            1.199495e+00            3.855889e+00            1.173118e+00            1.127593e+00            3.824593e+00            3.608943e+00            2.969355e+00            
16     2.514292e+02            1.909850e+02            8.323218e+01            7.674108e+03            6.548691e+01            6.235006e+01            8.969358e+01            1.083286e+02            7.639340e+01            2.868811e+03            1.195869e+03            
17     3.120103e+00            3.120103e+00            3.120103e+00            6.120423e+04            3.120104e+00            4.167353e+01            3.120103e+00            3.120103e+00            7.243554e+02            2.481882e+04            3.243778e+04            
18     3.230850e+00            2.671167e+00            2.672331e+00            5.896256e+01            2.690509e+00            2.644980e+01            2.672527e+00            2.672325e+00            3.575778e+01            4.268421e+01            3.770379e+01            
19     7.576987e+00            6.516026e+00            6.503388e+00            3.326983e+02            9.487224e+00            7.019019e+00            7.371004e+00            7.072162e+00            3.522519e+02            2.145574e+02            6.770590e+01            
20     2.024610e+01            1.113858e+01            6.973670e+00            3.476360e+01            6.819927e+00            3.335332e+01            1.044524e+01            1.511236e+01            3.302474e+01            2.955492e+01            2.143578e+01            
21     4.308028e+00            4.219739e+00            4.191249e+00            5.738158e+00            4.207046e+00            4.200055e+00            4.190760e+00            4.225989e+00            5.727941e+00            5.531202e+00            4.651564e+00            
22     2.333638e+00            2.228259e+00            2.244466e+00            1.836714e+01            3.559540e+00            2.455602e+00            2.343156e+00            2.243596e+00            1.814593e+01            1.595261e+01            1.149643e+01            
23     1.719972e+01            1.676638e+01            3.423437e+01            9.733936e+01            5.439115e+01            6.251356e+00            3.342967e+01            3.578585e+01            9.772118e+01            7.525908e+01            3.634495e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: SKIPPED (trivial)
Task  1: SKIPPED (trivial)
Task  2: SKIPPED (trivial)
Task  3: variant_01_idea_0.py  (error=1.373058e-04)
Task  4: SKIPPED (trivial)
Task  5: SKIPPED (trivial)
Task  6: variant_01_idea_0.py  (error=5.577221e-08)
Task  7: variant_01_idea_0.py  (error=3.432865e-06)
Task  8: original.py  (error=1.000000e-08)
Task  9: variant_01_idea_0.py  (error=2.439461e-04)
Task 10: variant_01_idea_0.py  (error=1.000000e-08)
Task 11: variant_02_idea_0.py  (error=2.289543e+00)
Task 12: variant_01_idea_0.py  (error=4.344611e-06)
Task 13: variant_01_idea_0.py  (error=1.069922e+00)
Task 14: variant_01_idea_0.py  (error=2.423631e+00)
Task 15: variant_07_idea_0.py  (error=1.127593e+00)
Task 16: variant_05_idea_0.py  (error=6.235006e+01)
Task 17: original.py  (error=3.120103e+00)
Task 18: variant_01_idea_0.py  (error=2.671167e+00)
Task 19: variant_02_idea_0.py  (error=6.503388e+00)
Task 20: variant_04_idea_0.py  (error=6.819927e+00)
Task 21: variant_06_idea_0.py  (error=4.190760e+00)
Task 22: variant_01_idea_0.py  (error=2.228259e+00)
Task 23: variant_05_idea_0.py  (error=6.251356e+00)

WIN COUNTS:
  variant_01_idea_0.py: 10 wins
  original.py: 2 wins
  variant_02_idea_0.py: 2 wins
  variant_05_idea_0.py: 2 wins
  variant_07_idea_0.py: 1 wins
  variant_04_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (10 wins) ---
```python
def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values with generation-adaptive scaling for better convergence."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Adaptive scale factor: starts at 0.1, decreases over generations for finer tuning
        # but never goes below 0.02 to maintain some exploration
        base_scale = 0.1
        generation = getattr(self, 'generation', 0)
        scale_f = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))
        scale_cr = max(0.02, base_scale * (1.0 / (1.0 + generation / 200.0)))

        # Vectorized Cauchy distribution for F with rejection sampling
        mu_f = memory_f[indices]
        f_values = np.zeros(n)
        remaining = np.ones(n, dtype=bool)
        max_attempts = 100
        attempt = 0

        while np.any(remaining) and attempt < max_attempts:
            count = np.sum(remaining)
            # Cauchy samples: tan(pi * (U - 0.5))
            u = np.random.uniform(0, 1, size=count)
            cauchy_samples = np.tan(np.pi * (u - 0.5))
            candidates = mu_f[remaining] + scale_f * cauchy_samples

            # Accept those > 0
            valid = candidates > 0
            idx_remaining = np.where(remaining)[0]
            accepted = idx_remaining[valid]
            f_values[accepted] = np.minimum(candidates[valid], 1.0)
            remaining[accepted] = False
            attempt += 1

        # Fallback for any remaining (shouldn't happen but be safe)
        if np.any(remaining):
            f_values[remaining] = np.clip(mu_f[remaining], 0.01, 1.0)

        # Normal distribution for CR with adaptive scale
        cr_values = np.random.normal(memory_cr[indices], scale_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        # For some fraction of population, use more explorative parameters
        # This helps escape local optima on multimodal functions
        explore_mask = np.random.random(n) < 0.1
        if np.any(explore_mask):
            n_explore = np.sum(explore_mask)
            # Use larger F for exploration
            f_values[explore_mask] = np.clip(
                np.random.uniform(0.5, 1.0, size=n_explore), 0.1, 1.0
            )
            # Use higher CR for exploration (more dimensions from donor)
            cr_values[explore_mask] = np.clip(
                np.random.uniform(0.8, 1.0, size=n_explore), 0.0, 1.0
            )

        return f_values, cr_values
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR with bimodal strategy: aggressive explorers + adaptive exploiters."""
        n = self.np_size

        # Determine fraction of explorers - more exploration early, less later
        explore_ratio = 0.4
        n_explore = max(1, int(explore_ratio * n))
        n_exploit = n - n_explore

        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        # --- Exploiters: standard SHADE-style from memory ---
        if n_exploit > 0:
            indices = np.random.randint(0, self.memory_size, size=n_exploit)
            # Cauchy for F
            for i in range(n_exploit):
                attempts = 0
                while attempts < 100:
                    f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                    attempts += 1
                else:
                    f_val = 0.5 + 0.1 * abs(np.random.standard_cauchy())
                f_values[i] = min(f_val, 1.0)
            # Normal for CR
            cr_values[:n_exploit] = np.clip(
                np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0
            )

        # --- Explorers: aggressive parameters for escaping local optima ---
        if n_explore > 0:
            start = n_exploit
            # Bimodal F: mix of high values for large jumps
            # Some very high F (0.7-1.2) for basin escape, some moderate (0.4-0.7)
            high_mask = np.random.random(n_explore) < 0.6
            n_high = np.sum(high_mask)
            n_mod = n_explore - n_high

            explore_f = np.zeros(n_explore)
            if n_high > 0:
                explore_f[high_mask] = np.random.uniform(0.7, 1.2, size=n_high)
            if n_mod > 0:
                explore_f[~high_mask] = np.random.uniform(0.4, 0.8, size=n_mod)

            # Clip F > 1 with wrapping: if F > 1, use 2 - F (reflects back)
            over_one = explore_f > 1.0
            # Allow F slightly > 1 for overshooting (cap at 1.2, then clip)
            explore_f = np.clip(explore_f, 0.1, 1.2)
            f_values[start:] = explore_f

            # High CR for non-separable problems (most hard benchmarks)
            cr_values[start:] = np.clip(
                np.random.normal(0.9, 0.1, size=n_explore), 0.5, 1.0
            )

        # Final safety: ensure all F > 0 and <= 1.2, CR in [0,1]
        f_values = np.clip(f_values, 0.01, 1.2)
        # For the mutation, F > 1 means overshooting which can help escape
        # But clip at 1.0 for safety in the standard DE framework
        f_values = np.minimum(f_values, 1.0)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        # Shuffle to avoid positional bias
        perm = np.random.permutation(n)
        f_values = f_values[perm]
        cr_values = cr_values[perm]

        return f_values, cr_values
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR with bimodal aggressive F and high CR for hard multimodal/non-separable tasks."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Bimodal F: 50% from aggressive high-F mode, 50% from memory-adapted mode
        f_values = np.zeros(n)
        aggressive_mask = np.random.random(n) < 0.5

        for i in range(n):
            if aggressive_mask[i]:
                # Aggressive mode: centered at 0.85 with wider Cauchy spread
                while True:
                    f_val = 0.85 + 0.15 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                f_values[i] = min(f_val, 1.0)
            else:
                # Memory-adapted mode but with inflated center and wider spread
                center = max(memory_f[indices[i]], 0.6)  # floor at 0.6
                while True:
                    f_val = center + 0.2 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                f_values[i] = min(f_val, 1.0)

        # CR: strongly biased toward 1.0 for non-separable problems
        # Mix: 60% high CR mode (centered at 0.95), 40% memory-adapted with inflation
        cr_high_mask = np.random.random(n) < 0.6

        cr_values = np.zeros(n)
        # High CR mode
        high_cr = np.random.normal(0.95, 0.05, size=n)
        # Memory-adapted but inflated CR
        memory_cr_inflated = np.maximum(memory_cr[indices], 0.7)
        adapted_cr = np.random.normal(memory_cr_inflated, 0.1)

        cr_values = np.where(cr_high_mask, high_cr, adapted_cr)
        cr_values = np.clip(cr_values, 0.0, 1.0)

        # Ensure minimum F for stability
        f_values = np.maximum(f_values, 0.1)

        return f_values, cr_values
```

# --- From variant_05_idea_0.py (2 wins) ---
```python
def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR with bimodal distributions for enhanced exploration."""
        n = self.np_size
        gen = getattr(self, 'generation', 0)

        # Adaptive exploration ratio: starts high, decays slowly
        explore_ratio = max(0.3, 0.85 - gen * 0.003)

        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        # Decide per-individual: explore or exploit
        is_explorer = np.random.random(n) < explore_ratio
        n_explore = np.sum(is_explorer)
        n_exploit = n - n_explore

        # EXPLORERS: Large F values for long jumps, bimodal CR
        if n_explore > 0:
            # F from heavy-tailed distribution centered at 0.85
            f_explore = np.random.normal(0.85, 0.15, n_explore)
            # Occasionally use very large F (> 1.0 capped at 1.0)
            boost = np.random.random(n_explore) < 0.2
            f_explore[boost] = np.random.uniform(0.9, 1.0, np.sum(boost))
            f_explore = np.clip(f_explore, 0.1, 1.0)
            f_values[is_explorer] = f_explore

            # Bimodal CR: either very low or very high
            cr_explore = np.zeros(n_explore)
            high_cr_mask = np.random.random(n_explore) < 0.6
            cr_explore[high_cr_mask] = np.random.normal(0.95, 0.05, np.sum(high_cr_mask))
            cr_explore[~high_cr_mask] = np.random.normal(0.1, 0.05, np.sum(~high_cr_mask))
            cr_explore = np.clip(cr_explore, 0.0, 1.0)
            cr_values[is_explorer] = cr_explore

        # EXPLOITERS: Standard SHADE-like from memory
        if n_exploit > 0:
            indices = np.random.randint(0, self.memory_size, size=n_exploit)

            # Cauchy for F
            f_exploit = np.zeros(n_exploit)
            for i in range(n_exploit):
                for _ in range(100):
                    f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        f_exploit[i] = min(f_val, 1.0)
                        break
                else:
                    f_exploit[i] = 0.5
            f_values[~is_explorer] = f_exploit

            # Normal for CR
            cr_exploit = np.clip(np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0)
            cr_values[~is_explorer] = cr_exploit

        return f_values, cr_values
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR using bimodal regime mixing with overshooting."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Determine regime: exploration vs exploitation
        # More exploration early, shifts toward exploitation over time
        explore_rate = max(0.3, 0.7 - self.generation * 0.002)
        is_explore = np.random.random(n) < explore_rate

        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        n_explore = np.sum(is_explore)
        n_exploit = n - n_explore

        # Exploration regime: large F from Cauchy centered at 0.9, allow up to 1.5
        if n_explore > 0:
            explore_f = np.zeros(n_explore)
            for j in range(n_explore):
                attempts = 0
                while attempts < 100:
                    f_val = 0.9 + 0.2 * np.random.standard_cauchy()
                    if f_val > 0.4:
                        break
                    attempts += 1
                explore_f[j] = min(f_val, 1.5)
            f_values[is_explore] = explore_f

            # High CR for exploration (non-separable problems benefit)
            cr_explore = np.clip(np.random.normal(0.9, 0.05, n_explore), 0.5, 1.0)
            cr_values[is_explore] = cr_explore

        # Exploitation regime: memory-guided small F, adaptive CR
        if n_exploit > 0:
            exploit_indices = indices[~is_explore]
            exploit_f = np.zeros(n_exploit)
            for j in range(n_exploit):
                attempts = 0
                while attempts < 100:
                    f_val = memory_f[exploit_indices[j]] + 0.1 * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                    attempts += 1
                exploit_f[j] = min(f_val, 1.0)
            f_values[~is_explore] = exploit_f

            cr_exploit = np.clip(
                np.random.normal(memory_cr[exploit_indices], 0.1), 0.0, 1.0
            )
            cr_values[~is_explore] = cr_exploit

        # Occasionally inject extreme diversity: a few individuals with very large F
        n_extreme = max(1, n // 10)
        extreme_idx = np.random.choice(n, n_extreme, replace=False)
        f_values[extreme_idx] = np.clip(1.0 + 0.3 * np.abs(np.random.standard_cauchy(n_extreme)), 1.0, 1.5)
        cr_values[extreme_idx] = np.random.uniform(0.8, 1.0, n_extreme)

        return f_values, cr_values
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR with bimodal distribution: aggressive explorers + local refiners."""
        n = self.np_size
        f_values = np.zeros(n)
        cr_values = np.zeros(n)

        # Determine generation-dependent exploration ratio (more exploration early)
        gen_ratio = min(1.0, self.generation / max(1, 200))
        explore_frac = 0.6 * (1.0 - 0.5 * gen_ratio)  # 0.6 -> 0.3 over time

        n_explore = int(n * explore_frac)
        n_refine = n - n_explore

        # === Aggressive Explorers: high F, high CR ===
        # Use Lévy-like heavy-tailed distribution centered at high F
        if n_explore > 0:
            # Base F from 0.7-1.0 range with heavy tails
            base_f_explore = np.random.uniform(0.6, 1.0, size=n_explore)
            # Add Cauchy perturbation with larger scale
            cauchy_perturb = 0.2 * np.random.standard_cauchy(size=n_explore)
            f_explore = base_f_explore + cauchy_perturb
            # Regenerate any non-positive values
            bad = f_explore <= 0
            while np.any(bad):
                f_explore[bad] = np.random.uniform(0.5, 1.0, size=np.sum(bad))
                bad = f_explore <= 0
            f_explore = np.clip(f_explore, 0.01, 1.5)  # Allow F > 1 for more exploration
            f_explore = np.minimum(f_explore, 1.0)

            # High CR for non-separable functions
            cr_explore = np.clip(np.random.normal(0.9, 0.1, size=n_explore), 0.5, 1.0)

            f_values[:n_explore] = f_explore
            cr_values[:n_explore] = cr_explore

        # === Local Refiners: memory-guided with wider spread ===
        if n_refine > 0:
            indices = np.random.randint(0, self.memory_size, size=n_refine)

            # Cauchy for F with adaptive scale
            for i in range(n_refine):
                scale = 0.15  # Slightly wider than default
                attempts = 0
                while attempts < 50:
                    f_val = memory_f[indices[i]] + scale * np.random.standard_cauchy()
                    if f_val > 0:
                        break
                    attempts += 1
                if attempts >= 50:
                    f_val = memory_f[indices[i]]
                f_values[n_explore + i] = min(f_val, 1.0)

            # CR from memory with wider variance
            cr_refine = np.clip(
                np.random.normal(memory_cr[indices], 0.15), 0.0, 1.0
            )
            cr_values[n_explore:] = cr_refine

        # Shuffle to avoid positional bias
        perm = np.random.permutation(n)
        f_values = f_values[perm]
        cr_values = cr_values[perm]

        return f_values, cr_values
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


class AdaptiveDEWithRestarts:
    """
    Differential Evolution with adaptive parameters, multiple mutation strategies,
    opposition-based learning, and intelligent restart mechanism.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(8 * dim, 300)
        self.archive_max = self.np_size
        self.memory_size = 5
        self.p_best_rate = 0.15
        self.restart_threshold = 50
        self.min_pop_size = max(4, dim // 2)

    def __call__(self, func, stopping_condition):
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.generation = 0

        population, fitness = self._initialize_population(func, stopping_condition)
        if stopping_condition():
            return self.f_opt, self.x_opt

        self._update_best(population, fitness)
        archive = np.empty((0, self.dim))
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        prev_best_fitness = self.f_opt

        while not stopping_condition():
            # Generate adaptive parameters
            f_values, cr_values = self._generate_parameters_batch(memory_f, memory_cr)

            # Select mutation strategy indices
            strategy_mask = self._select_strategies_batch()

            # Generate mutant vectors
            mutants = self._mutate_batch(population, fitness, archive, f_values, strategy_mask)

            # Crossover
            trials = self._crossover_batch(population, mutants, cr_values)

            # Clip to bounds
            trials = self._clip_to_bounds(trials)

            # Evaluate
            trial_fitness = func(trials)
            if stopping_condition():
                self._safe_update_best(trials, trial_fitness)
                break

            # Handle truncated results
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                f_values = f_values[:len(trial_fitness)]
                cr_values = cr_values[:len(trial_fitness)]

            valid = ~np.isnan(trial_fitness)
            
            # Selection and archive update
            population, fitness, archive, success_f, success_cr, success_delta = (
                self._select_survivors_batch(
                    population, fitness, trials, trial_fitness, archive, f_values, cr_values, valid
                )
            )

            # Update parameter memory
            memory_f, memory_cr, memory_idx = self._update_memory(
                memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta
            )

            self._update_best(population, fitness)

            # Detect stagnation
            stagnant = self._detect_stagnation(prev_best_fitness)
            if stagnant:
                population, fitness, archive, memory_f, memory_cr, memory_idx = (
                    self._restart(func, stopping_condition, population, fitness)
                )
                if stopping_condition():
                    break

            prev_best_fitness = self.f_opt
            self.generation += 1

            # Periodically try opposition-based learning
            if self.generation % 25 == 0 and not stopping_condition():
                population, fitness = self._opposition_based_jump(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
                self._update_best(population, fitness)

        return self.f_opt, self.x_opt

    def _initialize_population(self, func, stopping_condition):
        """Create initial random population and evaluate."""
        population = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        fitness = func(population)
        if len(fitness) < len(population):
            population = population[:len(fitness)]
        return population, fitness

    def _generate_parameters_batch(self, memory_f, memory_cr):
        """Generate F and CR values for entire population from parameter memory."""
        n = self.np_size
        indices = np.random.randint(0, self.memory_size, size=n)

        # Cauchy distribution for F
        f_values = np.zeros(n)
        for i in range(n):
            while True:
                f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    break
            f_values[i] = min(f_val, 1.0)

        # Normal distribution for CR
        cr_values = np.clip(
            np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0
        )

        return f_values, cr_values

    def _select_strategies_batch(self):
        """Select mutation strategy for each individual. Returns boolean mask."""
        # True = current-to-pbest/1, False = rand-to-pbest/1
        return np.random.random(self.np_size) < 0.7

    def _mutate_batch(self, population, fitness, archive, f_values, strategy_mask):
        """Generate mutant vectors using mixed strategies for the whole population."""
        n = len(population)
        dim = self.dim

        # Sort by fitness for p-best selection
        sorted_idx = np.argsort(fitness)
        p = max(2, int(np.ceil(self.p_best_rate * n)))

        # Select p-best indices for each individual
        pbest_idx = sorted_idx[np.random.randint(0, p, size=n)]

        # Select r1 != i
        r1 = self._random_indices_not_equal(n, np.arange(n))

        # Union of population and archive for r2
        if len(archive) > 0:
            union = np.vstack([population, archive])
        else:
            union = population.copy()
        union_size = len(union)

        # Select r2 from union, r2 != i and r2 != r1
        r2 = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                idx = np.random.randint(0, union_size)
                if idx != i and idx != r1[i]:
                    r2[i] = idx
                    break

        f_col = f_values[:, np.newaxis]
        mutants = np.empty((n, dim))

        # Strategy 1: current-to-pbest/1 (for strategy_mask == True)
        mask1 = strategy_mask
        if np.any(mask1):
            idx1 = np.where(mask1)[0]
            mutants[idx1] = (
                population[idx1]
                + f_col[idx1] * (population[pbest_idx[idx1]] - population[idx1])
                + f_col[idx1] * (population[r1[idx1]] - union[r2[idx1]])
            )

        # Strategy 2: rand-to-pbest/1 (for strategy_mask == False)
        mask2 = ~strategy_mask
        if np.any(mask2):
            idx2 = np.where(mask2)[0]
            r_base = self._random_indices_not_equal(n, np.arange(n))[idx2]
            mutants[idx2] = (
                population[r_base]
                + f_col[idx2] * (population[pbest_idx[idx2]] - population[r_base])
                + f_col[idx2] * (population[r1[idx2]] - union[r2[idx2]])
            )

        return mutants

    def _random_indices_not_equal(self, n, exclude_indices):
        """Generate random indices in [0, n) that differ from exclude_indices."""
        result = np.random.randint(0, n - 1, size=len(exclude_indices))
        mask = result >= exclude_indices
        result[mask] += 1
        result = np.clip(result, 0, n - 1)
        return result

    def _crossover_batch(self, population, mutants, cr_values):
        """Binomial crossover applied to whole population at once."""
        n, dim = population.shape
        rand_matrix = np.random.random((n, dim))
        cr_matrix = cr_values[:, np.newaxis]

        # Ensure at least one dimension is from mutant
        j_rand = np.random.randint(0, dim, size=n)
        cross_mask = rand_matrix < cr_matrix
        cross_mask[np.arange(n), j_rand] = True

        trials = np.where(cross_mask, mutants, population)
        return trials

    def _clip_to_bounds(self, trials):
        """Clip trial vectors to search bounds with midpoint reflection."""
        # For out-of-bounds, reflect towards the center
        too_low = trials < self.lb
        too_high = trials > self.ub
        trials = np.clip(trials, self.lb, self.ub)
        return trials

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 archive, f_values, cr_values, valid_mask):
        """Greedy selection: replace if trial is better. Track successful params."""
        n = min(len(population), len(trials))
        success_f = []
        success_cr = []
        success_delta = []

        improved = np.zeros(n, dtype=bool)
        for i in range(n):
            if valid_mask[i] and trial_fitness[i] <= fitness[i]:
                if trial_fitness[i] < fitness[i]:
                    improved[i] = True
                    success_f.append(f_values[i])
                    success_cr.append(cr_values[i])
                    success_delta.append(abs(fitness[i] - trial_fitness[i]))
                # Add old individual to archive
                if len(archive) < self.archive_max:
                    archive = np.vstack([archive, population[i:i+1]]) if len(archive) > 0 else population[i:i+1].copy()
                else:
                    replace_idx = np.random.randint(0, self.archive_max)
                    archive[replace_idx] = population[i]
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

        return (population, fitness, archive,
                np.array(success_f), np.array(success_cr), np.array(success_delta))

    def _update_memory(self, memory_f, memory_cr, memory_idx, success_f, success_cr, success_delta):
        """Update parameter memory using weighted Lehmer mean."""
        if len(success_f) == 0:
            return memory_f, memory_cr, memory_idx

        weights = success_delta / (np.sum(success_delta) + 1e-30)

        # Weighted Lehmer mean for F
        lehmer_f = np.sum(weights * success_f ** 2) / (np.sum(weights * success_f) + 1e-30)
        memory_f[memory_idx] = lehmer_f

        # Weighted arithmetic mean for CR
        mean_cr = np.sum(weights * success_cr)
        memory_cr[memory_idx] = mean_cr

        memory_idx = (memory_idx + 1) % self.memory_size
        return memory_f, memory_cr, memory_idx

    def _detect_stagnation(self, prev_best_fitness):
        """Detect if the algorithm is stagnating."""
        improvement = prev_best_fitness - self.f_opt
        if improvement < 1e-12 * (abs(prev_best_fitness) + 1e-30):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        return self.stagnation_counter >= self.restart_threshold

    def _restart(self, func, stopping_condition, population, fitness):
        """Partial restart: keep best individuals, reinitialize rest with opposition."""
        self.stagnation_counter = 0
        n = len(population)
        keep_count = max(2, n // 5)

        sorted_idx = np.argsort(fitness)
        elite = population[sorted_idx[:keep_count]].copy()
        elite_fit = fitness[sorted_idx[:keep_count]].copy()

        reinit_count = n - keep_count
        # Generate new individuals using quasi-opposition
        new_pop = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

        # Quasi-opposition: reflect around population center
        center = np.mean(elite, axis=0)
        opp_pop = 2.0 * center - new_pop
        opp_pop = np.clip(opp_pop, self.lb, self.ub)

        # Choose random mix of new and opposition
        mix_mask = np.random.random(reinit_count) < 0.5
        combined_new = np.where(mix_mask[:, np.newaxis], opp_pop, new_pop)

        if stopping_condition():
            population = np.vstack([elite, combined_new])
            fitness_new = np.full(reinit_count, np.inf)
            return population, np.concatenate([elite_fit, fitness_new]), np.empty((0, self.dim)), np.full(self.memory_size, 0.5), np.full(self.memory_size, 0.5), 0

        combined_new = self._clip_to_bounds(combined_new)
        new_fitness = func(combined_new)
        if len(new_fitness) < len(combined_new):
            combined_new = combined_new[:len(new_fitness)]

        self._safe_update_best(combined_new, new_fitness)

        population = np.vstack([elite, combined_new[:len(new_fitness)]])
        fitness_all = np.concatenate([elite_fit, new_fitness])

        # Reset memory
        memory_f = np.full(self.memory_size, 0.5)
        memory_cr = np.full(self.memory_size, 0.5)
        memory_idx = 0
        archive = np.empty((0, self.dim))

        self.np_size = len(population)

        return population, fitness_all, archive, memory_f, memory_cr, memory_idx

    def _opposition_based_jump(self, population, fitness, func, stopping_condition):
        """Apply opposition-based learning to diversify population."""
        center = (self.lb + self.ub) / 2.0
        opp_population = 2.0 * center - population
        opp_population = self._clip_to_bounds(opp_population)

        # Add random perturbation
        noise = np.random.normal(0, 1.0, opp_population.shape)
        opp_population = self._clip_to_bounds(opp_population + noise)

        if stopping_condition():
            return population, fitness

        opp_fitness = func(opp_population)
        if len(opp_fitness) < len(opp_population):
            opp_population = opp_population[:len(opp_fitness)]

        self._safe_update_best(opp_population, opp_fitness)

        # Combine and select best
        n = min(len(population), len(opp_population))
        combined_pop = np.vstack([population[:n], opp_population[:n]])
        combined_fit = np.concatenate([fitness[:n], opp_fitness[:n]])

        sorted_idx = np.argsort(combined_fit)
        best_idx = sorted_idx[:len(population)]
        return combined_pop[best_idx], combined_fit[best_idx]

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance."""
        center = np.mean(population, axis=0)
        diffs = population - center
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _update_best(self, population, fitness):
        """Update global best from population."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fitness = fitness[valid]
        valid_pop = population[valid]
        best_idx = np.argmin(valid_fitness)
        if valid_fitness[best_idx] < self.f_opt:
            self.f_opt = valid_fitness[best_idx]
            self.x_opt = valid_pop[best_idx].copy()

    def _safe_update_best(self, candidates, fitnesses):
        """Safely update best considering possible NaN values."""
        if len(fitnesses) == 0:
            return
        valid = ~np.isnan(fitnesses)
        if not np.any(valid):
            return
        best_idx = np.nanargmin(fitnesses)
        if fitnesses[best_idx] < self.f_opt:
            self.f_opt = fitnesses[best_idx]
            self.x_opt = candidates[best_idx].copy()

```