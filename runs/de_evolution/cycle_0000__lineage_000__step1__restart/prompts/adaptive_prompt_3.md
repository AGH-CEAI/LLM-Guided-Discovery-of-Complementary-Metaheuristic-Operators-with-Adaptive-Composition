Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_restart` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.130319e-07            1.133044e-07            1.104649e-07            1.158979e-07            1.181780e-07            1.134999e-07            1.105544e-07            -inf                    1.109955e-07            1.129265e-07            1.173996e-07            
1      2.393772e-07            2.364133e-07            2.365997e-07            2.352114e-07            2.422777e-07            2.374932e-07            2.383230e-07            -inf                    2.318426e-07            2.404806e-07            2.437398e-07            
2      2.450635e-05            2.430630e-05            2.384367e-05            2.431133e-05            2.372680e-05            2.475017e-05            2.415841e-05            -inf                    2.414510e-05            2.424487e-05            2.420268e-05            
3      3.543341e-04            3.469553e-04            3.433881e-04            3.542022e-04            3.547258e-04            3.408535e-04            3.491376e-04            -inf                    3.599999e-04            3.468399e-04            3.492730e-04            
4      2.465698e-02            2.433666e-02            2.444810e-02            2.456322e-02            2.453908e-02            2.474489e-02            2.453807e-02            -inf                    2.447537e-02            2.449332e-02            2.432521e-02            
5      1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            1.000000e-08            -inf                    1.000000e-08            1.000000e-08            1.000000e-08            
6      4.293185e-07            4.470667e-07            4.103743e-07            4.560179e-07            4.358513e-07            4.152130e-07            4.351821e-07            -inf                    4.367874e-07            4.202353e-07            4.317324e-07            
7      1.326514e-05            1.320667e-05            1.328555e-05            1.350237e-05            1.371797e-05            1.394644e-05            1.365621e-05            -inf                    1.379275e-05            1.356528e-05            1.335507e-05            
8      1.760301e-03            1.697227e-03            1.727331e-03            1.720922e-03            1.734595e-03            1.742181e-03            1.727073e-03            -inf                    1.702803e-03            1.737033e-03            1.722461e-03            
9      5.080293e-04            4.966435e-04            5.150019e-04            5.085245e-04            5.024078e-04            5.041848e-04            5.121823e-04            -inf                    5.020999e-04            5.029517e-04            5.015616e-04            
10     2.840159e+01            3.921184e+01            3.436991e+01            4.839481e+01            2.800821e+01            5.148212e+01            2.795846e+01            -inf                    2.706827e+01            2.843143e+01            6.062014e+01            
11     1.503471e+02            1.730162e+02            1.317940e+02            1.741846e+02            1.138449e+02            1.289082e+02            1.428517e+02            -inf                    6.525966e+01            1.789267e+01            1.573916e+02            
12     5.056515e+01            7.308729e+01            6.474984e+01            7.736376e+01            1.807024e+01            7.338511e+01            6.749537e+01            -inf                    1.361078e+01            3.755577e+00            7.633533e+01            
13     1.265450e+01            1.584367e+01            1.228786e+01            1.736511e+01            1.502896e+01            1.478771e+01            1.361948e+01            -inf                    9.708908e+00            1.050540e+01            1.556368e+01            
14     3.526667e+00            3.891463e+00            4.114533e+00            4.012147e+00            4.035834e+00            3.958096e+00            4.453330e+00            -inf                    3.921217e+00            4.105171e+00            4.004353e+00            
15     3.096218e+00            2.855960e+00            3.116460e+00            3.295742e+00            2.657530e+00            3.061792e+00            3.132612e+00            -inf                    2.818083e+00            1.253516e+00            3.041948e+00            
16     1.214422e+03            1.011418e+03            1.351705e+03            2.136505e+03            8.283076e+02            1.171500e+03            1.574869e+03            -inf                    9.986098e+02            1.786117e+03            1.874773e+03            
17     1.152904e+04            1.388472e+04            1.212074e+04            1.522066e+04            1.217600e+04            1.234222e+04            6.987954e+03            -inf                    4.347412e+03            1.246580e+01            3.335643e+03            
18     2.885672e+01            3.194968e+01            2.983813e+01            3.890114e+01            3.055855e+01            3.766186e+01            3.158054e+01            -inf                    2.904941e+01            3.180605e+01            3.520813e+01            
19     6.547631e+01            6.830277e+01            7.982592e+01            7.568772e+01            5.972508e+01            8.412485e+01            5.572432e+01            -inf                    9.517807e+01            8.282576e+01            7.952562e+01            
20     2.491690e+01            2.290264e+01            2.488755e+01            2.507548e+01            2.355967e+01            2.741946e+01            2.851785e+01            -inf                    2.552494e+01            2.843666e+01            2.410175e+01            
21     4.479540e+00            4.571348e+00            4.597492e+00            5.103614e+00            4.552773e+00            4.632795e+00            4.686474e+00            -inf                    4.523905e+00            4.432735e+00            4.598993e+00            
22     1.038131e+01            1.026057e+01            1.301533e+01            1.008723e+01            9.414412e+00            1.057979e+01            9.367205e+00            -inf                    1.017489e+01            9.138212e+00            1.158516e+01            
23     4.361901e+01            4.407219e+01            4.422746e+01            4.203676e+01            4.150014e+01            4.328296e+01            4.132167e+01            -inf                    3.511476e+01            4.750221e+01            4.320773e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_02_idea_0.py  (error=1.104649e-07)
Task  1: variant_08_idea_0.py  (error=2.318426e-07)
Task  2: variant_04_idea_0.py  (error=2.372680e-05)
Task  3: variant_05_idea_0.py  (error=3.408535e-04)
Task  4: variant_10_idea_0.py  (error=2.432521e-02)
Task  5: SKIPPED (trivial)
Task  6: variant_02_idea_0.py  (error=4.103743e-07)
Task  7: variant_01_idea_0.py  (error=1.320667e-05)
Task  8: variant_01_idea_0.py  (error=1.697227e-03)
Task  9: variant_01_idea_0.py  (error=4.966435e-04)
Task 10: variant_08_idea_0.py  (error=2.706827e+01)
Task 11: variant_09_idea_0.py  (error=1.789267e+01)
Task 12: variant_09_idea_0.py  (error=3.755577e+00)
Task 13: variant_08_idea_0.py  (error=9.708908e+00)
Task 14: original.py  (error=3.526667e+00)
Task 15: variant_09_idea_0.py  (error=1.253516e+00)
Task 16: variant_04_idea_0.py  (error=8.283076e+02)
Task 17: variant_09_idea_0.py  (error=1.246580e+01)
Task 18: original.py  (error=2.885672e+01)
Task 19: variant_06_idea_0.py  (error=5.572432e+01)
Task 20: variant_01_idea_0.py  (error=2.290264e+01)
Task 21: variant_09_idea_0.py  (error=4.432735e+00)
Task 22: variant_09_idea_0.py  (error=9.138212e+00)
Task 23: variant_08_idea_0.py  (error=3.511476e+01)

WIN COUNTS:
  variant_09_idea_0.py: 6 wins
  variant_08_idea_0.py: 4 wins
  variant_01_idea_0.py: 4 wins
  variant_02_idea_0.py: 2 wins
  variant_04_idea_0.py: 2 wins
  original.py: 2 wins
  variant_05_idea_0.py: 1 wins
  variant_10_idea_0.py: 1 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (4 wins) ---
```python
def _restart(self):
        """IPOP-style restart: double population size, keep best, initialize mean
        as weighted blend of best-known solution and random point."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # IPOP: double the population size (capped to avoid excessive memory)
        old_pop_size = self.pop_size
        self.pop_size = min(self.pop_size * 2, 512)
        # Keep pop_size even
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2

        # Recompute strategy parameters for new population size
        self._initialize_strategy_params()

        # Re-initialize state (mean, C, paths, sigma, etc.)
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Initialize mean as weighted combination of best and random
        # Use a blend that starts near the best but allows exploration
        # The larger the population, the more we can afford to explore
        random_point = np.random.uniform(self.lb, self.ub, size=self.dim)

        # Adaptive blending weight: more exploration with larger populations
        alpha = min(0.5, old_pop_size / float(self.pop_size))  # starts at 0.5, decreases

        self.mean = np.clip(
            alpha * saved_best_x + (1.0 - alpha) * random_point,
            self.lb, self.ub
        )

        # Adapt initial sigma based on distance from mean to best
        dist_to_best = np.linalg.norm(self.mean - saved_best_x)
        if dist_to_best > 1e-10:
            # Set sigma proportional to distance, but capped
            self.sigma = np.clip(dist_to_best / np.sqrt(self.dim), 1.0, 50.0)
        else:
            # If mean is very close to best, use moderate sigma for exploration
            self.sigma = 30.0
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _restart(self):
        """IPOP-style restart: double pop size, opposition-based diversification, full sigma."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track previous mean positions to avoid revisiting (store on first use)
        if not hasattr(self, '_previous_means'):
            self._previous_means = []
            self._restart_count = 0

        self._previous_means.append(self.mean.copy())
        self._restart_count += 1

        # IPOP: increase population size (double every restart, cap at reasonable size)
        old_pop_size = self.pop_size
        new_pop_size = min(old_pop_size * 2, 512)  # cap to avoid excessive evaluations

        # Reinitialize state
        self._initialize_state()

        # Apply increased population size
        self.pop_size = new_pop_size
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Use larger sigma to cover more of the search space
        self.sigma = 50.0 + 20.0 * min(self._restart_count, 5)

        # Diversify mean initialization: try to go far from previous basins
        if self._restart_count % 3 == 1:
            # Opposition-based: reflect best through center of domain
            center = (self.lb + self.ub) / 2.0
            self.mean = np.clip(2.0 * center - saved_best_x, self.lb, self.ub)
            # Add noise
            self.mean += np.random.randn(self.dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)
        elif self._restart_count % 3 == 2 and len(self._previous_means) > 0:
            # Maximize distance from all previous means
            candidate_means = []
            for _ in range(50):
                candidate = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
                min_dist = min(np.linalg.norm(candidate - pm) for pm in self._previous_means)
                candidate_means.append((min_dist, candidate))
            candidate_means.sort(key=lambda x: -x[0])
            self.mean = np.clip(candidate_means[0][1], self.lb, self.ub)
        else:
            # Random with wide spread
            self.mean = np.random.uniform(self.lb * 0.9, self.ub * 0.9, size=self.dim)

        # Reset covariance to identity scaled appropriately
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _restart(self):
        """IPOP-style restart: double population on restarts + alternate between exploration and local refinement."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count via an attribute
        if not hasattr(self, 'restart_count'):
            self.restart_count = 0
        self.restart_count += 1

        # IPOP: increase population size every other restart (capped)
        if self.restart_count % 2 == 0:
            # Double population for exploration restarts
            new_pop_size = min(self.pop_size * 2, 256)
            if new_pop_size % 2 != 0:
                new_pop_size += 1
            self.pop_size = new_pop_size
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()

        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        if self.restart_count % 3 == 0:
            # Local refinement restart: small sigma around best solution
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 0.5, self.lb, self.ub)
            self.sigma = 2.0  # small step size for fine-tuning
        elif self.restart_count % 3 == 1:
            # Medium-range exploration restart: moderate perturbation from best
            scale = 10.0 + 20.0 * np.random.rand()
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * scale, self.lb, self.ub)
            self.sigma = 20.0
        else:
            # Full random restart with large population (IPOP exploration)
            self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
            self.sigma = 50.0

        # Occasionally initialize covariance with diagonal scaling to handle ill-conditioning
        if self.restart_count % 4 == 0 and self.restart_count > 2:
            # Use a random diagonal covariance to break symmetry
            diag_scales = np.exp(np.random.randn(self.dim) * 1.5)
            diag_scales = np.clip(diag_scales, 0.1, 10.0)
            self.C = np.diag(diag_scales)
            self.D = np.sqrt(diag_scales)
            self.B = np.eye(self.dim)
            inv_D = 1.0 / self.D
            self.invsqrt_C = np.diag(inv_D)
```

# --- From variant_05_idea_0.py (1 wins) ---
```python
def _restart(self):
        """IPOP restart: double population size each restart, recompute strategy params."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count
        if not hasattr(self, 'restart_count'):
            self.restart_count = 0
        self.restart_count += 1

        # IPOP: double population size each restart, up to a limit
        max_pop = max(256, 20 * self.dim)
        new_pop_size = min(self.pop_size * 2, max_pop)
        self.pop_size = new_pop_size
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2

        # Recompute all strategy parameters for new population size
        self._initialize_strategy_params()

        # Re-initialize state
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Larger initial sigma for broader exploration
        self.sigma = 50.0 + 10.0 * min(self.restart_count, 5)

        # Alternate between random restarts and biased restarts
        if self.restart_count % 3 == 0:
            # Fully random restart - explore new basin
            self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
        elif self.restart_count % 3 == 1:
            # Restart near best with large perturbation
            scale = 30.0 + 10.0 * min(self.restart_count, 5)
            perturbation = np.random.randn(self.dim) * scale
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
        else:
            # Opposition-based restart: reflect best through center of domain
            center = (self.lb + self.ub) / 2.0
            opposite = 2.0 * center - saved_best_x
            noise = np.random.randn(self.dim) * 15.0
            self.mean = np.clip(opposite + noise, self.lb, self.ub)

        # Reset covariance to identity with slight random perturbation
        # to break symmetry in search directions
        diag_vals = np.exp(np.random.randn(self.dim) * 0.3)
        self.C = np.diag(diag_vals)
        self.D = np.sqrt(diag_vals)
        self.B = np.eye(self.dim)
        self.invsqrt_C = np.diag(1.0 / np.sqrt(diag_vals))
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _restart(self):
        """IPOP-style restart: double pop size each restart, opposition-based diversity."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count
        if not hasattr(self, 'restart_count'):
            self.restart_count = 0
        self.restart_count += 1

        # IPOP: double population size on each restart (capped)
        old_pop_size = self.pop_size
        self.pop_size = min(old_pop_size * 2, 256)
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2

        # Recompute strategy parameters for new pop size
        self._initialize_strategy_params()

        # Reinitialize state with new dimensions
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Use larger sigma to cover more of the search space
        self.sigma = min(60.0 * (1.0 + 0.5 * self.restart_count), 100.0)

        # Alternate restart strategies based on restart count
        strategy = self.restart_count % 4

        if strategy == 0:
            # Opposition-based: start from the point opposite to best
            self.mean = np.clip(-saved_best_x, self.lb, self.ub)
        elif strategy == 1:
            # Random point far from best
            direction = np.random.randn(self.dim)
            direction /= max(np.linalg.norm(direction), 1e-30)
            distance = 50.0 + 50.0 * np.random.rand()
            self.mean = np.clip(saved_best_x + direction * distance, self.lb, self.ub)
        elif strategy == 2:
            # Fully random in the domain (maximum diversity)
            self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
        else:
            # Mirrored sampling around center of domain with perturbation
            center = np.zeros(self.dim)
            perturbation = np.random.randn(self.dim) * 40.0
            self.mean = np.clip(center + perturbation, self.lb, self.ub)

        # Reset covariance to scaled identity to avoid inheriting bad structure
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.eigen_decomp_gen = 0
```

# --- From variant_08_idea_0.py (4 wins) ---
```python
def _restart(self):
        """IPOP-style restart: double pop size each time, alternate global/local phases."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count via an attribute
        if not hasattr(self, 'restart_count'):
            self.restart_count = 0
        self.restart_count += 1

        # IPOP: increase population size (up to a limit)
        if not hasattr(self, 'initial_pop_size'):
            self.initial_pop_size = self.pop_size

        # Double pop size every restart, cap at reasonable limit
        new_pop_size = min(self.initial_pop_size * (2 ** self.restart_count), 
                           max(256, 20 * self.dim))
        if new_pop_size % 2 != 0:
            new_pop_size += 1

        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2

        # Recompute strategy parameters for new pop size
        self._initialize_strategy_params()

        # Re-initialize state
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Alternate between global exploration and local refinement
        if self.restart_count % 3 == 0:
            # Local refinement: small sigma, mean near best
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 2.0, self.lb, self.ub)
            self.sigma = max(5.0, 0.1 * np.std(saved_best_x - self.mean + 1e-10))
            # Reset to smaller pop for local search
            self.pop_size = self.initial_pop_size
            if self.pop_size % 2 != 0:
                self.pop_size += 1
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
        elif self.restart_count % 3 == 1:
            # Global exploration: large sigma, random mean, large pop
            self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
            self.sigma = 50.0
        else:
            # Semi-local: moderate perturbation from best
            scale = min(30.0, 10.0 * self.restart_count)
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * scale, self.lb, self.ub)
            self.sigma = scale

        # Reset covariance to identity (fresh start)
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.eigen_decomp_gen = 0
        self.stagnation_counter = 0
```

# --- From variant_09_idea_0.py (6 wins) ---
```python
def _restart(self):
        """IPOP-style restart: double pop size, alternate random/biased starts, large sigma."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count
        if not hasattr(self, '_restart_count'):
            self._restart_count = 0
            self._base_pop_size = self.pop_size
        self._restart_count += 1

        # IPOP: increase population size (double each restart, cap at reasonable limit)
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)  # cap to avoid excessive evals
        if new_pop_size % 2 != 0:
            new_pop_size += 1

        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2

        # Recompute strategy parameters for new pop size
        self._initialize_strategy_params()

        # Re-initialize state
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Alternate between strategies based on restart count
        strategy = self._restart_count % 4

        if strategy == 0:
            # Fully random restart - explore completely new region
            self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
            self.sigma = 50.0  # large sigma for broad exploration
        elif strategy == 1:
            # Start near best but with very large sigma
            perturbation = np.random.randn(self.dim) * 40.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
            self.sigma = 60.0
        elif strategy == 2:
            # Opposition-based restart: mirror best through center
            center = np.zeros(self.dim)  # center of domain
            opposite = 2.0 * center - saved_best_x
            noise = np.random.randn(self.dim) * 10.0
            self.mean = np.clip(opposite + noise, self.lb, self.ub)
            self.sigma = 40.0
        else:
            # Start at best with moderate sigma for local refinement
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
            self.sigma = 15.0

        # Every 4th restart cycle, reset pop size to avoid getting stuck with huge pop
        if self._restart_count % 8 == 0:
            self._restart_count = 0
```

# --- From variant_10_idea_0.py (1 wins) ---
```python
def _restart(self):
        """IPOP/BIPOP-style restart: alternate between large-pop global and small-pop local restarts."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count
        if not hasattr(self, '_restart_count'):
            self._restart_count = 0
            self._base_pop_size = self.pop_size
            self._best_sigma_at_solution = 30.0

        self._restart_count += 1

        # Alternate between global (IPOP - increasing pop) and local (small pop, small sigma)
        do_local = (self._restart_count % 3 == 0) and saved_best_f < np.inf

        if do_local:
            # Local restart: small population, small sigma, centered on best
            self._initialize_state()
            self.pop_size = max(self._base_pop_size, 8)
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()

            # Center on best with small perturbation and small sigma
            scale = max(1e-3, min(10.0, abs(saved_best_f) * 1e-4))
            scale = min(scale, 5.0)
            perturbation = np.random.randn(self.dim) * scale
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
            self.sigma = max(scale * 2.0, 0.01)
        else:
            # IPOP restart: double population size each time (up to a limit)
            new_pop_size = min(self._base_pop_size * (2 ** min(self._restart_count // 2, 5)), 
                              self._base_pop_size * 32)
            if new_pop_size % 2 != 0:
                new_pop_size += 1

            self._initialize_state()
            self.pop_size = new_pop_size
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()

            # Randomize initial mean broadly
            self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)

            # Larger initial sigma for broader search
            self.sigma = 40.0 + 10.0 * min(self._restart_count, 5)

            # With some probability, bias toward best region
            if np.random.rand() < 0.2 and saved_best_f < np.inf:
                perturbation = np.random.randn(self.dim) * 30.0
                self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)

        # Always preserve best solution
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Reset stagnation counter with longer patience for larger populations
        self.stagnation_counter = 0
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


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES-inspired optimizer with:
    - Covariance matrix adaptation via rank-1 and rank-mu updates
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update
    - Periodic restarts on stagnation
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))  # ~14 for dim=30
        self.pop_size = max(self.pop_size, 8)
        # Make pop_size even
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2  # number of parents
        self._initialize_strategy_params()

    def _initialize_strategy_params(self):
        """Set up all CMA-ES-like strategy parameters."""
        dim = self.dim
        mu = self.mu
        lam = self.pop_size

        # Recombination weights
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size adaptation parameters
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        # Covariance adaptation parameters
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        # Expected length of N(0,I) vector
        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        """Initialize the mean, covariance, evolution paths, and step size."""
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0  # initial step size
        self.C = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.eigen_decomp_gen = 0
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.invsqrt_C = np.eye(dim)
        self.generation = 0
        self.best_fitness = np.inf
        self.best_x = self.mean.copy()
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def _update_eigen_decomposition(self):
        """Recompute eigen decomposition of C if needed (every few gens)."""
        update_interval = max(1, int(1.0 / (self.c_1 + self.c_mu) / self.dim / 10.0))
        if self.generation - self.eigen_decomp_gen >= update_interval:
            # Enforce symmetry
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigenvalues, self.B = np.linalg.eigh(self.C)
            # Clamp eigenvalues for numerical stability
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.sqrt(eigenvalues)
            inv_D = 1.0 / self.D
            self.invsqrt_C = self.B @ np.diag(inv_D) @ self.B.T
            self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        """Sample a batch of lambda offspring from N(mean, sigma^2 * C)."""
        dim = self.dim
        lam = self.pop_size
        # z ~ N(0, I)
        z = np.random.randn(lam, dim)
        # y = B * D * z  (transform to N(0, C))
        y = z @ np.diag(self.D) @ self.B.T
        # x = mean + sigma * y
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, y, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search domain."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, y_vectors):
        """Sort population by fitness, return sorted arrays."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, y_vectors
        # Put invalid (NaN/inf) at the end with large fitness
        sort_fit = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fit)
        return population[order], sort_fit[order], y_vectors[order]

    def _update_mean(self, sorted_pop, sorted_y):
        """Update the distribution mean using weighted recombination of the mu best."""
        old_mean = self.mean.copy()
        self.mean = self.weights @ sorted_pop[:self.mu]
        # Weighted mean of y-vectors (steps in original space)
        y_w = self.weights @ sorted_y[:self.mu]
        return old_mean, y_w

    def _update_evolution_path_sigma(self, y_w):
        """Update the conjugate evolution path for step-size control."""
        c_s = self.c_sigma
        self.p_sigma = (1.0 - c_s) * self.p_sigma + np.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (self.invsqrt_C @ y_w)

    def _update_evolution_path_c(self, y_w):
        """Update the evolution path for covariance matrix adaptation."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        h_sigma_threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if p_sigma_norm < h_sigma_threshold else 0.0

        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_w)
        return h_sigma

    def _update_covariance_matrix(self, sorted_y, h_sigma):
        """Rank-1 and rank-mu update of the covariance matrix."""
        dim = self.dim
        # Rank-1 update
        rank1 = np.outer(self.p_c, self.p_c)

        # Rank-mu update
        y_sel = sorted_y[:self.mu]  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_sel[i], y_sel[i])

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        self.C = ((1.0 - self.c_1 - self.c_mu + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu * rank_mu)

    def _update_step_size(self):
        """CSA: cumulative step-size adaptation."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        # Clamp sigma to prevent explosion or collapse
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)

    def _track_best(self, sorted_pop, sorted_fit):
        """Track the overall best solution found."""
        if len(sorted_fit) == 0:
            return
        valid = np.isfinite(sorted_fit)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, sorted_fit, np.inf))
        if sorted_fit[best_idx] < self.best_fitness:
            self.best_fitness = sorted_fit[best_idx]
            self.best_x = sorted_pop[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def _detect_stagnation(self):
        """Check if the algorithm has stagnated and needs a restart."""
        # Stagnation if no improvement for many generations
        stag_limit = 10 + int(30 * self.dim / self.pop_size)
        if self.stagnation_counter > stag_limit:
            return True
        # Check if sigma collapsed
        if self.sigma < 1e-16:
            return True
        # Check if C has degenerate eigenvalues
        if np.max(self.D) / max(np.min(self.D), 1e-30) > 1e7:
            return True
        return False

    def _restart(self):
        """Restart the search with a new random state, keeping best solution."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        # Bias new mean towards best found so far with some probability
        if np.random.rand() < 0.3:
            perturbation = np.random.randn(self.dim) * 20.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)

    def _handle_truncated_fitness(self, population, fitness, y_vectors):
        """Handle case where func returns fewer values than requested."""
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            y_vectors = y_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, y_vectors

    def _inject_best_into_population(self, population, fitness, y_vectors):
        """Optionally inject the best-known solution into the population to preserve it."""
        if self.best_fitness < np.inf and len(population) > 0:
            worst_idx = np.argmax(fitness)
            if self.best_fitness < fitness[worst_idx]:
                population[worst_idx] = self.best_x.copy()
                fitness[worst_idx] = self.best_fitness
                y_vectors[worst_idx] = (self.best_x - self.mean) / max(self.sigma, 1e-30)
        return population, fitness, y_vectors

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_state()

        # Initial evaluation of mean to seed best tracking
        mean_fit = func(self.mean.reshape(1, self.dim))
        if len(mean_fit) > 0 and np.isfinite(mean_fit[0]):
            self.best_fitness = mean_fit[0]
            self.best_x = self.mean.copy()

        while not stopping_condition():
            # Update eigen decomposition if needed
            self._update_eigen_decomposition()

            # Sample new population
            population, y_vectors, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            # Evaluate fitness
            fitness = func(population)
            if stopping_condition():
                # Still process results we got
                population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            # Handle truncated results
            population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
            if len(fitness) < self.mu:
                # Not enough to do a meaningful update
                self._track_best(population, fitness, y_vectors)
                break

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)

            # Track best
            self._track_best(sorted_pop, sorted_fit)

            # Update mean
            old_mean, y_w = self._update_mean(sorted_pop, sorted_y)

            # Update evolution paths
            self._update_evolution_path_sigma(y_w)
            h_sigma = self._update_evolution_path_c(y_w)

            # Update covariance matrix
            self._update_covariance_matrix(sorted_y, h_sigma)

            # Update step size
            self._update_step_size()

            self.generation += 1

            # Detect and handle stagnation
            if self._detect_stagnation():
                self._restart()

        return self.best_fitness, self.best_x

```