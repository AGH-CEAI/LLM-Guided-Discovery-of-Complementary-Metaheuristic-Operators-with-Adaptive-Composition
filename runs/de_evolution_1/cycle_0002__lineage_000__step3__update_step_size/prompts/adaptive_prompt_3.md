Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_update_step_size` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.178068e-07            1.130685e-07            7.458465e-08            8.135935e-08            9.783823e-08            7.504352e-08            1.198916e-07            1.161395e-07            4.483302e-05            8.698506e-08            1.080573e-07            
1      2.263999e-07            2.347512e-07            1.227423e+02            2.428583e-07            2.237545e-07            2.323710e-07            2.402959e-07            2.339388e-07            1.310122e-04            2.798673e-07            2.474662e-07            
2      2.443520e-05            2.408251e-05            5.721269e+02            4.914873e-05            3.308527e+01            2.226410e-05            2.393994e-05            2.439208e-05            4.289093e+01            5.819270e+02            6.945115e+00            
3      3.006561e-04            2.972001e-04            2.546495e+02            3.487573e-04            7.001964e+00            2.858032e-04            3.396295e-04            3.418265e-04            3.407590e+01            2.540062e+02            3.479208e-04            
4      2.433439e-02            2.454519e-02            2.439424e-02            2.438254e-02            2.439568e-02            2.436855e-02            2.433815e-02            2.475124e-02            1.197076e-01            2.408358e-02            2.459221e-02            
5      1.000000e-08            1.000000e-08            1.572328e+09            1.949353e+04            9.130673e+06            2.945644e+06            1.000000e-08            1.000000e-08            5.176654e+03            1.741806e+09            3.997635e+06            
6      4.395617e-07            4.664690e-07            6.624640e+03            3.815311e+03            1.593972e+03            9.650494e+02            7.373247e-07            4.516854e-07            2.577978e+03            6.621778e+03            9.503993e+02            
7      1.349606e-05            1.352113e-05            3.257555e+02            2.312808e+01            2.985647e-05            1.265256e-05            1.351863e-05            1.379288e-05            2.802402e+00            3.294793e+02            1.319165e-05            
8      1.694929e-03            1.733492e-03            1.587879e-03            1.742614e-03            1.716095e-03            1.661696e-03            1.676663e-03            1.711858e-03            3.786748e-02            1.691364e-03            1.786894e-03            
9      4.685333e-04            4.661955e-04            3.267615e+02            3.320019e-03            6.537037e-04            5.282666e-04            4.923830e-04            4.959757e-04            1.569132e+02            3.379163e+02            9.767942e-04            
10     4.722265e+00            5.216904e+00            2.757933e+02            2.515546e+02            2.709997e+02            1.080734e+01            7.543606e+00            6.174894e+00            1.443624e+02            2.768063e+02            2.101161e-02            
11     5.600149e+00            4.926165e+00            1.108234e+03            9.509611e+02            1.064737e+03            2.492979e+02            1.720026e+01            9.404259e+00            8.360114e+02            1.086807e+03            3.580300e+02            
12     2.491334e+00            4.099871e+00            2.046494e+02            1.850857e+02            1.965717e+02            9.497753e+01            3.666613e+00            8.972989e+00            9.768580e+01            1.973658e+02            1.238297e+01            
13     1.356267e+00            1.599884e+00            1.647068e+00            1.539426e+00            1.620298e+00            1.244594e+00            1.089998e+00            1.558316e+00            3.967511e-02            1.465058e+00            1.856703e+00            
14     2.624043e+00            2.695515e+00            2.566252e+00            2.545752e+00            2.860066e+00            2.525131e+00            2.565822e+00            3.241635e+00            4.794589e-01            2.533470e+00            2.563034e+00            
15     1.200456e+00            1.134704e+00            1.102001e+00            1.194357e+00            1.201526e+00            1.175332e+00            1.196269e+00            1.252294e+00            6.184649e-01            1.211681e+00            1.248683e+00            
16     9.494040e+02            9.373637e+02            1.833909e+04            1.542669e+04            8.843703e+03            1.986650e+03            3.666289e+03            9.628181e+02            1.290039e+04            1.772209e+04            2.349894e+03            
17     3.120103e+00            3.120103e+00            1.589206e+05            1.410430e+05            1.672530e+05            1.803132e+04            2.385920e+03            3.120103e+00            1.289730e+05            1.607211e+05            1.258075e+04            
18     2.674445e+00            2.674394e+00            2.674491e+00            2.784636e+00            3.668802e+00            2.674887e+00            2.674228e+00            2.674665e+00            2.748005e+00            2.674559e+00            2.673761e+00            
19     3.630560e+01            3.515881e+01            3.938638e+02            3.868328e+02            3.688946e+02            2.417480e+02            2.577078e+02            1.985726e+02            3.641884e+02            3.976133e+02            2.110449e+02            
20     1.317250e+01            1.258256e+01            8.427172e+00            2.062110e+01            1.277719e+01            6.756451e+00            5.287723e+00            1.563543e+01            4.730651e-01            7.132182e+00            6.362561e+00            
21     4.453001e+00            4.559182e+00            4.517291e+00            4.590640e+00            4.632614e+00            4.564466e+00            4.555137e+00            4.607489e+00            4.469999e+00            4.573652e+00            4.599817e+00            
22     2.324794e+00            2.480792e+00            2.323717e+00            2.525800e+00            2.626054e+00            2.315012e+00            2.561452e+00            2.256372e+00            5.615240e+00            2.378962e+00            2.311484e+00            
23     6.212768e+00            6.723575e+00            1.053073e+02            1.157891e+01            1.077852e+01            6.795034e+00            6.235690e+00            7.012383e+00            6.527841e-01            1.047004e+02            5.926875e+00            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_02_idea_0.py  (error=7.458465e-08)
Task  1: variant_04_idea_0.py  (error=2.237545e-07)
Task  2: variant_05_idea_0.py  (error=2.226410e-05)
Task  3: variant_05_idea_0.py  (error=2.858032e-04)
Task  4: variant_09_idea_0.py  (error=2.408358e-02)
Task  5: original.py  (error=1.000000e-08)
Task  6: original.py  (error=4.395617e-07)
Task  7: variant_05_idea_0.py  (error=1.265256e-05)
Task  8: variant_02_idea_0.py  (error=1.587879e-03)
Task  9: variant_01_idea_0.py  (error=4.661955e-04)
Task 10: variant_10_idea_0.py  (error=2.101161e-02)
Task 11: variant_01_idea_0.py  (error=4.926165e+00)
Task 12: original.py  (error=2.491334e+00)
Task 13: variant_08_idea_0.py  (error=3.967511e-02)
Task 14: variant_08_idea_0.py  (error=4.794589e-01)
Task 15: variant_08_idea_0.py  (error=6.184649e-01)
Task 16: variant_01_idea_0.py  (error=9.373637e+02)
Task 17: variant_01_idea_0.py  (error=3.120103e+00)
Task 18: variant_10_idea_0.py  (error=2.673761e+00)
Task 19: variant_01_idea_0.py  (error=3.515881e+01)
Task 20: variant_08_idea_0.py  (error=4.730651e-01)
Task 21: original.py  (error=4.453001e+00)
Task 22: variant_07_idea_0.py  (error=2.256372e+00)
Task 23: variant_08_idea_0.py  (error=6.527841e-01)

WIN COUNTS:
  variant_01_idea_0.py: 5 wins
  variant_08_idea_0.py: 5 wins
  original.py: 4 wins
  variant_05_idea_0.py: 3 wins
  variant_02_idea_0.py: 2 wins
  variant_10_idea_0.py: 2 wins
  variant_04_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (5 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA update
        csa_factor = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)

        # Adaptive damping based on dimension and progress
        # When sigma is very large relative to search space, increase damping
        search_range = self.ub - self.lb
        relative_sigma = self.sigma / search_range

        # Extra damping when sigma is disproportionately large or small
        if relative_sigma > 0.5:
            extra_damp = 1.0 + 0.5 * np.log(relative_sigma / 0.5)
            csa_factor /= extra_damp

        # Fitness-aware component: when we're close to a good solution,
        # be more conservative with sigma increases but allow decreases
        if self.best_fitness < 1e-4 and self.best_fitness > 0:
            # Near optimum: reduce sigma increase rate
            if csa_factor > 0:
                # Scale down increases when close to optimum
                scale = max(0.1, min(1.0, np.log10(self.best_fitness + 1e-20) / (-4.0)))
                csa_factor *= scale

        # Apply the update with numerical safety
        # Clip the exponent to prevent extreme jumps
        csa_factor = np.clip(csa_factor, -0.5, 0.5)
        self.sigma *= np.exp(csa_factor)

        # Condition-number aware lower bound: if covariance is well-conditioned,
        # allow smaller sigma
        if np.min(self.D) > 0:
            cond = np.max(self.D) / np.min(self.D)
            if cond > 1e6:
                # Poor conditioning: don't let sigma get too small
                min_sigma = 1e-15
            else:
                min_sigma = 1e-20
        else:
            min_sigma = 1e-20

        self.sigma = np.clip(self.sigma, min_sigma, 1e6)
```

# --- From variant_02_idea_0.py (2 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA update
        csa_factor = np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))

        # Compute a fitness-aware damping modifier
        # When fitness is large (far from optimum), be more aggressive with sigma increases
        # and less aggressive with sigma decreases
        if self.best_fitness > 1e2:
            # Very far from optimum - heavily bias toward exploration
            if csa_factor < 1.0:
                # Dampen decreases: don't let sigma shrink too fast
                csa_factor = csa_factor ** 0.3
            else:
                # Amplify increases
                csa_factor = csa_factor ** 1.5
        elif self.best_fitness > 1e0:
            if csa_factor < 1.0:
                csa_factor = csa_factor ** 0.5
            else:
                csa_factor = csa_factor ** 1.3
        elif self.best_fitness > 1e-4:
            if csa_factor < 1.0:
                csa_factor = csa_factor ** 0.7
            else:
                csa_factor = csa_factor ** 1.1
        # else: near optimum, use standard CSA factor (no modification)

        self.sigma *= csa_factor

        # Adaptive sigma floor: prevent premature convergence based on fitness level
        # The further from target, the higher the minimum sigma should be
        if self.best_fitness > 1e2:
            sigma_floor = 5.0
        elif self.best_fitness > 1e1:
            sigma_floor = 1.0
        elif self.best_fitness > 1e0:
            sigma_floor = 0.1
        elif self.best_fitness > 1e-2:
            sigma_floor = 0.01
        elif self.best_fitness > 1e-4:
            sigma_floor = 1e-4
        elif self.best_fitness > 1e-6:
            sigma_floor = 1e-8
        else:
            sigma_floor = 1e-15

        self.sigma = max(self.sigma, sigma_floor)

        # Periodic sigma boosting for hard problems - every N generations,
        # if we're still far from target, inject a sigma boost
        if hasattr(self, 'generation') and self.generation > 0:
            boost_interval = max(10, 5 + self.dim)
            if self.generation % boost_interval == 0 and self.best_fitness > 1.0:
                # Check if sigma has collapsed relative to what's needed
                max_cov_scale = self.sigma * np.max(self.D) if np.max(self.D) > 0 else self.sigma
                domain_size = self.ub - self.lb  # 200
                if max_cov_scale < domain_size * 0.001:
                    # Sigma has collapsed but we're still far - boost it
                    self.sigma = max(self.sigma * 10.0, 1.0)

            # Also boost if stagnation is building up and fitness is bad
            if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 3:
                if self.best_fitness > 10.0:
                    self.sigma = max(self.sigma * 1.5, 2.0)
                elif self.best_fitness > 1.0:
                    self.sigma = max(self.sigma * 1.2, 0.5)

        # Hard bounds on sigma
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA ratio
        csa_ratio = p_sigma_norm / self.chi_n - 1.0

        # Adaptive damping based on fitness level
        bf = self.best_fitness if np.isfinite(self.best_fitness) else 1e10

        if bf > 1e2:
            # Far from optimum: reduce damping to allow larger sigma changes
            effective_d = self.d_sigma * 0.5
        elif bf > 1e0:
            effective_d = self.d_sigma * 0.7
        elif bf > 1e-4:
            effective_d = self.d_sigma * 0.85
        else:
            # Near optimum: increase damping slightly for stability
            effective_d = self.d_sigma * 1.1

        # Base update
        self.sigma *= np.exp((self.c_sigma / max(effective_d, 1e-10)) * csa_ratio)

        # Stagnation-triggered exploration boost
        # If stagnating and fitness is still large, push sigma upward
        if hasattr(self, 'stagnation_counter') and hasattr(self, 'best_fitness'):
            stag = self.stagnation_counter

            if bf > 1e2 and stag > 3:
                # Strong boost when far from optimum and stagnating
                boost = 1.0 + 0.05 * min(stag, 20)
                self.sigma *= boost
            elif bf > 1e0 and stag > 5:
                boost = 1.0 + 0.02 * min(stag, 15)
                self.sigma *= boost
            elif bf > 1e-2 and stag > 10:
                boost = 1.0 + 0.01 * min(stag, 10)
                self.sigma *= boost

        # Prevent sigma from collapsing too early when fitness is still large
        if bf > 1e0:
            min_sigma = 1e-3 * (1.0 + np.log10(max(bf, 1.0)))
            self.sigma = max(self.sigma, min_sigma)
        elif bf > 1e-4:
            self.sigma = max(self.sigma, 1e-8)

        # Anti-explosion: if sigma is getting too large relative to search space
        max_sigma = min(1e6, 50.0 * (self.ub - self.lb))

        self.sigma = np.clip(self.sigma, 1e-20, max_sigma)
```

# --- From variant_05_idea_0.py (3 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA update as baseline
        standard_update = np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))

        # Fitness-aware damping: reduce damping when far from optimum to allow bigger steps
        bf = self.best_fitness
        if np.isfinite(bf) and bf > 0:
            log_fitness = np.log10(max(bf, 1e-20))
            # Scale damping: less damping (faster adaptation) when fitness is high
            if log_fitness > 2:  # fitness > 100
                damping_factor = 0.7
            elif log_fitness > 0:  # fitness > 1
                damping_factor = 0.85
            elif log_fitness > -4:  # fitness > 1e-4
                damping_factor = 1.0
            else:
                damping_factor = 1.1  # More conservative near optimum
        else:
            damping_factor = 1.0

        # Apply damped update
        effective_update = np.exp((self.c_sigma / (self.d_sigma * damping_factor)) * (p_sigma_norm / self.chi_n - 1.0))

        self.sigma *= effective_update

        # Fitness-aware sigma floor: prevent sigma from getting too small when far from optimum
        if np.isfinite(bf):
            if bf > 100:
                sigma_floor = 1.0
            elif bf > 10:
                sigma_floor = 0.1
            elif bf > 1:
                sigma_floor = 0.01
            elif bf > 1e-2:
                sigma_floor = 1e-4
            elif bf > 1e-4:
                sigma_floor = 1e-6
            else:
                sigma_floor = 1e-20

            if self.sigma < sigma_floor:
                self.sigma = sigma_floor

        # Stagnation-based sigma boosting: if stuck at high error, nudge sigma up
        if hasattr(self, 'stagnation_counter') and np.isfinite(bf):
            stag = self.stagnation_counter
            if bf > 1.0 and stag > 5:
                # Gentle exponential boost proportional to stagnation
                boost = 1.0 + 0.02 * min(stag, 50)
                self.sigma *= boost
            elif bf > 1e-2 and stag > 15:
                boost = 1.0 + 0.01 * min(stag - 15, 30)
                self.sigma *= boost

        # Hard bounds
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA update
        csa_factor = np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))

        # Fitness-aware correction: if best_fitness is large and stagnating, boost sigma
        fitness_val = self.best_fitness if np.isfinite(self.best_fitness) else 1e10

        # Track fitness improvement rate
        if not hasattr(self, '_sigma_fitness_prev'):
            self._sigma_fitness_prev = fitness_val
            self._sigma_no_improve_count = 0
            self._sigma_gen_counter = 0

        self._sigma_gen_counter += 1

        # Check improvement every few generations
        check_interval = max(3, self.dim // 2)
        if self._sigma_gen_counter % check_interval == 0:
            rel_improv = (self._sigma_fitness_prev - fitness_val) / (abs(self._sigma_fitness_prev) + 1e-30)
            if rel_improv < 1e-8:
                self._sigma_no_improve_count += 1
            else:
                self._sigma_no_improve_count = max(0, self._sigma_no_improve_count - 2)
            self._sigma_fitness_prev = fitness_val

        # Biphasic behavior based on fitness level and stagnation
        if fitness_val > 1.0 and self._sigma_no_improve_count > 3:
            # Aggressive: inflate sigma to escape local optima
            boost = 1.0 + 0.3 * min(self._sigma_no_improve_count, 10) / 10.0
            csa_factor = max(csa_factor, boost)
            # Add stochastic perturbation to break symmetry
            noise = np.exp(0.1 * np.random.randn())
            csa_factor *= noise
        elif fitness_val > 1e-2 and self._sigma_no_improve_count > 5:
            # Moderate boost
            boost = 1.0 + 0.15 * min(self._sigma_no_improve_count, 10) / 10.0
            csa_factor = max(csa_factor, boost)
        elif fitness_val < 1e-4:
            # Near optimum: use dampened CSA for precision, slower decay
            dampening = 0.7  # slow down sigma decrease
            if csa_factor < 1.0:
                csa_factor = csa_factor ** dampening

        # Prevent sigma from collapsing too fast
        min_sigma_factor = 0.85  # never shrink by more than 15% per step
        csa_factor = max(csa_factor, min_sigma_factor)

        self.sigma *= csa_factor

        # Adaptive bounds based on fitness
        if fitness_val > 1.0:
            sigma_min, sigma_max = 1e-10, 1e8
        elif fitness_val > 1e-4:
            sigma_min, sigma_max = 1e-15, 1e6
        else:
            sigma_min, sigma_max = 1e-20, 1e4

        self.sigma = np.clip(self.sigma, sigma_min, sigma_max)
```

# --- From variant_08_idea_0.py (5 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA ratio
        csa_ratio = p_sigma_norm / self.chi_n - 1.0

        # Two-phase approach based on current best fitness
        bf = self.best_fitness if np.isfinite(self.best_fitness) else 1e10

        if bf > 1e2:
            # Phase 1: High error - aggressive exploration
            # Increase step size more readily, decrease slowly
            if csa_ratio > 0:
                # Want to grow: amplify the growth signal
                factor = np.exp((self.c_sigma / self.d_sigma) * csa_ratio * 1.5)
            else:
                # Want to shrink: dampen the shrink signal to maintain exploration
                factor = np.exp((self.c_sigma / self.d_sigma) * csa_ratio * 0.3)

            # Periodically inject sigma boost to escape local optima
            if hasattr(self, 'generation') and self.generation > 0 and self.generation % 15 == 0:
                if self.stagnation_counter > 3:
                    factor *= 1.5

        elif bf > 1e-2:
            # Phase 2: Medium error - balanced adaptation with slight exploration bias
            damping_mult = 1.0 + 0.5 * np.log10(max(bf, 1e-10) + 1e-10) / 10.0
            damping_mult = np.clip(damping_mult, 0.5, 2.0)
            factor = np.exp((self.c_sigma / (self.d_sigma * damping_mult)) * csa_ratio)

        else:
            # Phase 3: Low error - precise convergence with very careful sigma control
            # Use stronger damping to avoid overshooting the optimum
            enhanced_damping = self.d_sigma * (1.0 + 2.0 * max(0, -np.log10(max(bf, 1e-20))))
            factor = np.exp((self.c_sigma / enhanced_damping) * csa_ratio)

            # If making progress (stagnation_counter low), allow slightly faster convergence
            if self.stagnation_counter == 0:
                pass  # keep factor as is
            elif self.stagnation_counter > 10:
                # Slightly increase sigma to avoid premature convergence
                factor *= 1.02

        self.sigma *= factor

        # Adaptive sigma bounds based on current fitness regime
        if bf > 1e2:
            sigma_min, sigma_max = 1e-10, 1e8
        elif bf > 1e-2:
            sigma_min, sigma_max = 1e-15, 1e6
        else:
            sigma_min, sigma_max = 1e-20, 1e4

        self.sigma = np.clip(self.sigma, sigma_min, sigma_max)

        # Safety: if sigma becomes NaN or inf, reset
        if not np.isfinite(self.sigma):
            self.sigma = 30.0 if bf > 1e2 else (1.0 if bf > 1e-2 else 0.01)
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)

        # Standard CSA update as baseline
        csa_factor = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)

        current_fitness = self.best_fitness

        if current_fitness > 1e2:
            # Very far from optimum: heavily damped decrease, amplified increase
            if csa_factor > 0:
                # sigma wants to grow — let it grow faster
                self.sigma *= np.exp(csa_factor * 2.0)
            else:
                # sigma wants to shrink — resist shrinkage strongly
                self.sigma *= np.exp(csa_factor * 0.3)

            # Periodic pulse: every 10 generations, boost sigma
            if self.generation > 0 and self.generation % 10 == 0:
                self.sigma *= 1.5

        elif current_fitness > 1e0:
            # Moderate distance: slightly asymmetric
            if csa_factor > 0:
                self.sigma *= np.exp(csa_factor * 1.5)
            else:
                self.sigma *= np.exp(csa_factor * 0.5)

            # Less frequent pulse
            if self.generation > 0 and self.generation % 20 == 0:
                self.sigma *= 1.2

        elif current_fitness > 1e-4:
            # Getting closer: standard CSA with mild asymmetry
            self.sigma *= np.exp(csa_factor * 0.8)
        else:
            # Near optimum: standard CSA, tight control
            self.sigma *= np.exp(csa_factor)

        # Collapse detection: if eigenvalue spread is too narrow relative to
        # remaining error, re-inflate
        max_D = np.max(self.D)
        min_D = max(np.min(self.D), 1e-30)
        effective_radius = self.sigma * max_D

        if current_fitness > 1e-4 and effective_radius < 1e-8:
            # Search has collapsed but we're nowhere near the optimum
            self.sigma = max(self.sigma * 100.0, 1.0)
        elif current_fitness > 1e0 and effective_radius < 1e-3:
            self.sigma = max(self.sigma * 50.0, 5.0)
        elif current_fitness > 1e2 and effective_radius < 1.0:
            self.sigma = max(self.sigma * 20.0, 10.0)

        # Stagnation-aware boost: if stuck for many generations at large error
        if hasattr(self, 'stagnation_counter') and current_fitness > 1e0:
            stag_thresh = 3 + int(5 * self.dim / self.pop_size)
            if self.stagnation_counter > stag_thresh:
                boost = 1.0 + 0.1 * (self.stagnation_counter - stag_thresh)
                self.sigma *= min(boost, 5.0)

        self.sigma = np.clip(self.sigma, 1e-20, 1e8)
```

# --- From variant_10_idea_0.py (2 wins) ---
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        ratio = p_sigma_norm / self.chi_n

        # Asymmetric update: increase sigma faster than decrease
        # This combats premature convergence on multimodal landscapes
        if ratio > 1.0:
            # Steps are longer than expected -> increase sigma more aggressively
            asymmetry = 1.5
        else:
            # Steps are shorter than expected -> decrease sigma more gently
            asymmetry = 0.7

        # Fitness-aware damping: less damping when far from optimum (more aggressive adaptation)
        if hasattr(self, 'best_fitness') and np.isfinite(self.best_fitness):
            bf = max(self.best_fitness, 1e-30)
            if bf > 1e2:
                d_adapt = self.d_sigma * 0.5  # Very aggressive when far
            elif bf > 1e0:
                d_adapt = self.d_sigma * 0.75
            elif bf > 1e-4:
                d_adapt = self.d_sigma * 1.0
            else:
                d_adapt = self.d_sigma * 1.5  # More conservative near optimum
        else:
            d_adapt = self.d_sigma

        # Core CSA update with asymmetric factor
        update_factor = asymmetry * (self.c_sigma / d_adapt) * (ratio - 1.0)
        self.sigma *= np.exp(update_factor)

        # Periodic sigma boost when fitness is still large and sigma is small
        # This prevents the search from getting trapped in local optima
        if hasattr(self, 'best_fitness') and np.isfinite(self.best_fitness) and hasattr(self, 'generation'):
            bf = self.best_fitness
            gen = self.generation

            # Condition number check: if sigma collapsed but error is large, boost
            if self.sigma < 1.0 and bf > 1.0:
                boost = min(2.0, 1.0 + 0.1 * np.log10(max(bf, 1.0)))
                self.sigma *= boost

            # Every N generations, apply a small stochastic perturbation to sigma
            # to help escape saddle points and narrow valleys
            period = max(10, int(5 * self.dim / self.pop_size))
            if gen > 0 and gen % period == 0:
                if bf > 1e-2:
                    # Log-normal perturbation
                    self.sigma *= np.exp(0.1 * np.random.randn())

            # If eigenvalue spread suggests ill-conditioning and error is large,
            # inflate sigma along compressed directions via overall boost
            if np.min(self.D) > 0:
                cond = np.max(self.D) / np.min(self.D)
                if cond > 1e4 and bf > 1.0:
                    self.sigma *= 1.01  # Gentle persistent inflation

        self.sigma = np.clip(self.sigma, 1e-20, 1e8)
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
    A simplified CMA-ES-inspired optimizer with adaptive stagnation detection.
    Uses Thompson Sampling to select among three stagnation detection strategies.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))
        self.pop_size = max(self.pop_size, 8)
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()

        # Adaptive operator selection: Thompson Sampling for 3 stagnation strategies
        # 0 = variant_09 (patient near optimum, aggressive far)
        # 1 = variant_05 (adaptive with plateau detection)
        # 2 = variant_06 (window-based improvement rate)
        self.n_strategies = 3
        self.ts_alpha = np.ones(self.n_strategies, dtype=float)  # Beta prior successes
        self.ts_beta = np.ones(self.n_strategies, dtype=float)   # Beta prior failures
        self.current_strategy = 0
        self._strategy_fitness_start = np.inf  # fitness at start of current strategy epoch
        self._strategy_evals_start = 0
        self._strategy_epoch_length = 0

    def _initialize_strategy_params(self):
        dim = self.dim
        mu = self.mu

        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0
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
        self._fitness_history = []
        self._fitness_window = []

    def _update_eigen_decomposition(self):
        update_interval = max(1, int(1.0 / (self.c_1 + self.c_mu) / self.dim / 10.0))
        if self.generation - self.eigen_decomp_gen >= update_interval:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigenvalues, self.B = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.sqrt(eigenvalues)
            inv_D = 1.0 / self.D
            self.invsqrt_C = self.B @ np.diag(inv_D) @ self.B.T
            self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        y = z @ np.diag(self.D) @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, y, z

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, y_vectors):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, y_vectors
        sort_fit = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fit)
        return population[order], sort_fit[order], y_vectors[order]

    def _update_mean(self, sorted_pop, sorted_y):
        old_mean = self.mean.copy()
        self.mean = self.weights @ sorted_pop[:self.mu]
        y_w = self.weights @ sorted_y[:self.mu]
        return old_mean, y_w

    def _update_evolution_path_sigma(self, y_w):
        c_s = self.c_sigma
        self.p_sigma = (1.0 - c_s) * self.p_sigma + np.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (self.invsqrt_C @ y_w)

    def _update_evolution_path_c(self, y_w):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        h_sigma_threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if p_sigma_norm < h_sigma_threshold else 0.0

        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_w)
        return h_sigma

    def _update_covariance_matrix(self, sorted_y, h_sigma):
        dim = self.dim
        rank1 = np.outer(self.p_c, self.p_c)

        y_sel = sorted_y[:self.mu]
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_sel[i], y_sel[i])

        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        self.C = ((1.0 - self.c_1 - self.c_mu + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu * rank_mu)

    def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)

    def _track_best(self, sorted_pop, sorted_fit):
        if len(sorted_fit) == 0:
            return
        valid = np.isfinite(sorted_fit)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, sorted_fit, np.inf))
        if sorted_fit[best_idx] < self.best_fitness:
            self.best_fitness = float(sorted_fit[best_idx])
            self.best_x = sorted_pop[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    # ---- Three stagnation detection strategies ----

    def _detect_stagnation_strategy0(self):
        """Strategy 0: variant_09 - patient near optimum, aggressive far away."""
        if self.best_fitness < 1e-6:
            stag_limit = 200 + int(100 * self.dim / self.pop_size)
        elif self.best_fitness < 1e-2:
            stag_limit = 50 + int(50 * self.dim / self.pop_size)
        elif self.best_fitness < 1.0:
            stag_limit = 15 + int(20 * self.dim / self.pop_size)
        elif self.best_fitness < 100.0:
            stag_limit = 8 + int(10 * self.dim / self.pop_size)
        else:
            stag_limit = 5 + int(5 * self.dim / self.pop_size)

        if self.stagnation_counter > stag_limit:
            return True

        if self.sigma < 1e-16:
            if self.best_fitness > 1e-8:
                return True
            else:
                return False

        if self.sigma > 1e4:
            return True

        if np.min(self.D) > 0:
            cond = np.max(self.D) / np.min(self.D)
            cond_limit = 1e7 if self.best_fitness > 1e-4 else 1e10
            if cond > cond_limit:
                return True

        if self.generation > 0 and self.generation % (20 + int(30 * self.dim / self.pop_size)) == 0:
            if self.best_fitness > 10.0:
                if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) > 10:
                    recent = self.best_fitness_history[-10:]
                    if len(recent) >= 2 and (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-30) < 0.01:
                        return True

        if hasattr(self, 'best_fitness_history'):
            if len(self.best_fitness_history) == 0 or self.best_fitness_history[-1] != self.best_fitness:
                self.best_fitness_history.append(float(self.best_fitness))

        return False

    def _detect_stagnation_strategy1(self):
        """Strategy 1: variant_05 - adaptive with plateau detection."""
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(float(self.best_fitness))

        max_hist = 200
        if len(self._fitness_history) > max_hist:
            self._fitness_history = self._fitness_history[-max_hist:]

        if self.sigma < 1e-18:
            return True

        if self.sigma > 1e5:
            return True

        cond = np.max(self.D) / max(np.min(self.D), 1e-30)
        if cond > 1e8:
            return True

        if self.best_fitness < 1e-4:
            stag_limit = 50 + int(80 * self.dim / self.pop_size)
        elif self.best_fitness < 1.0:
            stag_limit = 20 + int(40 * self.dim / self.pop_size)
        else:
            stag_limit = 5 + int(15 * self.dim / self.pop_size)

        if self.stagnation_counter > stag_limit:
            return True

        window = min(30, len(self._fitness_history))
        if window >= 10:
            old_fit = self._fitness_history[-window]
            new_fit = self._fitness_history[-1]
            if old_fit > 0 and np.isfinite(old_fit) and np.isfinite(new_fit):
                rel_improvement = (old_fit - new_fit) / (abs(old_fit) + 1e-30)
                if rel_improvement < 1e-12 and self.stagnation_counter > stag_limit // 2:
                    return True

        max_step = self.sigma * np.max(self.D)
        if max_step < 1e-15 * (self.ub - self.lb):
            return True

        min_step = self.sigma * np.min(self.D)
        if min_step < 1e-20 and cond > 1e4:
            return True

        return False

    def _detect_stagnation_strategy2(self):
        """Strategy 2: variant_06 - window-based improvement rate."""
        if not hasattr(self, '_fitness_window'):
            self._fitness_window = []
        self._fitness_window.append(float(self.best_fitness))

        max_window = 200
        if len(self._fitness_window) > max_window:
            self._fitness_window = self._fitness_window[-max_window:]

        if self.sigma < 1e-18:
            self._fitness_window = []
            return True

        cond = np.max(self.D) / max(np.min(self.D), 1e-30)
        if cond > 1e8:
            self._fitness_window = []
            return True

        if self.sigma > 1e5:
            self._fitness_window = []
            return True

        current_error = float(self.best_fitness)
        if current_error > 1e2:
            check_window = max(8, int(5 + self.dim // 4))
        elif current_error > 1e0:
            check_window = max(12, int(10 + self.dim // 3))
        elif current_error > 1e-4:
            check_window = max(20, int(15 + self.dim // 2))
        else:
            check_window = max(30, int(20 + self.dim))

        if len(self._fitness_window) >= check_window:
            old_val = self._fitness_window[-check_window]
            new_val = self._fitness_window[-1]

            if old_val == 0 or not np.isfinite(old_val):
                rel_improvement = 0.0
            else:
                rel_improvement = (old_val - new_val) / (abs(old_val) + 1e-30)

            if current_error > 1e2:
                min_improvement = 1e-3
            elif current_error > 1e0:
                min_improvement = 1e-4
            elif current_error > 1e-4:
                min_improvement = 1e-6
            else:
                min_improvement = 1e-8

            if rel_improvement < min_improvement:
                self._fitness_window = []
                return True

        hard_limit = 5 + int(15 * self.dim / self.pop_size)
        if self.stagnation_counter > hard_limit:
            self._fitness_window = []
            return True

        return False

    def _select_strategy(self):
        """Thompson Sampling to select a stagnation detection strategy."""
        samples = np.array([
            np.random.beta(max(self.ts_alpha[i], 0.01), max(self.ts_beta[i], 0.01))
            for i in range(self.n_strategies)
        ])
        return int(np.argmax(samples))

    def _update_strategy_reward(self, strategy_idx, fitness_before, fitness_after, n_gens):
        """Credit assignment: reward strategy based on fitness improvement rate."""
        strategy_idx = int(strategy_idx)
        if not np.isfinite(fitness_before) or not np.isfinite(fitness_after):
            return
        if n_gens <= 0:
            return

        # Compute relative improvement
        improvement = float(fitness_before) - float(fitness_after)
        rel_improvement = improvement / (abs(float(fitness_before)) + 1e-30)

        # Convert to reward: positive improvement = success
        if rel_improvement > 1e-6:
            self.ts_alpha[strategy_idx] += 1.0
        elif rel_improvement > 1e-10:
            self.ts_alpha[strategy_idx] += 0.3
            self.ts_beta[strategy_idx] += 0.3
        else:
            self.ts_beta[strategy_idx] += 1.0

        # Decay to allow adaptation over time
        decay = 0.995
        self.ts_alpha *= decay
        self.ts_beta *= decay
        # Ensure minimum values
        self.ts_alpha = np.maximum(self.ts_alpha, 0.5)
        self.ts_beta = np.maximum(self.ts_beta, 0.5)

    def _detect_stagnation(self):
        """Adaptive stagnation detection: delegates to selected strategy."""
        strategies = [
            self._detect_stagnation_strategy0,
            self._detect_stagnation_strategy1,
            self._detect_stagnation_strategy2,
        ]
        result = strategies[self.current_strategy]()
        self._strategy_epoch_length += 1
        return result

    def _restart(self):
        """IPOP-style restart with strategy reward update."""
        saved_best_x = self.best_x.copy()
        saved_best_f = float(self.best_fitness)

        # Update reward for current strategy
        self._update_strategy_reward(
            self.current_strategy,
            self._strategy_fitness_start,
            saved_best_f,
            self._strategy_epoch_length
        )

        # Select new strategy for next epoch
        self.current_strategy = self._select_strategy()
        self._strategy_fitness_start = saved_best_f
        self._strategy_epoch_length = 0

        if not hasattr(self, '_restart_count'):
            self._restart_count = 0
            self._base_pop_size = self.pop_size
        self._restart_count += 1

        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1

        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2

        self._initialize_strategy_params()
        self._initialize_state()

        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        strategy = self._restart_count % 4

        if strategy == 0:
            self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
            self.sigma = 50.0
        elif strategy == 1:
            perturbation = np.random.randn(self.dim) * 40.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
            self.sigma = 60.0
        elif strategy == 2:
            center = np.zeros(self.dim)
            opposite = 2.0 * center - saved_best_x
            noise = np.random.randn(self.dim) * 10.0
            self.mean = np.clip(opposite + noise, self.lb, self.ub)
            self.sigma = 40.0
        else:
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
            self.sigma = 15.0

        if self._restart_count % 8 == 0:
            self._restart_count = 0

    def _handle_truncated_fitness(self, population, fitness, y_vectors):
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            y_vectors = y_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, y_vectors

    def _inject_best_into_population(self, population, fitness, y_vectors):
        if self.best_fitness < np.inf and len(population) > 0:
            worst_idx = np.argmax(fitness)
            if self.best_fitness < fitness[worst_idx]:
                population[worst_idx] = self.best_x.copy()
                fitness[worst_idx] = self.best_fitness
                y_vectors[worst_idx] = (self.best_x - self.mean) / max(self.sigma, 1e-30)
        return population, fitness, y_vectors

    def __call__(self, func, stopping_condition):
        self._initialize_state()

        # Initialize adaptive strategy tracking
        self.current_strategy = self._select_strategy()
        self._strategy_fitness_start = np.inf
        self._strategy_epoch_length = 0

        mean_fit = func(self.mean.reshape(1, self.dim))
        if len(mean_fit) > 0 and np.isfinite(mean_fit[0]):
            self.best_fitness = float(mean_fit[0])
            self.best_x = self.mean.copy()
            self._strategy_fitness_start = float(self.best_fitness)

        while not stopping_condition():
            self._update_eigen_decomposition()

            population, y_vectors, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            fitness = func(population)
            if stopping_condition():
                population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
            if len(fitness) < self.mu:
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
            self._track_best(sorted_pop, sorted_fit)

            old_mean, y_w = self._update_mean(sorted_pop, sorted_y)
            self._update_evolution_path_sigma(y_w)
            h_sigma = self._update_evolution_path_c(y_w)
            self._update_covariance_matrix(sorted_y, h_sigma)
            self._update_step_size()

            self.generation += 1

            if self._detect_stagnation():
                self._restart()

        return self.best_fitness, self.best_x

```