Looking at the data, I need to fundamentally change how the population is initialized to break through the massive errors on tasks 17, 11, 16, etc. (errors stuck at 10-2000). The key insight: **extreme multi-modality and deception** require population diversity across multiple scales, not just uniform coverage.

**Idea: Multi-Scale Quasi-Random Initialization with Boundary Emphasis**
Use Halton quasi-random sequences with logarithmic scaling and intentional boundary concentration to ensure the population captures both fine-grained and coarse-grained structure, which is critical for escaping deceptive local optima.

```python
def _initialize_population(self):
    """Initialize population using multi-scale quasi-random sampling with boundary emphasis."""
    from numpy.linalg import LinAlgError
    
    # Use Halton quasi-random sequence for better space-filling than LHS
    def halton(n, base):
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0
            r = 0.0
            idx = i + 1
            while idx > 0:
                f = f / base
                r = r + f * (idx % base)
                idx = idx // base
            seq[i] = r
        return seq
    
    # Multi-base Halton for different dimensions
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    samples = np.zeros((self.NP, self.dim))
    
    for d in range(self.dim):
        base = bases[d % len(bases)]
        samples[:, d] = halton(self.NP, base)
    
    # Apply multi-scale transformation: 40% log-scale (fine), 40% linear, 20% boundary-focused
    n_log = self.NP // 2
    n_lin = self.NP - n_log - self.NP // 5
    n_bound = self.NP // 5
    
    # Log-scale region (for fine structure exploration)
    log_samples = samples[:n_log]
    log_range = np.log1p(self.ub - self.lb)
    scaled = np.expm1(log_samples[:n_log] * log_range)
    samples[:n_log] = self.lb + scaled
    
    # Linear region (standard coverage)
    lin_samples = samples[n_log:n_log + n_lin]
    samples[n_log:n_log + n_lin] = self.lb + lin_samples * (self.ub - self.lb)
    
    # Boundary-focused region (escape local optima near edges)
    bound_idx = n_log + n_lin
    bound_samples = samples[bound_idx:]
    half_n = len(bound_samples) // 2
    
    # Concentrate near boundaries with high variance
    eps = 1e-6
    range_vec = self.ub - self.lb
    for i in range(half_n):
        # Near lower boundary
        t = bound_samples[i] ** 0.25  # Concentrate near 0
        samples[bound_idx + i] = self.lb + eps + t * (0.1 * range_vec)
    for i in range(half_n, len(bound_samples)):
        # Near upper boundary
        t = (1.0 - bound_samples[i]) ** 0.25  # Concentrate near 1
        samples[bound_idx + i] = self.ub - eps - t * (0.1 * range_vec)
    
    # Add small perturbations for diversity
    perturbation_scale = 0.01 * (self.ub - self.lb)
    samples += np.random.randn(self.NP, self.dim) * perturbation_scale
    
    # Final bounds clipping
    samples = np.clip(samples, self.lb, self.ub)
    
    self.population = samples
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    # Robust covariance estimation with eigenvalue bounds
    C = np.cov(self.population.T)
    eigvals = np.linalg.eigvalsh(C)
    min_eig = np.min(eigvals)
    if min_eig < 1e-10:
        C += (1e-7 - min_eig) * np.eye(self.dim)
    self.C = C
    
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.sigma = np.clip(self.sigma, 1e-6, (self.ub[0] - self.lb[0]) / 3.0)
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
```