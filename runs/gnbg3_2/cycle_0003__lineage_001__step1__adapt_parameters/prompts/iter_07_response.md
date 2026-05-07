**Idea: Bootstrap Thompson Sampling**
Uses bootstrap resampling to estimate uncertainty in success rates, Latin Hypercube sampling to explore the (F, Cr) parameter space, and Thompson sampling to probabilistically select parameters based on estimated performance — fundamentally stochastic where the current implementation is deterministic.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using bootstrap confidence estimation and Thompson sampling."""
    n = len(improved_mask)
    success_rate = np.mean(improved_mask)
    
    # Initialize state on first call
    if not hasattr(self, 'param_samples'):
        self.param_samples = []
        self.sample_perfs = []
        self.F_history = [self.F]
        self.Cr_history = [self.Cr]
        self.sample_gen = 0
    
    self.sample_gen += 1
    
    # Bootstrap confidence interval for success rate
    n_bootstrap = 200
    bootstrap_successes = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_successes[b] = np.mean(improved_mask[indices])
    
    # Use bootstrap mean and std for uncertainty quantification
    bootstrap_mean = np.mean(bootstrap_successes)
    bootstrap_std = np.std(bootstrap_successes) + 1e-10
    
    # Latin Hypercube Sampling of (F, Cr) parameter space
    n_lhs = 12
    f_min, f_max = 0.1, 1.5
    cr_min, cr_max = 0.1, 0.9
    
    # Generate LHS samples
    f_samples = np.linspace(f_min, f_max, n_lhs)
    cr_samples = np.linspace(cr_min, cr_max, n_lhs)
    np.random.shuffle(f_samples)
    np.random.shuffle(cr_samples)
    
    # Estimate performance for each LHS sample via bootstrap simulation
    # Higher F -> more exploration; higher Cr -> more recombination
    # Bootstrap std informs exploration: high uncertainty -> favor exploration
    exploration_factor = 1.0 + 2.0 * bootstrap_std  # Scale by uncertainty
    
    lhs_scores = np.zeros(n_lhs)
    for i in range(n_lhs):
        # Simulate expected performance based on current conditions
        f_sim = f_samples[i]
        cr_sim = cr_samples[i]
        
        # Exploration potential: moderate F (0.4-0.7) usually best
        f_exploit = np.exp(-0.5 * ((f_sim - 0.55) / 0.25) ** 2)
        # Recombination potential: moderate Cr (0.3-0.6) usually best
        cr_exploit = np.exp(-0.5 * ((cr_sim - 0.45) / 0.2) ** 2)
        
        # Adjust based on current success rate: low success -> more exploration
        if bootstrap_mean < 0.15:
            # Low success: favor exploration (lower F, varied Cr)
            lhs_scores[i] = (1 - f_sim) + 0.5 * cr_sim + 0.3 * exploration_factor
        elif bootstrap_mean > 0.35:
            # High success: favor exploitation (moderate F, higher Cr)
            lhs_scores[i] = f_exploit + cr_exploit + 0.1 * exploration_factor
        else:
            # Moderate: balanced
            lhs_scores[i] = 0.5 * (1 - np.abs(f_sim - 0.5)) + 0.5 * (1 - np.abs(cr_sim - 0.5))
    
    # Thompson sampling: sample from posterior over LHS configurations
    # Use softmax of scores as probability distribution
    score_temp = 3.0 + 5.0 * bootstrap_std  # Higher uncertainty -> softer distribution
    exp_scores = np.exp(score_temp * (lhs_scores - np.max(lhs_scores)))
    probs = exp_scores / (np.sum(exp_scores) + 1e-10)
    
    # Sample index proportional to probability
    idx = np.random.choice(n_lhs, p=probs)
    F_candidate = f_samples[idx]
    Cr_candidate = cr_samples[idx]
    
    # Add stochastic perturbation scaled by bootstrap uncertainty
    F_noise_scale = 0.1 * (1.0 + bootstrap_std * 3.0)
    Cr_noise_scale = 0.08 * (1.0 + bootstrap_std * 3.0)
    F_new = F_candidate + np.random.randn() * F_noise_scale
    Cr_new = Cr_candidate + np.random.randn() * Cr_noise_scale
    
    # Clip to valid ranges
    self.F = np.clip(F_new, 0.1, 1.5)
    self.Cr = np.clip(Cr_new, 0.1, 0.9)
    
    # Track samples for Thompson sampling history
    self.param_samples.append((self.F, self.Cr))
    self.sample_perfs.append(success_rate)
    
    # Maintain bounded history
    if len(self.param_samples) > 20:
        self.param_samples.pop(0)
        self.sample_perfs.pop(0)
    
    # Update parameter history
    self.F_history.append(self.F)
    self.Cr_history.append(self.Cr)
    if len(self.F_history) > 15:
        self.F_history.pop(0)
        self.Cr_history.pop(0)
    
    # When we have enough historical samples, use Thompson sampling from history
    if len(self.sample_perfs) >= 8:
        # Bootstrap from historical performance to get uncertainty
        n_hist_boot = 100
        hist_boot_means = np.zeros(n_hist_boot)
        for b in range(n_hist_boot):
            indices = np.random.choice(len(self.sample_perfs), size=len(self.sample_perfs), replace=True)
            hist_boot_means[b] = np.mean([self.sample_perfs[i] for i in indices])
        
        # Compute confidence: if bootstrap variance is low, trust historical best
        confidence = 1.0 / (1.0 + np.std(hist_boot_means) * 10)
        
        # Thompson sampling from historical parameter-performance pairs
        # Weight by performance (invert: lower error = higher fitness proxy)
        perfs = np.array(self.sample_perfs)
        # Convert success rates to pseudo-fitness (invert and normalize)
        pseudo_fitness = 1.0 / (perfs + 0.01)
        exp_fit = np.exp(pseudo_fitness * 2)
        ts_probs = exp_fit / np.sum(exp_fit)
        
        ts_idx = np.random.choice(len(self.param_samples), p=ts_probs)
        F_ts, Cr_ts = self.param_samples[ts_idx]
        
        # Blend Thompson sample with LHS sample based on confidence
        blend = 0.3 * confidence  # Low confidence -> more Thompson exploration
        self.F = np.clip(blend * F_ts + (1 - blend) * self.F, 0.1, 1.5)
        self.Cr = np.clip(blend * Cr_ts + (1 - blend) * self.Cr, 0.1, 0.9)
```