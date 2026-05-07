**Idea: Monte Carlo Directional Explorer**
Using Latin Hypercube Sampling to generate diverse candidate mutation directions and Monte Carlo estimation to select among them stochastically, fundamentally replacing the deterministic ring topology with a sampling-driven exploration mechanism.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using Monte Carlo sampling:
        - Latin Hypercube Sampling for diverse candidate directions
        - Monte Carlo fitness estimation to rank candidates
        - Stochastic selection weighted by estimated improvement
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Parameters for MC sampling
        n_candidates = 6  # Number of candidate directions per individual
        n_probes = 3      # Monte Carlo probes per direction
        probe_scale = 0.3  # How far along direction to probe
        
        # Base F parameter
        current_F = np.clip(self.F * (1.0 + 0.2 * np.random.randn()), 0.1, 2.0)
        
        for i in range(np_pop):
            x_i = population[i]
            
            # Generate diverse candidate directions using Latin Hypercube Sampling
            candidate_dirs = np.zeros((n_candidates, dim))
            for d in range(dim):
                # Stratified sampling in [-1, 1] range
                u = (np.random.rand(n_candidates) + np.arange(n_candidates)) / n_candidates
                candidate_dirs[:, d] = (u * 2 - 1) * (1.0 + 0.5 * np.random.randn(n_candidates))
            
            # Add a few ring-topology based directions for diversity
            r1_idx, r2_idx = ring_prev[i], ring_next[i]
            if len(self.successful_directions) > 0:
                dir_stack = np.array(self.successful_directions)
                dir_norm = np.linalg.norm(dir_stack, axis=1, keepdims=True)
                dir_norm = np.maximum(dir_norm, 1e-10)
                successful_normed = dir_stack / dir_norm
                
                for k in range(min(2, len(successful_normed))):
                    cand_idx = n_candidates - 1 - k
                    if cand_idx >= 0:
                        candidate_dirs[cand_idx] = successful_normed[-(k+1)]
            
            # Monte Carlo evaluation: probe fitness along each candidate direction
            mc_scores = np.zeros(n_candidates)
            
            for c in range(n_candidates):
                direction = candidate_dirs[c]
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-10:
                    mc_scores[c] = fitness[i]  # Neutral direction
                    continue
                
                direction = direction / dir_norm
                
                # Sample points along this direction
                probe_points = np.zeros((n_probes, dim))
                for p in range(n_probes):
                    # Random step size in [0, probe_scale]
                    step = probe_scale * current_F * np.random.rand()
                    probe_points[p] = x_i + step * direction
                
                # Clip probes to bounds
                probe_points = np.clip(probe_points, self.lower, self.upper)
                
                # Evaluate probes (Monte Carlo sampling)
                probe_fitness = self.func(probe_points)
                if len(probe_fitness) < n_probes:
                    probe_fitness = np.resize(probe_fitness, n_probes)
                    probe_fitness[np.isnan(probe_fitness)] = np.inf
                
                # Use best probe fitness as direction quality estimate
                mc_scores[c] = np.min(probe_fitness)
            
            # Stochastic selection: lower score = better direction
            # Convert to selection probabilities using softmax
            score_weights = np.exp(-2.0 * (mc_scores - np.min(mc_scores) + 1e-10))
            probs = score_weights / np.sum(score_weights)
            
            # Sample selection index
            sel_idx = np.random.choice(n_candidates, p=probs)
            selected_dir = candidate_dirs[sel_idx]
            
            # Compute mutation vector with selected direction
            dir_norm = np.linalg.norm(selected_dir)
            if dir_norm < 1e-10:
                selected_dir = np.random.randn(dim)
                dir_norm = np.linalg.norm(selected_dir)
            
            selected_dir = selected_dir / dir_norm
            
            # Adaptive step size based on MC score quality
            step_factor = 1.0 + 0.5 * (mc_scores[sel_idx] - fitness[i])
            step_factor = np.clip(step_factor, 0.3, 2.0)
            
            trials[i] = x_i + current_F * step_factor * selected_dir
        
        return trials, current_F
```