**Idea: LHS Bootstrap Compass with Probabilistic Estimation**

This variant uses Latin Hypercube Sampling (LHS) to generate a diverse set of mutation directions, then employs bootstrap-style resampling to probabilistically estimate the expected improvement of each direction. Direction selection is stochastic based on improvement probability, fundamentally different from the deterministic ranking-based approach in other variants.
```python
def _mutate_compass_batch(self, population, fitness):
    """Category G: LHS-sampled compass with bootstrap probability estimation."""
    NP, dim = self.dim if hasattr(self, 'dim') else (len(population), population.shape[1])
    NP = len(population)
    dim = population.shape[1]
    
    # Compute fitness ranks
    fitness_ranks = self._compute_fitness_ranking(fitness)
    
    # Number of LHS samples and bootstrap resamples
    n_directions = 8
    n_bootstrap = 12
    
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Generate LHS indices for diverse sampling
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        if len(others) < 4:
            mutants[i] = self._mutate_rand(population, i, self.F)
            continue
        
        # Latin Hypercube Sampling: generate n_directions unique 4-tuples
        index_sets = []
        for _ in range(n_directions * 3):
            indices = np.random.choice(others, 4, replace=False)
            index_sets.append(tuple(indices))
            if len(index_sets) >= n_directions:
                break
        index_sets = index_sets[:n_directions]
        
        if not index_sets:
            mutants[i] = self._mutate_rand(population, i, self.F)
            continue
        
        # For each LHS direction, estimate improvement via bootstrap
        direction_scores = []
        for r1, r2, r3, r4 in index_sets:
            # DE-style mutation vector
            v_base = population[r1] + self.F * (population[r2] - population[r3])
            
            # Bootstrap: sample n_bootstrap points along this direction
            bootstrap_improvements = []
            for _ in range(n_bootstrap):
                # Random interpolation factor (Sobol-like quasi-random via jitter)
                t = np.clip(np.random.beta(2, 5), 0.0, 1.0)
                # Add small noise to simulate fitness landscape uncertainty
                noise_scale = 0.1 * (1.0 + fitness.std())
                trial_point = v_base + t * self.F * (population[r4] - population[i])
                trial_point += np.random.randn(dim) * noise_scale
                
                # Estimate rank improvement (stochastic proxy)
                # Use distance-based probability as proxy for fitness improvement
                dist_to_current = np.linalg.norm(trial_point - population[i])
                dist_to_best = np.linalg.norm(trial_point - population[np.argmin(fitness)])
                rank_proxy = fitness_ranks[i] - np.random.uniform(0, 0.1) * dist_to_best / (dist_to_current + 1e-8)
                bootstrap_improvements.append(max(0, fitness_ranks[i] - rank_proxy))
            
            # Bootstrap mean and variance
            mean_imp = np.mean(bootstrap_improvements)
            std_imp = np.std(bootstrap_improvements) + 1e-8
            
            # Thompson sampling: sample from estimated distribution
            sampled_score = np.random.normal(mean_imp, std_imp)
            direction_scores.append(sampled_score)
        
        # Select direction using softmax probabilities (stochastic)
        direction_scores = np.array(direction_scores)
        exp_scores = np.exp(direction_scores - direction_scores.max())
        probs = exp_scores / exp_scores.sum()
        
        # Stochastic selection based on probability
        selected_idx = np.random.choice(len(index_sets), p=probs)
        r1, r2, r3, r4 = index_sets[selected_idx]
        
        # Final mutation with small random perturbation
        mutants[i] = population[r1] + self.F * (population[r2] - population[r3]) + \
                     np.random.randn(dim) * 0.05 * self.F
    
    return mutants
```