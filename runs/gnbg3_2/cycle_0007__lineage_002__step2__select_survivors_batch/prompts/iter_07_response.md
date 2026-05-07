**Idea: Bootstrap Probabilistic Selection with Random Subspace Probes**

Category G: Uses Monte Carlo bootstrap resampling of population fitness and random subspace projections to estimate the probability that a trial solution would improve the population. This stochastic acceptance criterion adapts selection pressure based on bootstrapped confidence, rather than using deterministic spectral/geometry measures.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Bootstrap probabilistic selection with random subspace probes.
    
    Uses Monte Carlo sampling to estimate P(trial improves population)
    and random projections to probe solution quality in subspaces.
    """
    n_pop = len(population)
    n_bootstrap = 50  # Bootstrap resamples
    n_projections = 10  # Random subspace probes
    
    # --- Bootstrap estimation of improvement probability ---
    # Resample population indices and compare trial fitness to resampled pop
    improved_counts = np.zeros(n_pop)
    improvement_amounts = np.zeros(n_pop)
    
    for b in range(n_bootstrap):
        # Stratified resampling: sample each population member once per bootstrap
        resample_idx = np.random.choice(n_pop, size=n_pop, replace=True)
        resampled_fitness = fitness[resample_idx]
        
        # Count improvements and accumulate improvement amounts
        improved_counts += (trial_fitness < resampled_fitness).astype(float)
        improvement_amounts += np.maximum(0, resampled_fitness - trial_fitness)
    
    # Probability that trial improves a random population member
    improve_prob = improved_counts / n_bootstrap
    
    # Average improvement conditioned on being positive (avoid division by zero)
    pos_mask = improvement_amounts > 1e-15
    avg_improvement = np.zeros(n_pop)
    avg_improvement[pos_mask] = improvement_amounts[pos_mask] / np.maximum(1, improved_counts[pos_mask])
    
    # --- Random subspace probes ---
    # Project both population and trials into random subspaces to probe quality
    dim = population.shape[1]
    subspace_dim = max(dim // 4, 2)  # Use 1/4 of dimensions per probe
    
    pop_subspace_scores = np.zeros(n_pop)
    trial_subspace_scores = np.zeros(n_pop)
    
    for p in range(n_projections):
        # Random projection matrix (orthonormal)
        proj = np.random.randn(dim, subspace_dim)
        Q, _ = np.linalg.qr(proj)
        
        # Project into subspace
        pop_proj = population @ Q
        trial_proj = trials @ Q
        
        # Compute distance to centroid in subspace
        centroid = np.mean(pop_proj, axis=0)
        pop_dists = np.linalg.norm(pop_proj - centroid, axis=1)
        trial_dists = np.linalg.norm(trial_proj - centroid, axis=1)
        
        # Score: closer to centroid in subspace = more "representative"
        pop_subspace_scores += pop_dists
        trial_subspace_scores += trial_dists
    
    pop_subspace_scores /= n_projections
    trial_subspace_scores /= n_projections
    
    # Trials with lower subspace distance are in more "central" regions
    subspace_bonus = (pop_subspace_scores - trial_subspace_scores) / (pop_subspace_scores + 1e-10)
    subspace_bonus = np.clip(subspace_bonus, -2, 2)
    
    # --- Combined stochastic acceptance score ---
    # Fitness-based component: scaled improvement
    improvement = fitness - trial_fitness
    fit_range = np.ptp(fitness)
    fit_range = max(fit_range, 1e-10)
    scaled_improvement = improvement / fit_range
    
    # Bootstrap-based component: probabilistic improvement
    # Use log-odds transformation for stable combination
    log_odds = np.log(improve_prob + 1e-10) - np.log(1 - improve_prob + 1e-10)
    bootstrap_bonus = 0.5 * log_odds
    
    # Subspace bonus: favor trials in well-explored central regions
    # (reduces selection pressure in sparse regions)
    
    # Combined score: weighted sum of all components
    combined_score = (
        0.6 * np.clip(scaled_improvement, -1, 1) +
        0.25 * bootstrap_bonus +
        0.15 * subspace_bonus
    )
    
    # --- Probabilistic acceptance ---
    # Accept if deterministic improvement OR with probability based on combined score
    improved_mask = trial_fitness < fitness
    
    # Soft acceptance threshold: use sigmoid on combined score
    # Score > 0.2 → high acceptance; Score < -0.2 → low acceptance
    sigmoid_threshold = 1.0 / (1.0 + np.exp(-10 * combined_score))
    
    # Accept improved solutions with high probability, others probabilistically
    accept_prob = np.where(improved_mask, 
                          0.95 + 0.05 * sigmoid_threshold,
                          0.3 * sigmoid_threshold)
    
    # Add bootstrap confidence adjustment
    # High bootstrap variance → more conservative (reduce acceptance)
    bootstrap_variance = improve_prob * (1 - improve_prob) / n_bootstrap
    confidence_factor = 1.0 - 0.5 * np.sqrt(bootstrap_variance + 1e-10)
    accept_prob *= confidence_factor
    
    # Stochastic draw
    random_draw = np.random.rand(n_pop)
    accept_mask = random_draw < accept_prob
    
    # Always accept if significantly improved (safety net)
    accept_mask = accept_mask | improved_mask
    
    # Build new population
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]
    
    return new_population, new_fitness, accept_mask
```