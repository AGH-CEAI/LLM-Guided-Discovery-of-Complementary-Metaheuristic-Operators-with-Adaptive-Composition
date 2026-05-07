**Idea: Fitness-Landscape + Spatial Centroid Hybrid**

Combine fitness-rank correlation analysis (D) with spatial centroid tracking (A). The weighting is data-driven: when population shows high fitness-distance correlation, favor exploitation (centroid-directed); when correlation breaks, favor exploration (archive/random). Both signals are tracked temporally for robustness.

```python
def _mutate_compass_batch(self, population, fitness):
    """Hybrid mutation combining fitness-rank landscape analysis with spatial centroid tracking."""
    NP, dim = population.shape
    
    # Compute fitness ranks (D: fitness-landscape signal)
    ranks = self._compute_fitness_ranking(fitness)
    best_idx = np.argmin(fitness)
    
    # Initialize temporal tracking for fitness-distance correlation
    if not hasattr(self, '_fdc_history'):
        self._fdc_history = []
        self._centroid_history = []
        self._fdc_ema = 0.5
    
    # Compute fitness-distance correlation (FDC) - key landscape indicator
    centroid = population.mean(axis=0)
    dists_to_centroid = np.linalg.norm(population - centroid, axis=1)
    if dists_to_centroid.std() > 1e-10:
        fdc = np.corrcoef(ranks, dists_to_centroid)[0, 1]
        if np.isnan(fdc):
            fdc = 0.0
    else:
        fdc = 0.0
    
    # Update EMA of FDC for temporal stability (F: temporal/dynamical)
    self._fdc_ema = 0.7 * self._fdc_ema + 0.3 * fdc
    self._fdc_history.append(self._fdc_ema)
    if len(self._fdc_history) > 15:
        self._fdc_history.pop(0)
    
    # Compute centroid velocity (how fast centroid is moving)
    self._centroid_history.append(centroid.copy())
    if len(self._centroid_history) > 5:
        self._centroid_history.pop(0)
    
    centroid_velocity = 0.0
    if len(self._centroid_history) >= 2:
        centroid_velocity = np.linalg.norm(self._centroid_history[-1] - self._centroid_history[0])
    
    # Data-driven switching: FDC > threshold means good landscape -> exploit
    # FDC < threshold means deceptive landscape -> explore more
    fdc_threshold = 0.3
    exploit_weight = np.clip((self._fdc_ema - fdc_threshold) / (1.0 - fdc_threshold + 1e-10), 0.1, 0.9)
    explore_weight = 1.0 - exploit_weight
    
    # Mechanism 1: Spatial centroid direction (A: geometric/spatial)
    # Direction toward centroid weighted by exploration need
    direction_to_centroid = centroid - population
    centroid_mutations = population + explore_weight * self.F * direction_to_centroid
    
    # Mechanism 2: Fitness-rank based direction (D: fitness-landscape)
    # Current-to-best direction scaled by exploit weight
    direction_to_best = population[best_idx] - population
    best_mutations = population + exploit_weight * self.F * direction_to_best
    
    # Generate candidate mutations from both mechanisms
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Additional candidates for scoring
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        # Candidate A: Centroid-weighted (exploration-biased)
        cand_A = centroid_mutations[i]
        # Score: how much this moves toward better fitness region
        score_A = ranks[i] - np.mean([ranks[j] for j in [r1, r2, r3]])
        
        # Candidate B: Best-weighted (exploitation-biased)
        cand_B = best_mutations[i]
        score_B = ranks[i] - ranks[best_idx]
        
        # Candidate C: Archive-based if available (diversity injection)
        cand_C = None
        score_C = -np.inf
        if hasattr(self, 'archive') and len(self.archive) > 0:
            archive_concat = np.vstack(self.archive)
            if len(archive_concat) > 0:
                archive_idx = np.random.randint(len(archive_concat))
                archive_donor = archive_concat[archive_idx]
                # C = i + F*(pbest_archive - r1) + F*(r2 - r3)
                pbest_archive = population[np.random.choice(np.argsort(fitness)[:max(1, NP//10)])]
                cand_C = population[i] + self.F * (pbest_archive - population[r1]) + self.F * (population[r2] - population[r3])
                score_C = ranks[i] - np.mean([ranks[j] for j in [r1, r2]])
        
        # Candidate D: Random direction for baseline comparison
        cand_D = population[r1] + self.F * (population[r2] - population[r3])
        score_D = ranks[i] - np.mean([ranks[r1], ranks[r2], ranks[r3]])
        
        # Select best candidate with noise injection based on confidence
        candidates = [(cand_A, score_A), (cand_B, score_B), (cand_D, score_D)]
        if cand_C is not None:
            candidates.append((cand_C, score_C))
        
        best_cand, best_score = max(candidates, key=lambda x: x[1])
        
        # Add scaled noise proportional to centroid velocity (more noise when population is drifting)
        noise_scale = 0.01 + 0.1 * centroid_velocity / (np.linalg.norm(centroid) + 1e-10)
        noise = np.random.randn(dim) * noise_scale
        mutants[i] = np.clip(best_cand + noise, -100.0, 100.0)
    
    return mutants
```