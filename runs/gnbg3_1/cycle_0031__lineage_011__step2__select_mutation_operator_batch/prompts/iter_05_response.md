**Idea: Diversity-Triggered Exploration with Fitness Landscape Awareness**

Instead of UCB-based greedy selection, use a fundamentally different approach: dynamically adjust operator probabilities based on measured population diversity and fitness landscape features. When diversity is low (stuck in local optima), force exploration-heavy operators; when making progress, allow exploitation. This directly targets the worst tasks (16, 17, 19, 23, etc.) which are stuck at errors 10-1000× higher than target, indicating severe multimodality/deception.

```python
def _select_mutation_operator_batch(self):
    """Select mutation operator using diversity-triggered exploration strategy."""
    # Compute population diversity using fitness-based entropy
    sorted_fit = np.sort(self.fitness)
    fit_range = sorted_fit[-1] - sorted_fit[0] + 1e-10
    
    # Fitness distribution entropy (high entropy = diverse population)
    bins = 10
    hist, _ = np.histogram(self.fitness, bins=bins, range=(sorted_fit[0], sorted_fit[-1] + 1e-10))
    hist = hist / (len(self.fitness) + 1e-10)
    hist = hist[hist > 0]  # Remove zeros for log
    fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
    max_entropy = np.log(bins)
    normalized_entropy = fitness_entropy / (max_entropy + 1e-10)
    
    # Position diversity: average pairwise distance normalized
    pop_std = np.std(self.population, axis=0)
    position_diversity = np.mean(pop_std) / (self.upper_bound - self.lower_bound + 1e-10)
    
    # Combined diversity score
    diversity_score = 0.6 * normalized_entropy + 0.4 * position_diversity
    
    # Track diversity trend
    if not hasattr(self, 'diversity_history'):
        self.diversity_history = []
    self.diversity_history.append(diversity_score)
    if len(self.diversity_history) > 10:
        self.diversity_history.pop(0)
    
    # Compute diversity trend (positive = recovering, negative = declining)
    if len(self.diversity_history) >= 3:
        diversity_trend = (self.diversity_history[-1] - self.diversity_history[0]) / max(self.diversity_history[0], 1e-10)
    else:
        diversity_trend = 0.0
    
    # Stagnation detection
    if len(self.best_history) >= 5:
        recent_improvement = self.best_history[-1] - self.best_history[0]
        is_stagnant = abs(recent_improvement) < 1e-6
    else:
        is_stagnant = False
    
    # Fitness ratio for landscape analysis
    best_fit = np.min(self.fitness)
    worst_fit = np.max(self.fitness)
    fit_ratio = best_fit / (worst_fit + 1e-10)
    
    # Define operator pool with exploration/exploitation labels
    # 0: rand/1 (exploration), 1: current-to-pbest (balanced), 
    # 2: best/1 (exploitation), 3: rand-to-best (balanced-exploit)
    operators = ['rand', 'current_to_pbest', 'best', 'rand_to_best']
    
    # Determine exploration vs exploitation weights based on state
    if is_stagnant or diversity_score < 0.05:
        # CRITICAL: Stuck in local optima - force exploration
        explore_weight = 0.7
        exploit_weight = 0.3
    elif diversity_trend < -0.1 and diversity_score < 0.15:
        # Declining diversity - increase exploration
        explore_weight = 0.55
        exploit_weight = 0.45
    elif diversity_score > 0.25 and fit_ratio > 0.1:
        # Good diversity and making progress - allow exploitation
        explore_weight = 0.3
        exploit_weight = 0.7
    else:
        # Normal operation - moderate balance
        explore_weight = 0.45
        exploit_weight = 0.55
    
    # Operator probabilities: [rand, current_to_pbest, best, rand_to_best]
    # Exploration operators: rand (index 0)
    # Balanced: current_to_pbest (1), rand_to_best (3)
    # Exploitation: best (2)
    probs = np.array([
        explore_weight,           # rand - exploration
        0.3 * exploit_weight,     # current_to_pbest - balanced
        0.4 * exploit_weight,     # best - exploitation
        0.3 * exploit_weight      # rand_to_best - balanced
    ])
    
    # Normalize
    probs = probs / np.sum(probs)
    
    # Sample one operator per individual with diversity-aware perturbation
    selected_operator = np.random.choice(len(operators), p=probs)
    
    # Small random perturbation to prevent deterministic cycling
    perturbation = np.random.randint(-1, 2)
    selected_operator = np.clip(selected_operator + perturbation, 0, len(operators) - 1)
    
    return selected_operator
```