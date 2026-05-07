Looking at the data, all 24 tasks remain unsolved with errors ranging from 7.9e+01 down to 8.5e-05. The worst tasks (16, 17, 20, 23, 19) have errors in the 1e+01 to 7.9e+01 range, suggesting the algorithm gets trapped in local optima and the adaptive selection mechanism isn't responding aggressively enough.

**Analysis of the failure:**
- The current Thompson Sampling with Beta distributions is too conservative
- Rewards are log-scaled and incremental, making the system unresponsive to large changes
- When operators get "lucky" early and accumulate high alpha values, they dominate selection even when stuck
- For multi-modal tasks (10, 11, 16, 17, 18, 20, 23), the algorithm needs aggressive operator switching when progress stalls

**Key insight:** Replace the slow-reward Beta approach with a **softmax selection over absolute best-fitness per operator** — this directly tracks which operator has produced the best solution and selects proportionally, without the lag of incremental reward accumulation.

**Idea: Softmax Best-Fitness Selection**
Uses softmax probabilities over negative best-fitness values (lower fitness = better), with stagnation-triggered temperature annealing for aggressive exploration when stuck.

```python
def _select_operator_thompson(self):
    """Select operator using softmax over absolute best-fitness per operator."""
    # Track best fitness achieved by each operator
    if not hasattr(self, 'operator_best_fitness'):
        self.operator_best_fitness = np.full(self.num_operators, np.inf)
    
    # Update best fitness for current operator
    op = self.current_operator
    if self.f_opt < self.operator_best_fitness[op]:
        self.operator_best_fitness[op] = self.f_opt
    
    # Compute softmax probabilities from negative best fitness
    # Lower (more negative) is better, so use -best_fitness
    neg_best = -self.operator_best_fitness
    
    # Add small noise to break ties and ensure valid softmax input
    noise = np.random.uniform(0, 1e-10, self.num_operators)
    neg_best = neg_best + noise
    
    # Adaptive temperature: lower temp when stagnating (explore more)
    stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
    temperature = max(0.01, 1.0 - 0.9 * stagnation_ratio)
    
    # Compute softmax probabilities
    neg_best_shifted = neg_best - np.max(neg_best)  # for numerical stability
    exp_scores = np.exp(neg_best_shifted / temperature)
    probs = exp_scores / np.sum(exp_scores)
    
    # Sample operator based on probabilities
    cumsum = np.cumsum(probs)
    r = np.random.random()
    self.current_operator = int(np.searchsorted(cumsum, r))
    
    # Update selection counts
    self.operator_counts[self.current_operator] += 1
    
    return self.current_operator
```